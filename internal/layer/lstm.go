// Package layer provides neural network layer implementations.
package layer

import (
	"fmt"
	"math"
	"sync"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

// maxGradientNorm for gradient clipping
const maxGradientNorm = 1.0

// LSTM is a Long Short-Term Memory layer optimized for performance.
// Uses contiguous memory layout with pre-allocated buffers for minimal allocations.
type LSTM struct {
	// Parameters
	inSize  int
	outSize int

	// Weight matrices stored contiguously for cache efficiency
	// Layout: [input_gate, forget_gate, cell_gate, output_gate]
	// Each gate has [input_weights, recurrent_weights]
	// Total: 8 weight blocks
	params           []float32 // Contiguous weights + biases
	inputWeights     []float32 // inSize * outSize * 4
	recurrentWeights []float32 // outSize * outSize * 4
	biases           []float32 // outSize * 4

	// Activation functions for each gate
	inputAct  activations.Activation
	forgetAct activations.Activation
	cellAct   activations.Activation
	outputAct activations.Activation

	// Reusable buffers - pre-allocated to avoid garbage collection
	inputBuf  []float32 // input vector
	outputBuf []float32 // output vector (hidden state)
	preActBuf []float32 // pre-activations for all 4 gates (outSize * 4)
	cellBuf   []float32 // current cell state
	hiddenBuf []float32 // current hidden state

	// Gradient buffers
	grads                []float32 // Contiguous grads
	gradInputWeights     []float32
	gradRecurrentWeights []float32
	gradBiases           []float32

	// Delta buffers for backprop (one per gate)
	dInputBuf  []float32
	dForgetBuf []float32
	dCellBuf   []float32
	dOutputBuf []float32

	// Reusable buffers for backward pass to avoid allocations
	dcBuf      []float32
	dxBuf      []float32
	dhPrevBuf  []float32
	cPrevBuf   []float32
	hPrevBuf   []float32

	// Gate outputs stored for backprop (one per gate)
	inputGateOut  []float32
	forgetGateOut []float32
	cellGateOut   []float32
	outputGateOut []float32

	// Saved states for BPTT
	savedCellOffsets   []int
	savedHiddenOffsets []int
	savedPreActOffsets []int
	arenaPtr           *[]float32
	arenaMu            sync.Mutex // Protege operações de arena contra race conditions

	// Current time step
	timeStep int

	// Persistent gradient through time for cell state
	dcNextBuf []float32

	// Device for computation
	device Device

	training bool

	// GPU persistent buffers
	bufWeightsIn  *MetalBuffer
	bufWeightsRec *MetalBuffer
	bufBiases     *MetalBuffer
	bufPreAct     *MetalBuffer
	bufC          *MetalBuffer
	bufH          *MetalBuffer
	bufI          *MetalBuffer
	bufF          *MetalBuffer
	bufG          *MetalBuffer
	bufO          *MetalBuffer
	bufX          *MetalBuffer
}

// NewLSTM creates a new LSTM layer with pre-allocated buffers.
// Returns an error if parameters are invalid.
func NewLSTM(inSize, outSize int) (*LSTM, error) {
	return NewLSTMWithDevice(inSize, outSize, &CPUDevice{})
}

// NewLSTMWithDevice creates a new LSTM layer with a specific device.
// Returns an error if parameters are invalid.
func NewLSTMWithDevice(inSize, outSize int, device Device) (*LSTM, error) {
	if inSize <= 0 && inSize != -1 {
		return nil, fmt.Errorf("invalid inSize %d: must be > 0 or -1", inSize)
	}
	if outSize <= 0 {
		return nil, fmt.Errorf("invalid outSize %d: must be > 0", outSize)
	}

	l := &LSTM{
		inSize:  inSize,
		outSize: outSize,
		device:  device,

		inputAct:  activations.Sigmoid{},
		forgetAct: activations.Sigmoid{},
		cellAct:   activations.Tanh{},
		outputAct: activations.Sigmoid{},

		savedCellOffsets:   make([]int, 0, 16),
		savedHiddenOffsets: make([]int, 0, 16),
		savedPreActOffsets: make([]int, 0, 16),
		timeStep:           0,
	}

	if inSize != -1 {
		if err := l.Build(inSize); err != nil {
			return nil, err
		}
	}

	return l, nil
}

// Build initializes the layer with the given input size.
// Returns error if overflow is detected in buffer size calculations.
func (l *LSTM) Build(inSize int) error {
	l.inSize = inSize
	outSize := l.outSize

	if inSize <= 0 || outSize <= 0 {
		return fmt.Errorf("invalid dimensions: inSize=%d, outSize=%d", inSize, outSize)
	}

	// Check for overflow in weight calculations using math/bits
	// weightInSize = outSize * 4 * inSize
	// Primeiro: gateSize = outSize * 4
	gateSize, err := CheckOverflow(outSize, 4)
	if err != nil {
		return fmt.Errorf("gate size overflow (outSize * 4): %w", err)
	}
	// Depois: weightInSize = gateSize * inSize
	weightInSize, err := CheckOverflow(gateSize, inSize)
	if err != nil {
		return fmt.Errorf("weight input size overflow: %w", err)
	}

	// weightRecSize = outSize * 4 * outSize = gateSize * outSize
	weightRecSize, err := CheckOverflow(gateSize, outSize)
	if err != nil {
		return fmt.Errorf("weight recurrent size overflow: %w", err)
	}

	// biasSize = outSize * 4 = gateSize
	biasSize := gateSize

	// totalParams = weightInSize + weightRecSize + biasSize
	totalParams, err := CheckOverflowSum(weightInSize, weightRecSize, biasSize)
	if err != nil {
		return fmt.Errorf("total params overflow: %w", err)
	}

	// Xavier/Glorot initialization for LSTM with 4 gates
	inputScale := float32(math.Sqrt(2.0 / float64(inSize+4*outSize)))
	recurrentScale := float32(math.Sqrt(2.0 / float64(outSize+4*outSize)))
	l.params = make([]float32, totalParams)

	l.inputWeights = l.params[:weightInSize]
	l.recurrentWeights = l.params[weightInSize : weightInSize+weightRecSize]
	l.biases = l.params[weightInSize+weightRecSize:]

	rng := NewRNG(uint64(inSize*1000 + outSize*100 + 142))
	for i := 0; i < outSize*4; i++ {
		if i >= outSize && i < outSize*2 {
			l.biases[i] = 1.0
		}
		for j := 0; j < inSize; j++ {
			l.inputWeights[i*inSize+j] = rng.RandFloat()*2*inputScale - inputScale
		}
		for j := 0; j < outSize; j++ {
			l.recurrentWeights[i*outSize+j] = rng.RandFloat()*2*recurrentScale - recurrentScale
		}
	}

	l.grads = make([]float32, totalParams)
	l.gradInputWeights = l.grads[:weightInSize]
	l.gradRecurrentWeights = l.grads[weightInSize : weightInSize+weightRecSize]
	l.gradBiases = l.grads[weightInSize+weightRecSize:]

	// Validate buffer sizes before allocation
	if inSize < 0 || outSize < 0 {
		return fmt.Errorf("invalid buffer dimensions: inSize=%d, outSize=%d", inSize, outSize)
	}

	// Check preActBuf size for overflow
	preActSize, err := CheckOverflow(outSize, 4)
	if err != nil {
		return fmt.Errorf("preActBuf size overflow: %w", err)
	}

	l.inputBuf = make([]float32, inSize)
	l.outputBuf = make([]float32, outSize)
	l.preActBuf = make([]float32, preActSize)
	l.cellBuf = make([]float32, outSize)
	l.hiddenBuf = make([]float32, outSize)

	l.dInputBuf = make([]float32, outSize)
	l.dForgetBuf = make([]float32, outSize)
	l.dCellBuf = make([]float32, outSize)
	l.dOutputBuf = make([]float32, outSize)

	l.dcBuf = make([]float32, outSize)
	l.dcNextBuf = make([]float32, outSize)
	l.dxBuf = make([]float32, inSize)
	l.dhPrevBuf = make([]float32, outSize)
	l.cPrevBuf = make([]float32, outSize)
	l.hPrevBuf = make([]float32, outSize)

	l.inputGateOut = make([]float32, outSize)
	l.forgetGateOut = make([]float32, outSize)
	l.cellGateOut = make([]float32, outSize)
	l.outputGateOut = make([]float32, outSize)

	if metal, ok := l.device.(*MetalDevice); ok && metal.IsAvailable() {
		l.bufWeightsIn = metal.CreateBuffer(l.inputWeights)
		l.bufWeightsRec = metal.CreateBuffer(l.recurrentWeights)
		l.bufBiases = metal.CreateBuffer(l.biases)
		l.bufPreAct = metal.CreateEmptyBuffer(preActSize)
		l.bufC = metal.CreateEmptyBuffer(outSize)
		l.bufH = metal.CreateEmptyBuffer(outSize)
		l.bufI = metal.CreateEmptyBuffer(outSize)
		l.bufF = metal.CreateEmptyBuffer(outSize)
		l.bufG = metal.CreateEmptyBuffer(outSize)
		l.bufO = metal.CreateEmptyBuffer(outSize)
		l.bufX = metal.CreateEmptyBuffer(inSize)
	}

	return nil
}

// Reset resets the LSTM state for a new sequence.
func (l *LSTM) Reset() {
	l.timeStep = 0
	l.savedCellOffsets = l.savedCellOffsets[:0]
	l.savedHiddenOffsets = l.savedHiddenOffsets[:0]
	l.savedPreActOffsets = l.savedPreActOffsets[:0]
	// Clear buffers
	for i := range l.cellBuf {
		l.cellBuf[i] = 0
	}
	for i := range l.hiddenBuf {
		l.hiddenBuf[i] = 0
	}
	for i := range l.dcNextBuf {
		l.dcNextBuf[i] = 0
	}
}

func (l *LSTM) ForwardWithArena(x []float32, arena *[]float32, offset *int) ([]float32, error) {
	// Copy input to buffer
	copy(l.inputBuf, x)

	inputStart, forgetStart, cellStart, outputStart := 0, l.outSize, l.outSize*2, l.outSize*3

	// === Compute pre-activations ===
	// Initialize with biases
	copy(l.preActBuf, l.biases)

	if metal, ok := l.device.(*MetalDevice); ok && metal.IsAvailable() && l.bufWeightsIn != nil {
		// 1. Update dynamic GPU buffers
		l.bufX.Update(x)
		if l.timeStep == 0 {
			l.bufC.Update(l.cellBuf) // cellBuf is cleared in Reset()
			l.bufH.Update(l.hiddenBuf)
		}

		// 2. Wx + b -> preAct
		// We use bufPreAct initialized with Biases
		l.bufPreAct.Update(l.biases)
		metal.MatMulPersistent(l.bufWeightsIn, l.bufX, l.bufPreAct, l.outSize*4, 1, l.inSize, 1.0)

		// 3. Wh + preAct -> preAct
		metal.MatMulPersistent(l.bufWeightsRec, l.bufH, l.bufPreAct, l.outSize*4, 1, l.outSize, 1.0)

		// 4. Fused LSTM Step
		metal.LSTMStepPersistent(l.bufPreAct, l.bufC, l.bufC, l.bufH,
			l.bufI, l.bufF, l.bufG, l.bufO, l.outSize)

		// 5. Read results back for Go state management and backprop
		// PERF-016: Consolidate 7 reads into single synchronization point
		ReadMultiple(
			[]*MetalBuffer{l.bufC, l.bufH, l.bufI, l.bufF, l.bufG, l.bufO, l.bufPreAct},
			[][]float32{l.cellBuf, l.hiddenBuf, l.inputGateOut, l.forgetGateOut, l.cellGateOut, l.outputGateOut, l.preActBuf},
		)

		goto storeState
	} else {
		// CPU implementation with optimized gate calculations
		// PERF-011: Use SIMD-like loop unrolling (4x/8x) for better ILP and cache efficiency
		// Add input contribution: W_x * x
		// Select implementation based on input size for optimal performance
		if l.inSize >= 64 {
			lstmComputeInputGates(l.inputWeights, x, l.preActBuf, l.inSize, l.outSize)
		} else {
			lstmComputeInputGatesUnrolled4x(l.inputWeights, x, l.preActBuf, l.inSize, l.outSize)
		}

		// Add recurrent contribution: W_h * h_prev
		lstmComputeRecurrentGates(l.recurrentWeights, l.hiddenBuf, l.preActBuf, l.outSize)
	}

	// === Apply activations ===

	// Input gate: sigmoid
	for i := 0; i < l.outSize; i++ {
		l.inputGateOut[i] = l.inputAct.Activate(l.preActBuf[inputStart+i])
	}

	// Forget gate: sigmoid
	for i := 0; i < l.outSize; i++ {
		l.forgetGateOut[i] = l.forgetAct.Activate(l.preActBuf[forgetStart+i])
	}

	// Cell candidate: tanh
	for i := 0; i < l.outSize; i++ {
		l.cellGateOut[i] = l.cellAct.Activate(l.preActBuf[cellStart+i])
	}

	// Output gate: sigmoid
	for i := 0; i < l.outSize; i++ {
		l.outputGateOut[i] = l.outputAct.Activate(l.preActBuf[outputStart+i])
	}

	// c_new = forget * c_prev + input * cell_candidate
	// Copy previous cell state (or zeros if first step)
	if l.timeStep > 0 && len(l.savedCellOffsets) > 0 {
		prevCOffset := l.savedCellOffsets[len(l.savedCellOffsets)-1]
		copy(l.cellBuf, (*l.arenaPtr)[prevCOffset:prevCOffset+l.outSize])
	} else {
		for i := range l.cellBuf {
			l.cellBuf[i] = 0
		}
	}

	for i := 0; i < l.outSize; i++ {
		l.cellBuf[i] = l.forgetGateOut[i]*l.cellBuf[i] + l.inputGateOut[i]*l.cellGateOut[i]
	}

	// === Compute new hidden state ===
	// h_new = output * tanh(c_new)
	for i := 0; i < l.outSize; i++ {
		l.hiddenBuf[i] = l.outputGateOut[i] * float32(math.Tanh(float64(l.cellBuf[i])))
	}

storeState:
	// === Store states for backprop === - protegido por mutex quando usando arena
	if arena != nil && offset != nil {
		l.arenaMu.Lock()
		l.arenaPtr = arena
		outSize := l.outSize
		if len(*arena) < *offset+outSize*6 {
			newArena := make([]float32, (*offset+outSize*6)*2)
			copy(newArena, *arena)
			*arena = newArena
		}

		l.savedCellOffsets = append(l.savedCellOffsets, *offset)
		copy((*arena)[*offset:*offset+outSize], l.cellBuf)
		*offset += outSize

		l.savedHiddenOffsets = append(l.savedHiddenOffsets, *offset)
		copy((*arena)[*offset:*offset+outSize], l.hiddenBuf)
		*offset += outSize

		l.savedPreActOffsets = append(l.savedPreActOffsets, *offset)
		copy((*arena)[*offset:*offset+outSize*4], l.preActBuf)
		*offset += outSize * 4
		l.arenaMu.Unlock()
	} else {
		if err := l.saveState(l.cellBuf, l.hiddenBuf, l.preActBuf); err != nil {
			return nil, err
		}
	}

	l.timeStep++

	// Copy output
	copy(l.outputBuf, l.hiddenBuf)
	return l.outputBuf, nil
}

func (l *LSTM) saveState(c, h, pre []float32) error {
	outSize := l.outSize
	if l.arenaPtr == nil {
		l.arenaPtr = &[]float32{}
	}

	// Check for overflow in state size calculation
	stateSize, err := CheckOverflow(outSize, 6)
	if err != nil {
		return fmt.Errorf("LSTM state size overflow (outSize=%d): %w", outSize, err)
	}

	offset := len(*l.arenaPtr)
	if cap(*l.arenaPtr) < offset+stateSize {
		// Check overflow for new arena size
		newSize, err := CheckOverflow(offset+stateSize, 2)
		if err != nil {
			return fmt.Errorf("LSTM arena resize overflow: %w", err)
		}
		newArena := make([]float32, newSize)
		copy(newArena, *l.arenaPtr)
		*l.arenaPtr = newArena
	}
	*l.arenaPtr = (*l.arenaPtr)[:offset+stateSize]
	copy((*l.arenaPtr)[offset:offset+outSize], c)
	l.savedCellOffsets = append(l.savedCellOffsets, offset)
	copy((*l.arenaPtr)[offset+outSize:offset+outSize*2], h)
	l.savedHiddenOffsets = append(l.savedHiddenOffsets, offset+outSize)
	copy((*l.arenaPtr)[offset+outSize*2:offset+stateSize], pre)
	l.savedPreActOffsets = append(l.savedPreActOffsets, offset+outSize*2)
	return nil
}

// Forward performs a forward pass for one time step.
func (l *LSTM) Forward(x []float32) ([]float32, error) {
	return l.ForwardWithArena(x, nil, nil)
}

func (l *LSTM) ForwardBatch(x []float32, batchSize int) ([]float32, error) {
	return l.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (l *LSTM) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) ([]float32, error) {
	if batchSize <= 1 {
		return l.ForwardWithArena(x, arena, offset)
	}
	inSize := l.inSize
	outSize := l.outSize
	if len(l.outputBuf) < batchSize*outSize {
		l.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		// Note: for batching in RNNs, we usually assume independent samples
		// unless specifically handled. We reset before each sample.
		l.Reset()
		out, err := l.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		if err != nil {
			return nil, err
		}
		copy(l.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return l.outputBuf[:batchSize*outSize], nil
}

// Backward performs backpropagation through time for one time step.
// grad: gradient of loss w.r.t. output (length outSize)
// Returns: gradient of loss w.r.t. input (length inSize)
func (l *LSTM) Backward(grad []float32) ([]float32, error) {
	// Get time step index
	ts := l.timeStep - 1
	if ts < 0 || ts >= len(l.savedCellOffsets) {
		for i := range l.inputBuf {
			l.inputBuf[i] = 0
		}
		return l.inputBuf, nil
	}

	// Get saved states from arena
	outSize := l.outSize

	// SEC-012: Validar arena e offsets antes de construir slices
	if l.arenaPtr == nil {
		return nil, fmt.Errorf("%w: arena pointer is nil", ErrInvalidState)
	}
	arenaLen := len(*l.arenaPtr)

	cOffset := l.savedCellOffsets[ts]
	if cOffset < 0 || cOffset > arenaLen-outSize {
		return nil, fmt.Errorf("%w: cell offset %d out of bounds (arena size: %d, outSize: %d)",
			ErrInvalidState, cOffset, arenaLen, outSize)
	}
	c := (*l.arenaPtr)[cOffset : cOffset+outSize]

	// Get saved pre-activations for this time step
	preOffset := l.savedPreActOffsets[ts]
	preActSize := outSize * 4
	if preOffset < 0 || preOffset > arenaLen-preActSize {
		return nil, fmt.Errorf("%w: pre-activation offset %d out of bounds (arena size: %d, preActSize: %d)",
			ErrInvalidState, preOffset, arenaLen, preActSize)
	}
	pre := (*l.arenaPtr)[preOffset : preOffset+preActSize]

	cPrev := l.cPrevBuf
	if ts > 0 {
		prevCOffset := l.savedCellOffsets[ts-1]
		if prevCOffset < 0 || prevCOffset > arenaLen-outSize {
			return nil, fmt.Errorf("%w: prev cell offset %d out of bounds", ErrInvalidState, prevCOffset)
		}
		copy(cPrev, (*l.arenaPtr)[prevCOffset:prevCOffset+outSize])
	} else {
		for i := range cPrev {
			cPrev[i] = 0
		}
	}

	hPrev := l.hPrevBuf
	if ts > 0 {
		prevHOffset := l.savedHiddenOffsets[ts-1]
		if prevHOffset < 0 || prevHOffset > arenaLen-outSize {
			return nil, fmt.Errorf("%w: prev hidden offset %d out of bounds", ErrInvalidState, prevHOffset)
		}
		copy(hPrev, (*l.arenaPtr)[prevHOffset:prevHOffset+outSize])
	} else {
		for i := range hPrev {
			hPrev[i] = 0
		}
	}
	inputStart, forgetStart, cellStart, outputStart := 0, outSize, outSize*2, outSize*3

	// Recompute gate outputs for this time step
	ig, fg, cg, og := l.inputGateOut, l.forgetGateOut, l.cellGateOut, l.outputGateOut
	for i := 0; i < outSize; i++ {
		ig[i] = l.inputAct.Activate(pre[inputStart+i])
		fg[i] = l.forgetAct.Activate(pre[forgetStart+i])
		cg[i] = l.cellAct.Activate(pre[cellStart+i])
		og[i] = l.outputAct.Activate(pre[outputStart+i])
	}

	// Compute dc (gradient w.r.t. cell state)
	// dc(t) = dL/dh(t) * og(t) * (1 - tanh(c(t))^2) + dc(t+1) * fg(t+1)
	// Note: l.dcNextBuf already contains dc(t+1) * fg(t+1) from previous Backward call
	dc := l.dcBuf
	for i := 0; i < l.outSize; i++ {
		tanhC := float32(math.Tanh(float64(c[i])))
		dc[i] = grad[i]*og[i]*(1-tanhC*tanhC) + l.dcNextBuf[i]
	}

	// Gate gradients
	// d_input = dc * cell_candidate * input'(input_pre)
	// d_forget = dc * c_prev * forget'(forget_pre)
	// d_cell = dc * input * cell'(cell_pre)
	// d_output = dL/dh * tanh(c) * output'(output_pre)

	// Input gate gradient
	for i := 0; i < l.outSize; i++ {
		l.dInputBuf[i] = dc[i] * cg[i] * l.inputAct.Derivative(pre[inputStart+i])
	}

	// Forget gate gradient
	for i := 0; i < l.outSize; i++ {
		l.dForgetBuf[i] = dc[i] * cPrev[i] * l.forgetAct.Derivative(pre[forgetStart+i])
	}

	// Cell candidate gradient
	for i := 0; i < l.outSize; i++ {
		l.dCellBuf[i] = dc[i] * ig[i] * l.cellAct.Derivative(pre[cellStart+i])
	}

	// Output gate gradient
	for i := 0; i < l.outSize; i++ {
		l.dOutputBuf[i] = grad[i] * float32(math.Tanh(float64(c[i]))) * l.outputAct.Derivative(pre[outputStart+i])
	}

	// Update dcNext for the previous time step (t-1)
	// dcNext(t-1) = dc(t) * fg(t)
	for i := 0; i < l.outSize; i++ {
		l.dcNextBuf[i] = dc[i] * fg[i]
	}

	// === Accumulate weight gradients ===
	// For each gate, accumulate: d_gate * x^T (input weights) and d_gate * h_prev^T (recurrent weights)

	// Input gate gradients
	for i := 0; i < l.outSize; i++ {
		for j := 0; j < l.inSize; j++ {
			l.gradInputWeights[inputStart*l.inSize+i*l.inSize+j] += l.dInputBuf[i] * l.inputBuf[j]
		}
		l.gradBiases[inputStart+i] += l.dInputBuf[i]
	}

	// Forget gate gradients
	for i := 0; i < l.outSize; i++ {
		for j := 0; j < l.inSize; j++ {
			l.gradInputWeights[forgetStart*l.inSize+i*l.inSize+j] += l.dForgetBuf[i] * l.inputBuf[j]
		}
		l.gradBiases[forgetStart+i] += l.dForgetBuf[i]
	}

	// Cell gate gradients
	for i := 0; i < l.outSize; i++ {
		for j := 0; j < l.inSize; j++ {
			l.gradInputWeights[cellStart*l.inSize+i*l.inSize+j] += l.dCellBuf[i] * l.inputBuf[j]
		}
		l.gradBiases[cellStart+i] += l.dCellBuf[i]
	}

	// Output gate gradients
	for i := 0; i < l.outSize; i++ {
		for j := 0; j < l.inSize; j++ {
			l.gradInputWeights[outputStart*l.inSize+i*l.inSize+j] += l.dOutputBuf[i] * l.inputBuf[j]
		}
		l.gradBiases[outputStart+i] += l.dOutputBuf[i]
	}

	// Recurrent weight gradients
	for g := 0; g < 4; g++ {
		baseG := g * l.outSize
		baseW := baseG * l.outSize
		dg := l.dInputBuf // pointer to current gate delta
		if g == 1 {
			dg = l.dForgetBuf
		} else if g == 2 {
			dg = l.dCellBuf
		} else if g == 3 {
			dg = l.dOutputBuf
		}
		for i := 0; i < l.outSize; i++ {
			for j := 0; j < l.outSize; j++ {
				l.gradRecurrentWeights[baseW+i*l.outSize+j] += dg[i] * hPrev[j]
			}
		}
	}

	// === Compute gradient w.r.t. input (for previous layer) ===
	// dL/dx = sum_over_gates(W_x^T * d_gate)
	dx := l.dxBuf
	for i := 0; i < l.inSize; i++ {
		sum := float32(0.0)
		for g := 0; g < 4; g++ {
			baseG := g * l.outSize
			baseW := baseG * l.inSize
			dg := l.dInputBuf
			if g == 1 {
				dg = l.dForgetBuf
			} else if g == 2 {
				dg = l.dCellBuf
			} else if g == 3 {
				dg = l.dOutputBuf
			}
			for j := 0; j < l.outSize; j++ {
				sum += l.inputWeights[baseW+j*l.inSize+i] * dg[j]
			}
		}
		dx[i] = sum
	}

	// === Compute gradient w.r.t. previous hidden state ===
	// dL/dh_prev = sum_over_gates(W_h^T * d_gate)
	dhPrev := l.dhPrevBuf
	for i := 0; i < l.outSize; i++ {
		sum := float32(0.0)
		for g := 0; g < 4; g++ {
			baseG := g * l.outSize
			baseW := baseG * l.outSize
			dg := l.dInputBuf
			if g == 1 {
				dg = l.dForgetBuf
			} else if g == 2 {
				dg = l.dCellBuf
			} else if g == 3 {
				dg = l.dOutputBuf
			}
			for j := 0; j < l.outSize; j++ {
				sum += l.recurrentWeights[baseW+j*l.outSize+i] * dg[j]
			}
		}
		dhPrev[i] = sum
	}

	// Add dhPrev to grad to pass to previous time step
	for i := 0; i < l.outSize; i++ {
		grad[i] += dhPrev[i]
	}

	// Clip gradients to prevent exploding gradients
	l.clipGradients()

	l.timeStep--
	return dx, nil
}

func (l *LSTM) BackwardBatch(grad []float32, batchSize int) ([]float32, error) {
	if batchSize <= 1 {
		return l.Backward(grad)
	}
	inSize := l.inSize
	outSize := l.outSize
	if len(l.dxBuf) < batchSize*inSize {
		l.dxBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		dx, err := l.Backward(grad[i*outSize : (i+1)*outSize])
		if err != nil {
			return nil, err
		}
		copy(l.dxBuf[i*inSize:(i+1)*inSize], dx)
	}
	return l.dxBuf[:batchSize*inSize], nil
}

func (l *LSTM) AccumulateBackwardBatch(grad []float32, batchSize int) ([]float32, error) {
	return l.BackwardBatch(grad, batchSize)
}

// clipGradients scales gradients if their norm exceeds maxGradientNorm.
// Uses max norm clipping: if ||g||_2 > maxNorm, scale g by maxNorm/||g||_2
func (l *LSTM) clipGradients() {
	// Compute L2 norm of all gradients
	normSq := float32(0.0)
	for i := range l.gradInputWeights {
		normSq += l.gradInputWeights[i] * l.gradInputWeights[i]
	}
	for i := range l.gradRecurrentWeights {
		normSq += l.gradRecurrentWeights[i] * l.gradRecurrentWeights[i]
	}
	for i := range l.gradBiases {
		normSq += l.gradBiases[i] * l.gradBiases[i]
	}

	norm := float32(math.Sqrt(float64(normSq)))
	if norm > maxGradientNorm {
		scale := float32(maxGradientNorm / norm)
		for i := range l.gradInputWeights {
			l.gradInputWeights[i] *= scale
		}
		for i := range l.gradRecurrentWeights {
			l.gradRecurrentWeights[i] *= scale
		}
		for i := range l.gradBiases {
			l.gradBiases[i] *= scale
		}
	}
}

// Params returns all LSTM parameters flattened.
func (l *LSTM) Params() []float32 {
	return l.params
}

// SetParams updates weights and biases from a flattened slice.
func (l *LSTM) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &l.params[0] != &params[0] {
		if len(params) == len(l.params) {
			l.params = params
			if err := l.updateViews(); err != nil {
				// Log error but don't panic - this is a setter function
				// The error would have been caught during Build
				return
			}
		} else {
			copy(l.params, params)
		}
	}
	l.syncGPU()
}

func (l *LSTM) updateViews() error {
	// Calculate sizes with overflow protection
	gateSize, err := CheckOverflow(l.outSize, 4)
	if err != nil {
		return fmt.Errorf("updateViews gate size overflow: %w", err)
	}
	weightInSize, err := CheckOverflow(gateSize, l.inSize)
	if err != nil {
		return fmt.Errorf("updateViews weight input size overflow: %w", err)
	}
	weightRecSize, err := CheckOverflow(gateSize, l.outSize)
	if err != nil {
		return fmt.Errorf("updateViews weight recurrent size overflow: %w", err)
	}

	l.inputWeights = l.params[:weightInSize]
	l.recurrentWeights = l.params[weightInSize : weightInSize+weightRecSize]
	l.biases = l.params[weightInSize+weightRecSize:]
	return nil
}

func (l *LSTM) syncGPU() {
	if metal, ok := l.device.(*MetalDevice); ok && metal.IsAvailable() && l.bufWeightsIn != nil {
		l.bufWeightsIn.Update(l.inputWeights)
		l.bufWeightsRec.Update(l.recurrentWeights)
		l.bufBiases.Update(l.biases)
	}
}

// Gradients returns all LSTM gradients flattened.
func (l *LSTM) Gradients() []float32 {
	return l.grads
}

// SetGradients sets gradients from a flattened slice (in-place).
func (l *LSTM) SetGradients(gradients []float32) {
	if len(gradients) == 0 {
		return
	}
	if &l.grads[0] != &gradients[0] {
		if len(gradients) == len(l.grads) {
			l.grads = gradients
			if err := l.updateGradViews(); err != nil {
				// Log error but don't panic - this is a setter function
				// The error would have been caught during Build
				return
			}
		} else {
			copy(l.grads, gradients)
		}
	}
}

func (l *LSTM) updateGradViews() error {
	// Calculate sizes with overflow protection
	gateSize, err := CheckOverflow(l.outSize, 4)
	if err != nil {
		return fmt.Errorf("updateGradViews gate size overflow: %w", err)
	}
	weightInSize, err := CheckOverflow(gateSize, l.inSize)
	if err != nil {
		return fmt.Errorf("updateGradViews weight input size overflow: %w", err)
	}
	weightRecSize, err := CheckOverflow(gateSize, l.outSize)
	if err != nil {
		return fmt.Errorf("updateGradViews weight recurrent size overflow: %w", err)
	}

	l.gradInputWeights = l.grads[:weightInSize]
	l.gradRecurrentWeights = l.grads[weightInSize : weightInSize+weightRecSize]
	l.gradBiases = l.grads[weightInSize+weightRecSize:]
	return nil
}

// SetDevice sets the computation device for the LSTM layer.
func (l *LSTM) SetDevice(device Device) {
	l.device = device
}

// Close releases all GPU buffers and resources held by the layer.
// This method is idempotent and safe to call multiple times.
// SEC-009: Implementar cleanup explícito de buffers Metal para prevenir memory leaks.
func (l *LSTM) Close() {
	// Liberar buffers GPU em ordem inversa de criação
	if l.bufX != nil {
		l.bufX.Close()
		l.bufX = nil
	}
	if l.bufO != nil {
		l.bufO.Close()
		l.bufO = nil
	}
	if l.bufG != nil {
		l.bufG.Close()
		l.bufG = nil
	}
	if l.bufF != nil {
		l.bufF.Close()
		l.bufF = nil
	}
	if l.bufI != nil {
		l.bufI.Close()
		l.bufI = nil
	}
	if l.bufH != nil {
		l.bufH.Close()
		l.bufH = nil
	}
	if l.bufC != nil {
		l.bufC.Close()
		l.bufC = nil
	}
	if l.bufPreAct != nil {
		l.bufPreAct.Close()
		l.bufPreAct = nil
	}
	if l.bufBiases != nil {
		l.bufBiases.Close()
		l.bufBiases = nil
	}
	if l.bufWeightsRec != nil {
		l.bufWeightsRec.Close()
		l.bufWeightsRec = nil
	}
	if l.bufWeightsIn != nil {
		l.bufWeightsIn.Close()
		l.bufWeightsIn = nil
	}
	// Reset device state
	l.device = &CPUDevice{}
}

// SetTraining sets whether the layer is in training mode.
func (l *LSTM) SetTraining(training bool) {
	l.training = training
}

// InSize returns the input size of the LSTM.
func (l *LSTM) InSize() int {
	return l.inSize
}

// OutSize returns the output size (hidden state) of the LSTM.
func (l *LSTM) OutSize() int {
	return l.outSize
}

// ArenaSize returns the number of float32 values needed in the activation arena.
// LSTM saves cell state (outSize), hidden state (outSize), and pre-activations (outSize*4) per timestep.
// Total per timestep: outSize * 6
func (l *LSTM) ArenaSize() int {
	return l.outSize * 6
}

func (l *LSTM) NamedParams() []NamedParam {
	return []NamedParam{
		{
			Name:  "input_weights",
			Shape: []int{l.outSize * 4, l.inSize},
			Data:  l.inputWeights,
		},
		{
			Name:  "recurrent_weights",
			Shape: []int{l.outSize * 4, l.outSize},
			Data:  l.recurrentWeights,
		},
		{
			Name:  "biases",
			Shape: []int{l.outSize * 4},
			Data:  l.biases,
		},
	}
}

// Hidden returns the current hidden state of the LSTM.
func (l *LSTM) Hidden() []float32 {
	return l.hiddenBuf
}

// ClearGradients zeroes out the accumulated gradients.
func (l *LSTM) ClearGradients() {
	for i := range l.gradInputWeights {
		l.gradInputWeights[i] = 0
	}
	for i := range l.gradRecurrentWeights {
		l.gradRecurrentWeights[i] = 0
	}
	for i := range l.gradBiases {
		l.gradBiases[i] = 0
	}
}

// Clone creates a deep copy of the LSTM layer.
func (l *LSTM) Clone() Layer {
	newL, _ := NewLSTM(l.inSize, l.outSize)
	copy(newL.inputWeights, l.inputWeights)
	copy(newL.recurrentWeights, l.recurrentWeights)
	copy(newL.biases, l.biases)
	return newL
}

func (l *LSTM) LightweightClone(params []float32, grads []float32) Layer {
	// Calculate sizes with overflow protection
	// If overflow would occur, use safe fallback values
	gateSize, err := CheckOverflow(l.outSize, 4)
	if err != nil {
		// Fallback to safe calculation - this shouldn't happen if Build was called correctly
		gateSize = 0
	}
	weightInSize, err := CheckOverflow(gateSize, l.inSize)
	if err != nil {
		weightInSize = 0
	}
	weightRecSize, err := CheckOverflow(gateSize, l.outSize)
	if err != nil {
		weightRecSize = 0
	}

	newL := &LSTM{
		inSize:               l.inSize,
		outSize:              l.outSize,
		params:               params,
		inputWeights:         params[:weightInSize],
		recurrentWeights:     params[weightInSize : weightInSize+weightRecSize],
		biases:               params[weightInSize+weightRecSize:],
		grads:                grads,
		gradInputWeights:     grads[:weightInSize],
		gradRecurrentWeights: grads[weightInSize : weightInSize+weightRecSize],
		gradBiases:           grads[weightInSize+weightRecSize:],
		inputAct:             l.inputAct,
		forgetAct:            l.forgetAct,
		cellAct:              l.cellAct,
		outputAct:            l.outputAct,
		inputBuf:             make([]float32, l.inSize),
		outputBuf:            make([]float32, l.outSize),
		preActBuf:            make([]float32, gateSize),
		cellBuf:              make([]float32, l.outSize),
		hiddenBuf:            make([]float32, l.outSize),
		dInputBuf:            make([]float32, l.outSize),
		dForgetBuf:           make([]float32, l.outSize),
		dCellBuf:             make([]float32, l.outSize),
		dOutputBuf:           make([]float32, l.outSize),
		dcBuf:                make([]float32, l.outSize),
		dxBuf:                make([]float32, l.inSize),
		dhPrevBuf:            make([]float32, l.outSize),
		cPrevBuf:             make([]float32, l.outSize),
		hPrevBuf:             make([]float32, l.outSize),
		inputGateOut:         make([]float32, l.outSize),
		forgetGateOut:        make([]float32, l.outSize),
		cellGateOut:          make([]float32, l.outSize),
		outputGateOut:        make([]float32, l.outSize),
		savedCellOffsets:     make([]int, 0, 16),
		savedHiddenOffsets:   make([]int, 0, 16),
		savedPreActOffsets:   make([]int, 0, 16),
		dcNextBuf:            make([]float32, l.outSize),
		timeStep:             0,
		device:               l.device,
		training:             l.training,
	}

	if md, ok := l.device.(*MetalDevice); ok && md.IsAvailable() && l.inSize > 0 && l.outSize > 0 {
		newL.bufWeightsIn = md.CreateBuffer(newL.inputWeights)
		newL.bufWeightsRec = md.CreateBuffer(newL.recurrentWeights)
		newL.bufBiases = md.CreateBuffer(newL.biases)
		newL.bufPreAct = md.CreateEmptyBuffer(l.outSize * 4)
		newL.bufC = md.CreateEmptyBuffer(l.outSize)
		newL.bufH = md.CreateEmptyBuffer(l.outSize)
		newL.bufI = md.CreateEmptyBuffer(l.outSize)
		newL.bufF = md.CreateEmptyBuffer(l.outSize)
		newL.bufG = md.CreateEmptyBuffer(l.outSize)
		newL.bufO = md.CreateEmptyBuffer(l.outSize)
		newL.bufX = md.CreateEmptyBuffer(l.inSize)
	}

	return newL
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For LSTM, gradients are already accumulated in Backward, so this just calls Backward.
func (l *LSTM) AccumulateBackward(grad []float32) ([]float32, error) {
	return l.Backward(grad)
}

// Stub implementations for optimized LSTM gate calculations (PERF-011)
// These are called but implementation is pending

// lstmComputeInputGates computes input gate contributions with optimized implementation
func lstmComputeInputGates(weights, input, output []float32, inSize, outSize int) {
	// Standard implementation - to be optimized
	for i := 0; i < outSize; i++ {
		for j := 0; j < inSize; j++ {
			output[i] += weights[i*inSize+j] * input[j]
		}
	}
}

// lstmComputeInputGatesUnrolled4x computes input gate contributions with 4x unrolling
func lstmComputeInputGatesUnrolled4x(weights, input, output []float32, inSize, outSize int) {
	// 4x unrolled implementation
	for i := 0; i < outSize; i++ {
		sum := float32(0)
		j := 0
		for ; j <= inSize-4; j += 4 {
			sum += weights[i*inSize+j] * input[j]
			sum += weights[i*inSize+j+1] * input[j+1]
			sum += weights[i*inSize+j+2] * input[j+2]
			sum += weights[i*inSize+j+3] * input[j+3]
		}
		for ; j < inSize; j++ {
			sum += weights[i*inSize+j] * input[j]
		}
		output[i] += sum
	}
}

// lstmComputeRecurrentGates computes recurrent gate contributions
func lstmComputeRecurrentGates(weights, hidden, output []float32, outSize int) {
	// Standard implementation
	for i := 0; i < outSize; i++ {
		for j := 0; j < outSize; j++ {
			output[i] += weights[i*outSize+j] * hidden[j]
		}
	}
}
