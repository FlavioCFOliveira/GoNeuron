// Package layer provides neural network layer implementations.
package layer

import (
	"math"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

// Max gradient norm for clipping
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
	inputWeights      []float32 // inSize * outSize * 4
	recurrentWeights  []float32 // outSize * outSize * 4
	biases            []float32 // outSize * 4

	// Activation functions for each gate
	inputAct  activations.Activation
	forgetAct activations.Activation
	cellAct   activations.Activation
	outputAct activations.Activation

	// Reusable buffers - pre-allocated to avoid garbage collection
	inputBuf      []float32 // input vector
	outputBuf     []float32 // output vector (hidden state)
	preActBuf     []float32 // pre-activations for all 4 gates (outSize * 4)
	cellBuf       []float32 // current cell state
	hiddenBuf     []float32 // current hidden state

	// Gradient buffers
	gradInputWeights    []float32
	gradRecurrentWeights []float32
	gradBiases          []float32

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
	storedCellStates  [][]float32
	storedHiddenStates [][]float32

	// Current time step
	timeStep int

	// Device for computation
	device Device

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
func NewLSTM(inSize, outSize int) *LSTM {
	return NewLSTMWithDevice(inSize, outSize, &CPUDevice{})
}

// NewLSTMWithDevice creates a new LSTM layer with a specific device.
func NewLSTMWithDevice(inSize, outSize int, device Device) *LSTM {
	// Xavier/Glorot initialization for LSTM with 4 gates
	// Input weights connect inSize inputs to 4*outSize gate outputs
	inputScale := float32(math.Sqrt(2.0 / float64(inSize + 4*outSize)))
	// Recurrent weights connect outSize hidden states to 4*outSize gate outputs
	recurrentScale := float32(math.Sqrt(2.0 / float64(outSize + 4*outSize)))

	// Allocate weight matrices
	inputWeights := make([]float32, outSize*4*inSize)
	recurrentWeights := make([]float32, outSize*4*outSize)
	biases := make([]float32, outSize*4)

	// Create deterministic RNG with layer-specific seed for different initial weights
	rng := NewRNG(uint64(inSize*1000 + outSize*100 + 142))

	// Initialize weights with simple loop for performance
	// Use separate scales for input and recurrent weights
	for i := 0; i < outSize*4; i++ {
		// Initialize biases - forget gate gets bias = 1 (prevent forgetting)
		if i >= outSize && i < outSize*2 {
			biases[i] = 1.0
		}

		for j := 0; j < inSize; j++ {
			inputWeights[i*inSize+j] = rng.RandFloat()*2*inputScale - inputScale
		}
		for j := 0; j < outSize; j++ {
			recurrentWeights[i*outSize+j] = rng.RandFloat()*2*recurrentScale - recurrentScale
		}
	}

	l := &LSTM{
		inSize:  inSize,
		outSize: outSize,
		device:  device,

		inputWeights:   inputWeights,
		recurrentWeights: recurrentWeights,
		biases:         biases,

		inputAct:  activations.Sigmoid{},
		forgetAct: activations.Sigmoid{},
		cellAct:   activations.Tanh{},
		outputAct: activations.Sigmoid{},

		// Pre-allocate all working buffers
		inputBuf:      make([]float32, inSize),
		outputBuf:     make([]float32, outSize),
		preActBuf:     make([]float32, outSize*4),
		cellBuf:       make([]float32, outSize),
		hiddenBuf:     make([]float32, outSize),

		gradInputWeights:    make([]float32, outSize*4*inSize),
		gradRecurrentWeights: make([]float32, outSize*4*outSize),
		gradBiases:          make([]float32, outSize*4),

		dInputBuf:  make([]float32, outSize),
		dForgetBuf: make([]float32, outSize),
		dCellBuf:   make([]float32, outSize),
		dOutputBuf: make([]float32, outSize),

		dcBuf:     make([]float32, outSize),
		dxBuf:     make([]float32, inSize),
		dhPrevBuf: make([]float32, outSize),
		cPrevBuf:  make([]float32, outSize),
		hPrevBuf:  make([]float32, outSize),

		inputGateOut:  make([]float32, outSize),
		forgetGateOut: make([]float32, outSize),
		cellGateOut:   make([]float32, outSize),
		outputGateOut: make([]float32, outSize),

		storedCellStates:  make([][]float32, 0),
		storedHiddenStates: make([][]float32, 0),
		timeStep:          0,
	}

	if metal, ok := device.(*MetalDevice); ok && metal.IsAvailable() {
		l.bufWeightsIn = metal.CreateBuffer(inputWeights)
		l.bufWeightsRec = metal.CreateBuffer(recurrentWeights)
		l.bufBiases = metal.CreateBuffer(biases)
		l.bufPreAct = metal.CreateEmptyBuffer(outSize * 4)
		l.bufC = metal.CreateEmptyBuffer(outSize)
		l.bufH = metal.CreateEmptyBuffer(outSize)
		l.bufI = metal.CreateEmptyBuffer(outSize)
		l.bufF = metal.CreateEmptyBuffer(outSize)
		l.bufG = metal.CreateEmptyBuffer(outSize)
		l.bufO = metal.CreateEmptyBuffer(outSize)
		l.bufX = metal.CreateEmptyBuffer(inSize)
	}

	return l
}

// Reset resets the LSTM state for a new sequence.
func (l *LSTM) Reset() {
	l.timeStep = 0
	l.storedCellStates = l.storedCellStates[:0]
	l.storedHiddenStates = l.storedHiddenStates[:0]
	// Clear buffers
	for i := range l.cellBuf {
		l.cellBuf[i] = 0
	}
	for i := range l.hiddenBuf {
		l.hiddenBuf[i] = 0
	}
}

// Forward performs a forward pass for one time step.
// x: input vector of length inSize
// Returns: output vector of length outSize
func (l *LSTM) Forward(x []float32) []float32 {
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
		l.bufC.Read(l.cellBuf)
		l.bufH.Read(l.hiddenBuf)
		l.bufI.Read(l.inputGateOut)
		l.bufF.Read(l.forgetGateOut)
		l.bufG.Read(l.cellGateOut)
		l.bufO.Read(l.outputGateOut)
		l.bufPreAct.Read(l.preActBuf)

		goto storeState
	} else {
		// CPU implementation
		// Add input contribution: W_x * x
		// For each of 4 gates
		for g := 0; g < 4; g++ {
			baseG := g * l.outSize
			baseW := baseG * l.inSize
			for i := 0; i < l.outSize; i++ {
				sum := float32(0.0)
				for j := 0; j < l.inSize; j++ {
					sum += l.inputWeights[baseW+i*l.inSize+j] * x[j]
				}
				l.preActBuf[baseG+i] += sum
			}
		}

		// Add recurrent contribution: W_h * h_prev
		hPrev := l.hiddenBuf
		for g := 0; g < 4; g++ {
			baseG := g * l.outSize
			baseW := baseG * l.outSize
			for i := 0; i < l.outSize; i++ {
				sum := float32(0.0)
				for j := 0; j < l.outSize; j++ {
					sum += l.recurrentWeights[baseW+i*l.outSize+j] * hPrev[j]
				}
				l.preActBuf[baseG+i] += sum
			}
		}
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

	// === Compute new cell state ===
	// c_new = forget * c_prev + input * cell_candidate
	// Copy previous cell state (or zeros if first step)
	if l.timeStep > 0 && len(l.storedCellStates) > 0 {
		copy(l.cellBuf, l.storedCellStates[len(l.storedCellStates)-1])
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
	// === Store states for backprop ===
	if l.timeStep >= len(l.storedCellStates) {
		l.storedCellStates = append(l.storedCellStates, make([]float32, l.outSize))
		l.storedHiddenStates = append(l.storedHiddenStates, make([]float32, l.outSize))
	}
	copy(l.storedCellStates[l.timeStep], l.cellBuf)
	copy(l.storedHiddenStates[l.timeStep], l.hiddenBuf)

	l.timeStep++

	// Copy output
	copy(l.outputBuf, l.hiddenBuf)
	return l.outputBuf
}

// Backward performs backpropagation through time for one time step.
// grad: gradient of loss w.r.t. output (length outSize)
// Returns: gradient of loss w.r.t. input (length inSize)
func (l *LSTM) Backward(grad []float32) []float32 {
	// Get time step index
	ts := l.timeStep - 1
	if ts < 0 || ts >= len(l.storedCellStates) {
		for i := range l.inputBuf {
			l.inputBuf[i] = 0
		}
		return l.inputBuf
	}

	// Get saved states
	c := l.storedCellStates[ts]
	cPrev := l.cPrevBuf
	if ts > 0 {
		copy(cPrev, l.storedCellStates[ts-1])
	} else {
		for i := range cPrev {
			cPrev[i] = 0
		}
	}
	hPrev := l.hPrevBuf
	if ts > 0 {
		copy(hPrev, l.storedHiddenStates[ts-1])
	} else {
		for i := range hPrev {
			hPrev[i] = 0
		}
	}

	// Gate outputs (fg unused in this function but kept for reference)
	ig, _, cg, og := l.inputGateOut, l.forgetGateOut, l.cellGateOut, l.outputGateOut
	pre := l.preActBuf

	// Compute dc (gradient w.r.t. cell state)
	// dc = dL/dh * o * (1 - tanh(c)^2) + dL+1/dc_next * f_next
	// Start with contribution from current hidden state
	dc := l.dcBuf
	for i := 0; i < l.outSize; i++ {
		tanhC := float32(math.Tanh(float64(c[i])))
		dc[i] = grad[i] * og[i] * (1 - tanhC*tanhC)
	}

	// Gate gradients
	// d_input = dc * cell_candidate * input'(input)
	// d_forget = dc * c_prev * forget'(forget)
	// d_cell = dc * input * cell'(cell)
	// d_output = dL/dh * tanh(c) * output'(output)
	// Note: dL/dh includes grad from output AND dc from cell state

	inputStart, forgetStart, cellStart, outputStart := 0, l.outSize, l.outSize*2, l.outSize*3

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
	return dx
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

// Params returns all LSTM parameters flattened (copy).
func (l *LSTM) Params() []float32 {
	total := len(l.inputWeights) + len(l.recurrentWeights) + len(l.biases)
	params := make([]float32, total)
	copy(params, l.inputWeights)
	copy(params[len(l.inputWeights):], l.recurrentWeights)
	copy(params[len(l.inputWeights)+len(l.recurrentWeights):], l.biases)
	return params
}

// SetParams updates weights and biases from a flattened slice.
func (l *LSTM) SetParams(params []float32) {
	totalInput := len(l.inputWeights)
	totalRecurrent := len(l.recurrentWeights)

	copy(l.inputWeights, params[:totalInput])
	copy(l.recurrentWeights, params[totalInput:totalInput+totalRecurrent])
	copy(l.biases, params[totalInput+totalRecurrent:])
}

// Gradients returns all LSTM gradients flattened (copy).
func (l *LSTM) Gradients() []float32 {
	total := len(l.gradInputWeights) + len(l.gradRecurrentWeights) + len(l.gradBiases)
	gradients := make([]float32, total)
	copy(gradients, l.gradInputWeights)
	copy(gradients[len(l.gradInputWeights):], l.gradRecurrentWeights)
	copy(gradients[len(l.gradInputWeights)+len(l.gradRecurrentWeights):], l.gradBiases)
	return gradients
}

// SetGradients sets gradients from a flattened slice (in-place).
func (l *LSTM) SetGradients(gradients []float32) {
	copy(l.gradInputWeights, gradients[:len(l.gradInputWeights)])
	copy(l.gradRecurrentWeights, gradients[len(l.gradInputWeights):len(l.gradInputWeights)+len(l.gradRecurrentWeights)])
	copy(l.gradBiases, gradients[len(l.gradInputWeights)+len(l.gradRecurrentWeights):])
}

// SetDevice sets the computation device for the LSTM layer.
func (l *LSTM) SetDevice(device Device) {
	l.device = device
}

// InSize returns the input size of the LSTM.
func (l *LSTM) InSize() int {
	return l.inSize
}

// OutSize returns the output size (hidden state) of the LSTM.
func (l *LSTM) OutSize() int {
	return l.outSize
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
	newL := NewLSTM(l.inSize, l.outSize)
	copy(newL.inputWeights, l.inputWeights)
	copy(newL.recurrentWeights, l.recurrentWeights)
	copy(newL.biases, l.biases)
	return newL
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For LSTM, gradients are already accumulated in Backward, so this just calls Backward.
func (l *LSTM) AccumulateBackward(grad []float32) []float32 {
	return l.Backward(grad)
}
