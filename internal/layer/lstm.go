// Package layer provides neural network layer implementations.
package layer

import (
	"math"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

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
	inputWeights      []float64 // inSize * outSize * 4
	recurrentWeights  []float64 // outSize * outSize * 4
	biases            []float64 // outSize * 4

	// Activation functions for each gate
	inputAct  activations.Activation
	forgetAct activations.Activation
	cellAct   activations.Activation
	outputAct activations.Activation

	// Reusable buffers - pre-allocated to avoid garbage collection
	inputBuf      []float64 // input vector
	outputBuf     []float64 // output vector (hidden state)
	preActBuf     []float64 // pre-activations for all 4 gates (outSize * 4)
	cellBuf       []float64 // current cell state
	hiddenBuf     []float64 // current hidden state

	// Gradient buffers
	gradInputWeights    []float64
	gradRecurrentWeights []float64
	gradBiases          []float64

	// Delta buffers for backprop (one per gate)
	dInputBuf  []float64
	dForgetBuf []float64
	dCellBuf   []float64
	dOutputBuf []float64

	// Gate outputs stored for backprop (one per gate)
	inputGateOut  []float64
	forgetGateOut []float64
	cellGateOut   []float64
	outputGateOut []float64

	// Saved states for BPTT
	storedCellStates  [][]float64
	storedHiddenStates [][]float64

	// Current time step
	timeStep int
}

// NewLSTM creates a new LSTM layer with pre-allocated buffers.
// inSize: dimension of input vectors
// outSize: dimension of hidden state/output
func NewLSTM(inSize, outSize int) *LSTM {
	// Xavier/Glorot initialization
	scale := math.Sqrt(2.0 / float64(inSize+outSize))

	// Allocate weight matrices
	inputWeights := make([]float64, outSize*4*inSize)
	recurrentWeights := make([]float64, outSize*4*outSize)
	biases := make([]float64, outSize*4)

	// Create deterministic RNG for reproducible initialization
	rng := NewRNG(42)

	// Initialize weights with simple loop for performance
	for i := 0; i < outSize*4; i++ {
		// Initialize biases - forget gate gets bias = 1 (prevent forgetting)
		if i >= outSize && i < outSize*2 {
			biases[i] = 1.0
		}

		for j := 0; j < inSize; j++ {
			inputWeights[i*inSize+j] = rng.RandFloat()*2*scale - scale
		}
		for j := 0; j < outSize; j++ {
			recurrentWeights[i*outSize+j] = rng.RandFloat()*2*scale - scale
		}
	}

	return &LSTM{
		inSize:  inSize,
		outSize: outSize,

		inputWeights:   inputWeights,
		recurrentWeights: recurrentWeights,
		biases:         biases,

		inputAct:  activations.Sigmoid{},
		forgetAct: activations.Sigmoid{},
		cellAct:   activations.Tanh{},
		outputAct: activations.Sigmoid{},

		// Pre-allocate all working buffers
		inputBuf:      make([]float64, inSize),
		outputBuf:     make([]float64, outSize),
		preActBuf:     make([]float64, outSize*4),
		cellBuf:       make([]float64, outSize),
		hiddenBuf:     make([]float64, outSize),

		gradInputWeights:    make([]float64, outSize*4*inSize),
		gradRecurrentWeights: make([]float64, outSize*4*outSize),
		gradBiases:          make([]float64, outSize*4),

		dInputBuf:  make([]float64, outSize),
		dForgetBuf: make([]float64, outSize),
		dCellBuf:   make([]float64, outSize),
		dOutputBuf: make([]float64, outSize),

		inputGateOut:  make([]float64, outSize),
		forgetGateOut: make([]float64, outSize),
		cellGateOut:   make([]float64, outSize),
		outputGateOut: make([]float64, outSize),

		storedCellStates:  make([][]float64, 0),
		storedHiddenStates: make([][]float64, 0),
		timeStep:          0,
	}
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
func (l *LSTM) Forward(x []float64) []float64 {
	// Copy input to buffer
	copy(l.inputBuf, x)

	// === Compute pre-activations ===
	// Initialize with biases
	copy(l.preActBuf, l.biases)

	// Add input contribution: W_x * x
	// For each of 4 gates
	for g := 0; g < 4; g++ {
		baseG := g * l.outSize
		baseW := baseG * l.inSize
		for i := 0; i < l.outSize; i++ {
			sum := 0.0
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
			sum := 0.0
			for j := 0; j < l.outSize; j++ {
				sum += l.recurrentWeights[baseW+i*l.outSize+j] * hPrev[j]
			}
			l.preActBuf[baseG+i] += sum
		}
	}

	// === Apply activations ===
	inputStart, forgetStart, cellStart, outputStart := 0, l.outSize, l.outSize*2, l.outSize*3

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
		l.hiddenBuf[i] = l.outputGateOut[i] * math.Tanh(l.cellBuf[i])
	}

	// === Store states for backprop ===
	if l.timeStep >= len(l.storedCellStates) {
		l.storedCellStates = append(l.storedCellStates, make([]float64, l.outSize))
		l.storedHiddenStates = append(l.storedHiddenStates, make([]float64, l.outSize))
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
func (l *LSTM) Backward(grad []float64) []float64 {
	// Clear gradient buffers
	for i := range l.gradInputWeights {
		l.gradInputWeights[i] = 0
	}
	for i := range l.gradRecurrentWeights {
		l.gradRecurrentWeights[i] = 0
	}
	for i := range l.gradBiases {
		l.gradBiases[i] = 0
	}

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
	cPrev := make([]float64, l.outSize)
	if ts > 0 {
		copy(cPrev, l.storedCellStates[ts-1])
	}
	hPrev := make([]float64, l.outSize)
	if ts > 0 {
		copy(hPrev, l.storedHiddenStates[ts-1])
	}

	// Gate outputs (fg unused in this function but kept for reference)
	ig, _, cg, og := l.inputGateOut, l.forgetGateOut, l.cellGateOut, l.outputGateOut
	pre := l.preActBuf

	// Compute dc (gradient w.r.t. cell state)
	// dc = dL/dh * o * (1 - tanh(c)^2) + dL+1/dc_next * f_next
	// Start with contribution from current hidden state
	dc := make([]float64, l.outSize)
	for i := 0; i < l.outSize; i++ {
		tanhC := math.Tanh(c[i])
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
		l.dOutputBuf[i] = grad[i] * math.Tanh(c[i]) * l.outputAct.Derivative(pre[outputStart+i])
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
	dx := make([]float64, l.inSize)
	for i := 0; i < l.inSize; i++ {
		sum := 0.0
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
	dhPrev := make([]float64, l.outSize)
	for i := 0; i < l.outSize; i++ {
		sum := 0.0
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

	// Copy to inputBuf and return
	copy(l.inputBuf, dx)
	return l.inputBuf
}

// Params returns all LSTM parameters flattened (copy).
func (l *LSTM) Params() []float64 {
	total := len(l.inputWeights) + len(l.recurrentWeights) + len(l.biases)
	params := make([]float64, total)
	copy(params, l.inputWeights)
	copy(params[len(l.inputWeights):], l.recurrentWeights)
	copy(params[len(l.inputWeights)+len(l.recurrentWeights):], l.biases)
	return params
}

// SetParams updates weights and biases from a flattened slice.
func (l *LSTM) SetParams(params []float64) {
	totalInput := len(l.inputWeights)
	totalRecurrent := len(l.recurrentWeights)

	copy(l.inputWeights, params[:totalInput])
	copy(l.recurrentWeights, params[totalInput:totalInput+totalRecurrent])
	copy(l.biases, params[totalInput+totalRecurrent:])
}

// Gradients returns all LSTM gradients flattened (copy).
func (l *LSTM) Gradients() []float64 {
	total := len(l.gradInputWeights) + len(l.gradRecurrentWeights) + len(l.gradBiases)
	gradients := make([]float64, total)
	copy(gradients, l.gradInputWeights)
	copy(gradients[len(l.gradInputWeights):], l.gradRecurrentWeights)
	copy(gradients[len(l.gradInputWeights)+len(l.gradRecurrentWeights):], l.gradBiases)
	return gradients
}

// SetGradients sets gradients from a flattened slice (in-place).
func (l *LSTM) SetGradients(gradients []float64) {
	copy(l.gradInputWeights, gradients[:len(l.gradInputWeights)])
	copy(l.gradRecurrentWeights, gradients[len(l.gradInputWeights):len(l.gradInputWeights)+len(l.gradRecurrentWeights)])
	copy(l.gradBiases, gradients[len(l.gradInputWeights)+len(l.gradRecurrentWeights):])
}

// InSize returns the input size of the LSTM.
func (l *LSTM) InSize() int {
	return l.inSize
}

// OutSize returns the output size (hidden state) of the LSTM.
func (l *LSTM) OutSize() int {
	return l.outSize
}
