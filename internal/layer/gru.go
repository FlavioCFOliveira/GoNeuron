// Package layer provides neural network layer implementations.
package layer

import (
	"math"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

// GRU implements a Gated Recurrent Unit layer.
// GRUs are similar to LSTMs but with 2 gates instead of 4 (update and reset gates).
type GRU struct {
	// Parameters
	inSize  int
	outSize int

	// Weight matrices stored contiguously for cache efficiency
	// Layout:
	// - inputWeights: [update_gate, reset_gate, cell_candidate] (3 gates)
	// - recurrentWeights: [update_gate, reset_gate] (2 gates, reset used by cell)
	// - biases: [update_gate, reset_gate, cell_candidate] (3 gates)
	inputWeights      []float64 // outSize * 3 * inSize
	recurrentWeights  []float64 // outSize * outSize * 2
	biases            []float64 // outSize * 3

	// Activation functions
	updateAct activations.Activation
	resetAct  activations.Activation
	cellAct   activations.Activation

	// Reusable buffers
	inputBuf     []float64
	outputBuf    []float64
	preActBuf    []float64 // pre-activations for 2 gates + cell candidate (outSize * 3)
	cellBuf      []float64 // current hidden state

	// Gradient buffers
	gradInputWeights    []float64
	gradRecurrentWeights []float64
	gradBiases          []float64

	// Delta buffers for backprop
	dUpdateBuf []float64
	dResetBuf  []float64
	dCellBuf   []float64

	// Reusable buffers for backward pass
	dhBuf          []float64
	dhCandidateBuf []float64
	dUpdateBiBuf   []float64 // dL/dz before activation derivative
	dxBuf          []float64
	dhPrevBuf      []float64
	hPrevBuf       []float64

	// Gate outputs stored for backprop
	updateGateOut []float64
	resetGateOut  []float64
	cellGateOut   []float64

	// Saved states for BPTT
	storedHiddenStates [][]float64

	// Current time step
	timeStep int
}

// NewGRU creates a new GRU layer with pre-allocated buffers.
// inSize: dimension of input vectors
// outSize: dimension of hidden state/output
func NewGRU(inSize, outSize int) *GRU {
	// Xavier/Glorot initialization
	scale := math.Sqrt(2.0 / float64(inSize+outSize))

	// Allocate weight matrices
	// inputWeights: 3 gates (update, reset, cell candidate)
	// recurrentWeights: 2 gates (update, reset)
	// biases: 3 gates
	inputWeights := make([]float64, outSize*3*inSize)
	recurrentWeights := make([]float64, outSize*outSize*2)
	biases := make([]float64, outSize*3)

	// Create deterministic RNG with layer-specific seed
	rng := NewRNG(uint64(inSize*1000 + outSize*100 + 242))

	// Initialize weights for 3 input gates
	for i := 0; i < outSize*3; i++ {
		for j := 0; j < inSize; j++ {
			inputWeights[i*inSize+j] = rng.RandFloat()*2*scale - scale
		}
	}
	// Initialize recurrent weights for 2 gates
	for i := 0; i < outSize*2; i++ {
		for j := 0; j < outSize; j++ {
			recurrentWeights[i*outSize+j] = rng.RandFloat()*2*scale - scale
		}
	}

	return &GRU{
		inSize:  inSize,
		outSize: outSize,

		inputWeights:     inputWeights,
		recurrentWeights: recurrentWeights,
		biases:           biases,

		updateAct: activations.Sigmoid{},
		resetAct:  activations.Sigmoid{},
		cellAct:   activations.Tanh{},

		// Pre-allocate all working buffers
		inputBuf:     make([]float64, inSize),
		outputBuf:    make([]float64, outSize),
		preActBuf:    make([]float64, outSize*3),
		cellBuf:      make([]float64, outSize),

		gradInputWeights:    make([]float64, outSize*3*inSize),
		gradRecurrentWeights: make([]float64, outSize*outSize*2),
		gradBiases:          make([]float64, outSize*3),

		dUpdateBuf: make([]float64, outSize),
		dResetBuf:  make([]float64, outSize),
		dCellBuf:   make([]float64, outSize),

		dhBuf:          make([]float64, outSize),
		dhCandidateBuf: make([]float64, outSize),
		dUpdateBiBuf:   make([]float64, outSize),
		dxBuf:          make([]float64, inSize),
		dhPrevBuf:      make([]float64, outSize),
		hPrevBuf:       make([]float64, outSize),

		updateGateOut: make([]float64, outSize),
		resetGateOut:  make([]float64, outSize),
		cellGateOut:   make([]float64, outSize),

		storedHiddenStates: make([][]float64, 0),
		timeStep:           0,
	}
}

// Reset resets the GRU state for a new sequence.
func (g *GRU) Reset() {
	g.timeStep = 0
	g.storedHiddenStates = g.storedHiddenStates[:0]
	// Clear buffers
	for i := range g.cellBuf {
		g.cellBuf[i] = 0
	}
}

// Forward performs a forward pass for one time step.
// x: input vector of length inSize
// Returns: output vector of length outSize
func (g *GRU) Forward(x []float64) []float64 {
	// Copy input to buffer
	copy(g.inputBuf, x)

	outSize := g.outSize
	inSize := g.inSize

	// === Compute pre-activations ===
	// Initialize with biases
	copy(g.preActBuf, g.biases)

	// Add input contribution: W_x * x
	// For all 3 gates: update, reset, and cell candidate
	hPrev := g.cellBuf
	for gateIdx := 0; gateIdx < 3; gateIdx++ {
		baseG := gateIdx * outSize
		baseW := baseG * inSize
		for i := 0; i < outSize; i++ {
			sum := 0.0
			for j := 0; j < inSize; j++ {
				sum += g.inputWeights[baseW+i*inSize+j] * x[j]
			}
			g.preActBuf[baseG+i] += sum
		}
	}

	// Add recurrent contribution: W_h * h_prev
	// For update and reset gates only (cell uses reset * h_prev)
	for gateIdx := 0; gateIdx < 2; gateIdx++ {
		baseG := gateIdx * outSize
		baseW := baseG * outSize
		for i := 0; i < outSize; i++ {
			sum := 0.0
			for j := 0; j < outSize; j++ {
				sum += g.recurrentWeights[baseW+i*outSize+j] * hPrev[j]
			}
			g.preActBuf[baseG+i] += sum
		}
	}

	// === Apply activations ===
	updateStart, resetStart, cellStart := 0, outSize, outSize*2

	// Update gate: sigmoid
	for i := 0; i < outSize; i++ {
		g.updateGateOut[i] = g.updateAct.Activate(g.preActBuf[updateStart+i])
	}

	// Reset gate: sigmoid
	for i := 0; i < outSize; i++ {
		g.resetGateOut[i] = g.resetAct.Activate(g.preActBuf[resetStart+i])
	}

	// Cell candidate: tanh (uses updated hidden state with reset gate)
	// h_candidate = tanh(W_x * x + W_h * (r * h_prev))
	// Note: we compute this with the reset gate applied to h_prev
	// The cell candidate uses the same recurrent weights as the reset gate
	// but with reset applied to h_prev
	recurrentOffset := resetStart * outSize // = outSize * outSize for recurrent weights
	for i := 0; i < outSize; i++ {
		// Compute W_h * (r * h_prev) for cell candidate
		sum := 0.0
		for j := 0; j < outSize; j++ {
			sum += g.recurrentWeights[recurrentOffset+i*outSize+j] * (g.resetGateOut[j] * hPrev[j])
		}
		g.preActBuf[cellStart+i] += sum
		g.cellGateOut[i] = g.cellAct.Activate(g.preActBuf[cellStart+i])
	}

	// === Compute new hidden state ===
	// h_new = (1 - z) * h_candidate + z * h_prev
	// Or equivalently: h_new = h_prev + z * (h_candidate - h_prev)
	for i := 0; i < outSize; i++ {
		g.cellBuf[i] = hPrev[i] + g.updateGateOut[i]*(g.cellGateOut[i]-hPrev[i])
	}

	// === Store states for backprop ===
	if g.timeStep >= len(g.storedHiddenStates) {
		g.storedHiddenStates = append(g.storedHiddenStates, make([]float64, outSize))
	}
	copy(g.storedHiddenStates[g.timeStep], g.cellBuf)

	g.timeStep++

	// Copy output
	copy(g.outputBuf, g.cellBuf)
	return g.outputBuf
}

// Backward performs backpropagation through time for one time step.
// grad: gradient of loss w.r.t. output (length outSize)
// Returns: gradient of loss w.r.t. input (length inSize)
func (g *GRU) Backward(grad []float64) []float64 {
	outSize := g.outSize
	inSize := g.inSize

	// Get time step index
	ts := g.timeStep - 1
	if ts < 0 || ts >= len(g.storedHiddenStates) {
		for i := range g.inputBuf {
			g.inputBuf[i] = 0
		}
		return g.inputBuf
	}

	// Get saved states
	hPrev := g.hPrevBuf
	if ts > 0 {
		copy(hPrev, g.storedHiddenStates[ts-1])
	} else {
		for i := range hPrev {
			hPrev[i] = 0
		}
	}

	updateStart, resetStart, cellStart := 0, outSize, outSize*2

	// Compute gradient of loss w.r.t. hidden state
	// dL/dh = dL/doutput + dL+1/dh_next
	dh := g.dhBuf
	for i := 0; i < outSize; i++ {
		dh[i] = grad[i]
	}

	// Compute dh_candidate = dh * (h - h_prev) = dh * (1 - z) * dh_candidate/d(h_candidate)
	// Actually: dL/dh_candidate = dL/dh * (1 - z)
	dhCandidate := g.dhCandidateBuf
	for i := 0; i < outSize; i++ {
		dhCandidate[i] = dh[i] * (1 - g.updateGateOut[i])
	}

	// Compute dL/dz = dL/dh * (h_candidate - h_prev)
	dUpdate := g.dUpdateBiBuf
	for i := 0; i < outSize; i++ {
		dUpdate[i] = dh[i] * (g.cellGateOut[i] - hPrev[i])
	}

	// Gate gradients (before activation)
	// d_update = dL/dz * z * (1 - z)
	for i := 0; i < outSize; i++ {
		g.dUpdateBuf[i] = dUpdate[i] * g.updateGateOut[i] * (1 - g.updateGateOut[i])
	}

	// d_cell = dL/dh_candidate * (1 - tanh^2)
	for i := 0; i < outSize; i++ {
		tanhC := g.cellGateOut[i]
		dCell := dhCandidate[i] * (1 - tanhC*tanhC)
		g.dCellBuf[i] = dCell
	}

	// Accumulate weight gradients
	// Update gate gradients
	for i := 0; i < outSize; i++ {
		for j := 0; j < inSize; j++ {
			g.gradInputWeights[updateStart*inSize+i*inSize+j] += g.dUpdateBuf[i] * g.inputBuf[j]
		}
		g.gradBiases[updateStart+i] += g.dUpdateBuf[i]
	}

	// Reset gate gradients
	for i := 0; i < outSize; i++ {
		for j := 0; j < inSize; j++ {
			g.gradInputWeights[resetStart*inSize+i*inSize+j] += g.dResetBuf[i] * g.inputBuf[j]
		}
		g.gradBiases[resetStart+i] += g.dResetBuf[i]
	}

	// Cell candidate gradients
	for i := 0; i < outSize; i++ {
		for j := 0; j < inSize; j++ {
			g.gradInputWeights[cellStart*inSize+i*inSize+j] += g.dCellBuf[i] * g.inputBuf[j]
		}
		g.gradBiases[cellStart+i] += g.dCellBuf[i]
	}

	// Recurrent weight gradients
	// Update gate
	for i := 0; i < outSize; i++ {
		for j := 0; j < outSize; j++ {
			g.gradRecurrentWeights[updateStart*outSize+i*outSize+j] += g.dUpdateBuf[i] * hPrev[j]
		}
	}

	// Reset gate
	for i := 0; i < outSize; i++ {
		for j := 0; j < outSize; j++ {
			g.gradRecurrentWeights[resetStart*outSize+i*outSize+j] += g.dResetBuf[i] * hPrev[j]
		}
	}

	// Cell candidate - uses reset * h_prev (same recurrent weights as reset gate)
	recurrentOffset := resetStart * outSize
	for i := 0; i < outSize; i++ {
		for j := 0; j < outSize; j++ {
			g.gradRecurrentWeights[recurrentOffset+i*outSize+j] += g.dCellBuf[i] * (g.resetGateOut[j] * hPrev[j])
		}
	}

	// Compute gradient w.r.t. input (for previous layer)
	// All 3 gates use input weights: update, reset, and cell candidate
	dx := g.dxBuf
	for i := 0; i < inSize; i++ {
		sum := 0.0
		for gateIdx := 0; gateIdx < 3; gateIdx++ {
			baseG := gateIdx * outSize
			baseW := baseG * inSize
			dg := g.dUpdateBuf
			if gateIdx == 1 {
				dg = g.dResetBuf
			} else if gateIdx == 2 {
				dg = g.dCellBuf
			}
			for j := 0; j < outSize; j++ {
				sum += g.inputWeights[baseW+j*inSize+i] * dg[j]
			}
		}
		dx[i] = sum
	}

	// Compute gradient w.r.t. previous hidden state
	// Update and reset gates use h_prev directly, cell candidate uses reset * h_prev
	dhPrev := g.dhPrevBuf
	for i := 0; i < outSize; i++ {
		sum := 0.0
		// Update gate contribution
		for j := 0; j < outSize; j++ {
			sum += g.recurrentWeights[updateStart*outSize+j*outSize+i] * g.dUpdateBuf[j]
		}
		// Reset gate contribution
		for j := 0; j < outSize; j++ {
			sum += g.recurrentWeights[resetStart*outSize+j*outSize+i] * g.dResetBuf[j]
		}
		// Cell candidate contribution (uses reset * h_prev)
		for j := 0; j < outSize; j++ {
			sum += g.recurrentWeights[recurrentOffset+j*outSize+i] * g.dCellBuf[j] * g.resetGateOut[j]
		}
		// Add contribution from update gate: dL/dh * z
		sum += dh[i] * g.updateGateOut[i]
		dhPrev[i] = sum
	}

	// Add dhPrev to grad to pass to previous time step
	for i := 0; i < outSize; i++ {
		grad[i] += dhPrev[i]
	}

	g.timeStep--
	return dx
}

// Params returns all GRU parameters flattened (copy).
func (g *GRU) Params() []float64 {
	total := len(g.inputWeights) + len(g.recurrentWeights) + len(g.biases)
	params := make([]float64, total)
	copy(params, g.inputWeights)
	copy(params[len(g.inputWeights):], g.recurrentWeights)
	copy(params[len(g.inputWeights)+len(g.recurrentWeights):], g.biases)
	return params
}

// SetParams updates weights and biases from a flattened slice.
func (g *GRU) SetParams(params []float64) {
	totalInput := len(g.inputWeights)
	totalRecurrent := len(g.recurrentWeights)

	copy(g.inputWeights, params[:totalInput])
	copy(g.recurrentWeights, params[totalInput:totalInput+totalRecurrent])
	copy(g.biases, params[totalInput+totalRecurrent:])
}

// Gradients returns all GRU gradients flattened (copy).
func (g *GRU) Gradients() []float64 {
	total := len(g.gradInputWeights) + len(g.gradRecurrentWeights) + len(g.gradBiases)
	gradients := make([]float64, total)
	copy(gradients, g.gradInputWeights)
	copy(gradients[len(g.gradInputWeights):], g.gradRecurrentWeights)
	copy(gradients[len(g.gradInputWeights)+len(g.gradRecurrentWeights):], g.gradBiases)
	return gradients
}

// SetGradients sets gradients from a flattened slice (in-place).
func (g *GRU) SetGradients(gradients []float64) {
	copy(g.gradInputWeights, gradients[:len(g.gradInputWeights)])
	copy(g.gradRecurrentWeights, gradients[len(g.gradInputWeights):len(g.gradInputWeights)+len(g.gradRecurrentWeights)])
	copy(g.gradBiases, gradients[len(g.gradInputWeights)+len(g.gradRecurrentWeights):])
}

// InSize returns the input size of the GRU.
func (g *GRU) InSize() int {
	return g.inSize
}

// OutSize returns the output size (hidden state) of the GRU.
func (g *GRU) OutSize() int {
	return g.outSize
}

// Hidden returns the current hidden state of the GRU.
func (g *GRU) Hidden() []float64 {
	return g.cellBuf
}

// ClearGradients zeroes out the accumulated gradients.
func (g *GRU) ClearGradients() {
	for i := range g.gradInputWeights {
		g.gradInputWeights[i] = 0
	}
	for i := range g.gradRecurrentWeights {
		g.gradRecurrentWeights[i] = 0
	}
	for i := range g.gradBiases {
		g.gradBiases[i] = 0
	}
}

// Clone creates a deep copy of the GRU layer.
func (g *GRU) Clone() Layer {
	newG := NewGRU(g.inSize, g.outSize)
	copy(newG.inputWeights, g.inputWeights)
	copy(newG.recurrentWeights, g.recurrentWeights)
	copy(newG.biases, g.biases)
	return newG
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For GRU, gradients are already accumulated in Backward, so this just calls Backward.
func (g *GRU) AccumulateBackward(grad []float64) []float64 {
	return g.Backward(grad)
}
