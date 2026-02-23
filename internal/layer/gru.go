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

	// Parameters stored contiguously
	params []float32 // Contiguous inputWeights + recurrentWeights + biases
	grads  []float32 // Contiguous gradInputWeights + gradRecurrentWeights + gradBiases

	// Views into params
	inputWeights     []float32 // outSize * 3 * inSize
	recurrentWeights []float32 // outSize * outSize * 2
	biases           []float32 // outSize * 3

	// Views into grads
	gradInputWeights     []float32
	gradRecurrentWeights []float32
	gradBiases           []float32

	// Activation functions
	updateAct activations.Activation
	resetAct  activations.Activation
	cellAct   activations.Activation

	// Reusable buffers
	inputBuf     []float32
	outputBuf    []float32
	preActBuf    []float32 // pre-activations for 2 gates + cell candidate (outSize * 3)
	cellBuf      []float32 // current hidden state

	// Delta buffers for backprop
	dUpdateBuf []float32
	dResetBuf  []float32
	dCellBuf   []float32

	// Reusable buffers for backward pass
	dhBuf          []float32
	dhCandidateBuf []float32
	dUpdateBiBuf   []float32 // dL/dz before activation derivative
	dxBuf          []float32
	dhPrevBuf      []float32
	hPrevBuf       []float32

	// Gate outputs stored for backprop
	updateGateOut []float32
	resetGateOut  []float32
	cellGateOut   []float32

	// Saved states for BPTT
	savedHiddenOffsets []int
	arena              []float32

	// Current time step
	timeStep int

	device Device

	training bool
}

// NewGRU creates a new GRU layer with pre-allocated buffers.
// inSize: dimension of input vectors
// outSize: dimension of hidden state/output
func NewGRU(inSize, outSize int) *GRU {
	g := &GRU{
		inSize:  inSize,
		outSize: outSize,

		updateAct: activations.Sigmoid{},
		resetAct:  activations.Sigmoid{},
		cellAct:   activations.Tanh{},

		savedHiddenOffsets: make([]int, 0, 16),
		timeStep:           0,
		device:             &CPUDevice{},
	}

	if inSize != -1 {
		g.Build(inSize)
	}

	return g
}

// Build initializes the layer with the given input size.
func (g *GRU) Build(inSize int) {
	g.inSize = inSize
	outSize := g.outSize

	// Xavier/Glorot initialization
	scale := float32(math.Sqrt(2.0 / (float64(inSize) + float64(outSize))))

	// Allocate weight matrices contiguously
	weightInSize := outSize * 3 * inSize
	weightRecSize := outSize * outSize * 2
	biasSize := outSize * 3
	totalParams := weightInSize + weightRecSize + biasSize
	g.params = make([]float32, totalParams)

	g.inputWeights = g.params[:weightInSize]
	g.recurrentWeights = g.params[weightInSize : weightInSize+weightRecSize]
	g.biases = g.params[weightInSize+weightRecSize:]

	rng := NewRNG(uint64(inSize*1000 + outSize*100 + 242))
	for i := 0; i < outSize*3; i++ {
		for j := 0; j < inSize; j++ {
			g.inputWeights[i*inSize+j] = rng.RandFloat()*2*scale - scale
		}
	}
	for i := 0; i < outSize*2; i++ {
		for j := 0; j < outSize; j++ {
			g.recurrentWeights[i*outSize+j] = rng.RandFloat()*2*scale - scale
		}
	}

	g.grads = make([]float32, totalParams)
	g.gradInputWeights = g.grads[:weightInSize]
	g.gradRecurrentWeights = g.grads[weightInSize : weightInSize+weightRecSize]
	g.gradBiases = g.grads[weightInSize+weightRecSize:]

	g.inputBuf = make([]float32, inSize)
	g.outputBuf = make([]float32, outSize)
	g.preActBuf = make([]float32, outSize*3)
	g.cellBuf = make([]float32, outSize)

	g.dUpdateBuf = make([]float32, outSize)
	g.dResetBuf = make([]float32, outSize)
	g.dCellBuf = make([]float32, outSize)

	g.dhBuf = make([]float32, outSize)
	g.dhCandidateBuf = make([]float32, outSize)
	g.dUpdateBiBuf = make([]float32, outSize)
	g.dxBuf = make([]float32, inSize)
	g.dhPrevBuf = make([]float32, outSize)
	g.hPrevBuf = make([]float32, outSize)

	g.updateGateOut = make([]float32, outSize)
	g.resetGateOut = make([]float32, outSize)
	g.cellGateOut = make([]float32, outSize)
}

// SetDevice sets the computation device.
func (g *GRU) SetDevice(device Device) {
	g.device = device
}

// Reset resets the GRU state for a new sequence.
func (g *GRU) Reset() {
	g.timeStep = 0
	g.savedHiddenOffsets = g.savedHiddenOffsets[:0]
	// Clear buffers
	for i := range g.cellBuf {
		g.cellBuf[i] = 0
	}
}

// SetTraining sets whether the layer is in training mode.
func (g *GRU) SetTraining(training bool) {
	g.training = training
}

func (g *GRU) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
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
			sum := float32(0.0)
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
			sum := float32(0.0)
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
		sum := float32(0.0)
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
	if arena != nil && offset != nil {
		g.arena = *arena
		if len(*arena) < *offset+outSize {
			newArena := make([]float32, (*offset+outSize)*2)
			copy(newArena, *arena)
			*arena = newArena
			g.arena = *arena
		}
		g.savedHiddenOffsets = append(g.savedHiddenOffsets, *offset)
		copy((*arena)[*offset:*offset+outSize], g.cellBuf)
		*offset += outSize
	} else {
		g.saveState(g.cellBuf)
	}

	g.timeStep++

	// Copy output
	copy(g.outputBuf, g.cellBuf)
	return g.outputBuf
}

func (g *GRU) saveState(h []float32) {
	outSize := g.outSize
	offset := len(g.arena)
	if cap(g.arena) < offset+outSize {
		newArena := make([]float32, (offset+outSize)*2)
		copy(newArena, g.arena)
		g.arena = newArena
	}
	g.arena = g.arena[:offset+outSize]
	copy(g.arena[offset:offset+outSize], h)
	g.savedHiddenOffsets = append(g.savedHiddenOffsets, offset)
}

// Forward performs a forward pass for one time step.
func (g *GRU) Forward(x []float32) []float32 {
	return g.ForwardWithArena(x, nil, nil)
}


// Backward performs backpropagation through time for one time step.
// grad: gradient of loss w.r.t. output (length outSize)
// Returns: gradient of loss w.r.t. input (length inSize)
func (g *GRU) Backward(grad []float32) []float32 {
	outSize := g.outSize
	inSize := g.inSize

	// Get time step index
	ts := g.timeStep - 1
	if ts < 0 || ts >= len(g.savedHiddenOffsets) {
		for i := range g.inputBuf {
			g.inputBuf[i] = 0
		}
		return g.inputBuf
	}

	// Get saved states from arena
	hPrev := g.hPrevBuf
	if ts > 0 {
		prevHOffset := g.savedHiddenOffsets[ts-1]
		copy(hPrev, g.arena[prevHOffset:prevHOffset+outSize])
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
		sum := float32(0.0)
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
		sum := float32(0.0)
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

// Params returns all GRU parameters flattened.
func (g *GRU) Params() []float32 {
	return g.params
}

// SetParams updates weights and biases from a flattened slice.
func (g *GRU) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &g.params[0] != &params[0] {
		if len(params) == len(g.params) {
			g.params = params
			g.updateViews()
		} else {
			copy(g.params, params)
		}
	}
}

func (g *GRU) updateViews() {
	weightInSize := g.outSize * 3 * g.inSize
	weightRecSize := g.outSize * g.outSize * 2
	g.inputWeights = g.params[:weightInSize]
	g.recurrentWeights = g.params[weightInSize : weightInSize+weightRecSize]
	g.biases = g.params[weightInSize+weightRecSize:]
}

// Gradients returns all GRU gradients flattened.
func (g *GRU) Gradients() []float32 {
	return g.grads
}

// SetGradients sets gradients from a flattened slice (in-place).
func (g *GRU) SetGradients(gradients []float32) {
	if len(gradients) == 0 {
		return
	}
	if &g.grads[0] != &gradients[0] {
		if len(gradients) == len(g.grads) {
			g.grads = gradients
			g.updateGradViews()
		} else {
			copy(g.grads, gradients)
		}
	}
}

func (g *GRU) updateGradViews() {
	weightInSize := g.outSize * 3 * g.inSize
	weightRecSize := g.outSize * g.outSize * 2
	g.gradInputWeights = g.grads[:weightInSize]
	g.gradRecurrentWeights = g.grads[weightInSize : weightInSize+weightRecSize]
	g.gradBiases = g.grads[weightInSize+weightRecSize:]
}

// InSize returns the input size of the GRU.
func (g *GRU) InSize() int {
	return g.inSize
}

// OutSize returns the output size (hidden state) of the GRU.
func (g *GRU) OutSize() int {
	return g.outSize
}

func (g *GRU) NamedParams() []NamedParam {
	return []NamedParam{
		{
			Name:  "input_weights",
			Shape: []int{g.outSize * 3, g.inSize},
			Data:  g.inputWeights,
		},
		{
			Name:  "recurrent_weights",
			Shape: []int{g.outSize * 2, g.outSize},
			Data:  g.recurrentWeights,
		},
		{
			Name:  "biases",
			Shape: []int{g.outSize * 3},
			Data:  g.biases,
		},
	}
}

// Hidden returns the current hidden state of the GRU.
func (g *GRU) Hidden() []float32 {
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
	newG.device = g.device
	return newG
}

func (g *GRU) LightweightClone(params []float32, grads []float32) Layer {
	weightInSize := g.outSize * 3 * g.inSize
	weightRecSize := g.outSize * g.outSize * 2

	newG := &GRU{
		inSize:               g.inSize,
		outSize:              g.outSize,
		params:               params,
		inputWeights:         params[:weightInSize],
		recurrentWeights:     params[weightInSize : weightInSize+weightRecSize],
		biases:               params[weightInSize+weightRecSize:],
		grads:                grads,
		gradInputWeights:     grads[:weightInSize],
		gradRecurrentWeights: grads[weightInSize : weightInSize+weightRecSize],
		gradBiases:           grads[weightInSize+weightRecSize:],
		updateAct:            g.updateAct,
		resetAct:             g.resetAct,
		cellAct:              g.cellAct,
		inputBuf:             make([]float32, g.inSize),
		outputBuf:            make([]float32, g.outSize),
		preActBuf:            make([]float32, g.outSize*3),
		cellBuf:              make([]float32, g.outSize),
		dUpdateBuf:           make([]float32, g.outSize),
		dResetBuf:            make([]float32, g.outSize),
		dCellBuf:             make([]float32, g.outSize),
		dhBuf:                make([]float32, g.outSize),
		dhCandidateBuf:       make([]float32, g.outSize),
		dUpdateBiBuf:         make([]float32, g.outSize),
		dxBuf:                make([]float32, g.inSize),
		dhPrevBuf:            make([]float32, g.outSize),
		hPrevBuf:             make([]float32, g.outSize),
		updateGateOut:        make([]float32, g.outSize),
		resetGateOut:         make([]float32, g.outSize),
		cellGateOut:          make([]float32, g.outSize),
		savedHiddenOffsets:   make([]int, 0, 16),
		timeStep:             0,
		device:               g.device,
		training:             g.training,
	}
	return newG
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For GRU, gradients are already accumulated in Backward, so this just calls Backward.
func (g *GRU) AccumulateBackward(grad []float32) []float32 {
	return g.Backward(grad)
}
