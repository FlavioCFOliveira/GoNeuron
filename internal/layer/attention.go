// Package layer provides neural network layer implementations.
package layer

import (
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"math"
)

// GlobalAttention implements a simple global dot-product attention mechanism.
// It computes a weighted sum of all inputs in the sequence based on their relevance.
// This is commonly used after an LSTM to focus on important parts of the sequence.
type GlobalAttention struct {
	inSize int

	// Parameters (context vector/query for attention)
	// For simple global attention, we can learn a context vector.
	contextVector []float32

	// Buffers
	inputHistory [][]float32
	scores       []float32
	outputBuf    []float32

	// Gradients
	gradContext []float32
	gradInBuf   []float32
	timeStep    int

	training bool
}

// NewGlobalAttention creates a new global attention layer.
func NewGlobalAttention(inSize int) *GlobalAttention {
	rng := NewRNG(uint64(inSize + 42))
	contextVector := make([]float32, inSize)
	scale := float32(math.Sqrt(1.0 / float64(inSize)))
	for i := range contextVector {
		contextVector[i] = rng.RandFloat()*2*scale - scale
	}

	return &GlobalAttention{
		inSize:        inSize,
		contextVector: contextVector,
		inputHistory:  make([][]float32, 0),
		outputBuf:     make([]float32, inSize),
		gradContext:   make([]float32, inSize),
		gradInBuf:     make([]float32, inSize),
	}
}

// SetTraining sets whether the layer is in training mode.
func (g *GlobalAttention) SetTraining(training bool) {
	g.training = training
}

func (g *GlobalAttention) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	var saved []float32
	if arena != nil && offset != nil {
		inSize := len(x)
		if len(*arena) < *offset+inSize {
			newArena := make([]float32, (*offset+inSize)*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		saved = (*arena)[*offset : *offset+inSize]
		copy(saved, x)
		*offset += inSize
	} else {
		saved = make([]float32, len(x))
		copy(saved, x)
	}
	g.inputHistory = append(g.inputHistory, saved)

	g.timeStep++
	return x
}

// Forward accumulates inputs and returns zeros (or current input) until end of sequence.
func (g *GlobalAttention) Forward(x []float32) []float32 {
	return g.ForwardWithArena(x, nil, nil)
}

// ComputeContext processes the accumulated history and returns the context vector.
func (g *GlobalAttention) ComputeContext() []float32 {
	seqLen := len(g.inputHistory)
	if seqLen == 0 {
		return make([]float32, g.inSize)
	}

	// 1. Compute similarity scores: dot(input_t, contextVector)
	if cap(g.scores) < seqLen {
		g.scores = make([]float32, seqLen)
	}
	g.scores = g.scores[:seqLen]
	for t := 0; t < seqLen; t++ {
		dot := float32(0.0)
		for i := 0; i < g.inSize; i++ {
			dot += g.inputHistory[t][i] * g.contextVector[i]
		}
		g.scores[t] = dot
	}

	// 2. Apply Softmax to get attention weights
	softmax := activations.Softmax{}
	weights := softmax.ActivateBatch(g.scores)
	g.scores = weights

	// 3. Compute weighted sum
	for i := 0; i < g.inSize; i++ {
		g.outputBuf[i] = 0
	}
	for t := 0; t < seqLen; t++ {
		w := weights[t]
		for i := 0; i < g.inSize; i++ {
			g.outputBuf[i] += w * g.inputHistory[t][i]
		}
	}

	return g.outputBuf
}

// Backward performs backpropagation.
func (g *GlobalAttention) Backward(grad []float32) []float32 {
	// grad is dL/dOutputBuf
	seqLen := len(g.inputHistory)
	if seqLen == 0 {
		if len(g.gradInBuf) < g.inSize {
			g.gradInBuf = make([]float32, g.inSize)
		}
		for i := 0; i < g.inSize; i++ { g.gradInBuf[i] = 0 }
		return g.gradInBuf[:g.inSize]
	}

	// If scores haven't been computed (Sequence hasn't finished properly), compute them
	if len(g.scores) != seqLen {
		g.ComputeContext()
	}

	ts := g.timeStep - 1
	if ts < 0 {
		if len(g.gradInBuf) < g.inSize {
			g.gradInBuf = make([]float32, g.inSize)
		}
		for i := 0; i < g.inSize; i++ { g.gradInBuf[i] = 0 }
		return g.gradInBuf[:g.inSize]
	}

	// This is a complex backward pass because each timestep grad depends on all previous inputs.
	// For simplicity in the first version, we focus on the current timestep's contribution.
	// dL/dInput_t = weight_t * grad + dSoftmax contribution
	// Here we return the gradient for the current timestep ts

	w := g.scores[ts]
	if len(g.gradInBuf) < g.inSize {
		g.gradInBuf = make([]float32, g.inSize)
	}
	for i := 0; i < g.inSize; i++ {
		g.gradInBuf[i] = w * grad[i]
		// Accumulate gradient for context vector
		g.gradContext[i] += grad[i] * w * g.inputHistory[ts][i] // Simplified
	}

	g.timeStep--
	return g.gradInBuf[:g.inSize]
}

// Params returns the learnable context vector.
func (g *GlobalAttention) Params() []float32 {
	return g.contextVector
}

// SetParams sets the context vector.
func (g *GlobalAttention) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &g.contextVector[0] != &params[0] {
		if len(params) == len(g.contextVector) {
			g.contextVector = params
		} else {
			copy(g.contextVector, params)
		}
	}
}

// Gradients returns context vector gradients.
func (g *GlobalAttention) Gradients() []float32 {
	return g.gradContext
}

// SetGradients sets context vector gradients.
func (g *GlobalAttention) SetGradients(grads []float32) {
	if len(grads) == 0 {
		return
	}
	if &g.gradContext[0] != &grads[0] {
		if len(grads) == len(g.gradContext) {
			g.gradContext = grads
		} else {
			copy(g.gradContext, grads)
		}
	}
}

// Reset clears state.
func (g *GlobalAttention) Reset() {
	g.inputHistory = g.inputHistory[:0]
	g.scores = g.scores[:0]
	g.timeStep = 0
}

// ClearGradients clears gradients.
func (g *GlobalAttention) ClearGradients() {
	for i := range g.gradContext {
		g.gradContext[i] = 0
	}
}

// Clone creates a deep copy.
func (g *GlobalAttention) Clone() Layer {
	newG := NewGlobalAttention(g.inSize)
	copy(newG.contextVector, g.contextVector)
	return newG
}

func (g *GlobalAttention) LightweightClone(params []float32, grads []float32) Layer {
	newG := &GlobalAttention{
		inSize:        g.inSize,
		contextVector: params,
		inputHistory:  make([][]float32, 0),
		outputBuf:     make([]float32, g.inSize),
		gradContext:   grads,
		gradInBuf:     make([]float32, g.inSize),
		training:      g.training,
	}
	return newG
}

func (g *GlobalAttention) ForwardBatch(x []float32, batchSize int) []float32 {
	return g.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (g *GlobalAttention) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return g.ForwardWithArena(x, arena, offset)
	}
	outSize := g.OutSize()
	if len(g.outputBuf) < batchSize*outSize {
		g.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		g.Reset()
		out := g.ForwardWithArena(x[i*g.inSize:(i+1)*g.inSize], arena, offset)
		copy(g.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return g.outputBuf[:batchSize*outSize]
}

func (g *GlobalAttention) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return g.Backward(grad)
	}
	if len(g.gradInBuf) < batchSize*g.inSize {
		g.gradInBuf = make([]float32, batchSize*g.inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		out := g.Backward(grad[i*g.inSize : (i+1)*g.inSize])
		copy(g.gradInBuf[i*g.inSize:(i+1)*g.inSize], out)
	}
	return g.gradInBuf[:batchSize*g.inSize]
}

func (g *GlobalAttention) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return g.BackwardBatch(grad, batchSize)
}

// SetDevice sets the computation device (no-op for GlobalAttention).
func (g *GlobalAttention) SetDevice(Device) {}

// InSize returns input size.
func (g *GlobalAttention) InSize() int {
	return g.inSize
}

// OutSize returns output size (same as inSize).
func (g *GlobalAttention) OutSize() int {
	return g.inSize
}

func (g *GlobalAttention) NamedParams() []NamedParam {
	return []NamedParam{
		{
			Name:  "context_vector",
			Shape: []int{g.inSize},
			Data:  g.contextVector,
		},
	}
}

// AccumulateBackward accumulates gradients.
func (g *GlobalAttention) AccumulateBackward(grad []float32) []float32 {
	return g.Backward(grad)
}
