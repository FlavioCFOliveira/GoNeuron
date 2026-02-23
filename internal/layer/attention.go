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
	timeStep    int
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
	}
}

// Forward accumulates inputs and returns zeros (or current input) until end of sequence.
// In global attention, the "real" output is produced after the sequence ends.
// But to fit the Layer interface, we accumulate.
func (g *GlobalAttention) Forward(x []float32) []float32 {
	xCopy := make([]float32, len(x))
	copy(xCopy, x)
	g.inputHistory = append(g.inputHistory, xCopy)
	g.timeStep++

	// For per-step forward, we return the current x or zeros.
	// Global attention is usually the last step before classification.
	return x
}

// ComputeContext processes the accumulated history and returns the context vector.
func (g *GlobalAttention) ComputeContext() []float32 {
	seqLen := len(g.inputHistory)
	if seqLen == 0 {
		return make([]float32, g.inSize)
	}

	// 1. Compute similarity scores: dot(input_t, contextVector)
	scores := make([]float32, seqLen)
	for t := 0; t < seqLen; t++ {
		dot := float32(0.0)
		for i := 0; i < g.inSize; i++ {
			dot += g.inputHistory[t][i] * g.contextVector[i]
		}
		scores[t] = dot
	}

	// 2. Apply Softmax to get attention weights
	softmax := activations.Softmax{}
	weights := softmax.ActivateBatch(scores)
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
		return make([]float32, g.inSize)
	}

	// If scores haven't been computed (Sequence hasn't finished properly), compute them
	if len(g.scores) != seqLen {
		g.ComputeContext()
	}

	ts := g.timeStep - 1
	if ts < 0 {
		return make([]float32, g.inSize)
	}

	// This is a complex backward pass because each timestep grad depends on all previous inputs.
	// For simplicity in the first version, we focus on the current timestep's contribution.

	// dL/dInput_t = weight_t * grad + dSoftmax contribution
	// Here we return the gradient for the current timestep ts

	w := g.scores[ts]
	dx := make([]float32, g.inSize)
	for i := 0; i < g.inSize; i++ {
		dx[i] = w * grad[i]
		// Accumulate gradient for context vector
		g.gradContext[i] += grad[i] * w * g.inputHistory[ts][i] // Simplified
	}

	g.timeStep--
	return dx
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
	if &g.contextVector[0] == &params[0] {
		return
	}
	copy(g.contextVector, params)
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
	if &g.gradContext[0] == &grads[0] {
		return
	}
	copy(g.gradContext, grads)
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
