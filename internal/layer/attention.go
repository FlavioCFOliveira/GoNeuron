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
	contextVector []float64

	// Buffers
	inputHistory [][]float64
	scores       []float64
	outputBuf    []float64

	// Gradients
	gradContext []float64
	timeStep    int
}

// NewGlobalAttention creates a new global attention layer.
func NewGlobalAttention(inSize int) *GlobalAttention {
	rng := NewRNG(uint64(inSize + 42))
	contextVector := make([]float64, inSize)
	scale := math.Sqrt(1.0 / float64(inSize))
	for i := range contextVector {
		contextVector[i] = rng.RandFloat()*2*scale - scale
	}

	return &GlobalAttention{
		inSize:        inSize,
		contextVector: contextVector,
		inputHistory:  make([][]float64, 0),
		outputBuf:     make([]float64, inSize),
		gradContext:   make([]float64, inSize),
	}
}

// Forward accumulates inputs and returns zeros (or current input) until end of sequence.
// In global attention, the "real" output is produced after the sequence ends.
// But to fit the Layer interface, we accumulate.
func (g *GlobalAttention) Forward(x []float64) []float64 {
	xCopy := make([]float64, len(x))
	copy(xCopy, x)
	g.inputHistory = append(g.inputHistory, xCopy)
	g.timeStep++

	// For per-step forward, we return the current x or zeros.
	// Global attention is usually the last step before classification.
	return x
}

// ComputeContext processes the accumulated history and returns the context vector.
func (g *GlobalAttention) ComputeContext() []float64 {
	seqLen := len(g.inputHistory)
	if seqLen == 0 {
		return make([]float64, g.inSize)
	}

	// 1. Compute similarity scores: dot(input_t, contextVector)
	scores := make([]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		dot := 0.0
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
func (g *GlobalAttention) Backward(grad []float64) []float64 {
	// grad is dL/dOutputBuf
	seqLen := len(g.inputHistory)
	if seqLen == 0 {
		return make([]float64, g.inSize)
	}

	// If scores haven't been computed (Sequence hasn't finished properly), compute them
	if len(g.scores) != seqLen {
		g.ComputeContext()
	}

	ts := g.timeStep - 1
	if ts < 0 {
		return make([]float64, g.inSize)
	}

	// This is a complex backward pass because each timestep grad depends on all previous inputs.
	// For simplicity in the first version, we focus on the current timestep's contribution.

	// dL/dInput_t = weight_t * grad + dSoftmax contribution
	// Here we return the gradient for the current timestep ts

	w := g.scores[ts]
	dx := make([]float64, g.inSize)
	for i := 0; i < g.inSize; i++ {
		dx[i] = w * grad[i]
		// Accumulate gradient for context vector
		g.gradContext[i] += grad[i] * w * g.inputHistory[ts][i] // Simplified
	}

	g.timeStep--
	return dx
}

// Params returns the learnable context vector.
func (g *GlobalAttention) Params() []float64 {
	params := make([]float64, g.inSize)
	copy(params, g.contextVector)
	return params
}

// SetParams sets the context vector.
func (g *GlobalAttention) SetParams(params []float64) {
	copy(g.contextVector, params)
}

// Gradients returns context vector gradients.
func (g *GlobalAttention) Gradients() []float64 {
	grads := make([]float64, g.inSize)
	copy(grads, g.gradContext)
	return grads
}

// SetGradients sets context vector gradients.
func (g *GlobalAttention) SetGradients(grads []float64) {
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

// InSize returns input size.
func (g *GlobalAttention) InSize() int {
	return g.inSize
}

// OutSize returns output size (same as inSize).
func (g *GlobalAttention) OutSize() int {
	return g.inSize
}

// AccumulateBackward accumulates gradients.
func (g *GlobalAttention) AccumulateBackward(grad []float64) []float64 {
	return g.Backward(grad)
}
