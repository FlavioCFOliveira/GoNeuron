// Package layer provides neural network layer implementations.
package layer

import (
	"math"
)

// Embedding implements an embedding layer for categorical inputs.
// Maps integer indices to dense embedding vectors.
type Embedding struct {
	// Configuration
	num_embeddings int
	embedding_dim  int
	padding_idx    int
	max_norm       float32

	// Learnable parameters: weight matrix [num_embeddings, embedding_dim]
	weights []float32

	// Pre-allocated buffers
	outputBuf   []float32
	gradWeights []float32

	// Saved indices for backward pass
	savedIndices []int

	device Device
}

// NewEmbedding creates a new embedding layer.
// num_embeddings: size of the dictionary of embeddings
// embedding_dim: size of each embedding vector
func NewEmbedding(num_embeddings, embedding_dim int) *Embedding {
	// Xavier/Glorot initialization
	scale := float32(math.Sqrt(3.0 / float64(embedding_dim)))

	weights := make([]float32, num_embeddings*embedding_dim)
	rng := NewRNG(42) // Use deterministic RNG for reproducibility
	for i := range weights {
		weights[i] = scale * (2*rng.RandFloat() - 1)
	}

	return &Embedding{
		num_embeddings: num_embeddings,
		embedding_dim:  embedding_dim,
		weights:        weights,
		outputBuf:      make([]float32, embedding_dim),
		gradWeights:    make([]float32, num_embeddings*embedding_dim),
		savedIndices:   make([]int, 0),
		device:         &CPUDevice{},
	}
}

// SetDevice sets the computation device.
func (e *Embedding) SetDevice(device Device) {
	e.device = device
}

// Forward performs a forward pass through the embedding layer.
// x: input indices of shape [batch_size]
// Returns: embedded vectors of shape [batch_size, embedding_dim]
func (e *Embedding) Forward(x []float32) []float32 {
	batchSize := len(x)
	outputSize := batchSize * e.embedding_dim

	// Ensure output buffer is sized correctly
	if len(e.outputBuf) < outputSize {
		e.outputBuf = make([]float32, outputSize)
	}

	// Save indices for backward pass
	e.savedIndices = make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		e.savedIndices[i] = int(x[i])
	}

	output := e.outputBuf[:outputSize]

	// For each input index
	for b := 0; b < batchSize; b++ {
		idx := e.savedIndices[b]

		// Bounds check
		if idx < 0 || idx >= e.num_embeddings {
			idx = 0 // Default to first embedding for out-of-bounds
		}

		// Copy embedding vector
		start := b * e.embedding_dim
		weightStart := idx * e.embedding_dim
		for d := 0; d < e.embedding_dim; d++ {
			output[start+d] = e.weights[weightStart+d]
		}
	}

	return output
}

// Backward performs backpropagation through the embedding layer.
// In embedding layers, gradients are sparse - only the used embeddings get updates.
func (e *Embedding) Backward(grad []float32) []float32 {
	batchSize := len(e.savedIndices)
	gradBatchSize := len(grad) / e.embedding_dim

	// Clear gradient buffers
	for i := range e.gradWeights {
		e.gradWeights[i] = 0
	}

	// Gradient w.r.t. input is zero (indices don't have gradients)
	gradInput := make([]float32, batchSize)

	for b := 0; b < gradBatchSize; b++ {
		idx := e.savedIndices[b]

		// Bounds check
		if idx < 0 || idx >= e.num_embeddings {
			continue
		}

		weightStart := idx * e.embedding_dim
		gradStart := b * e.embedding_dim

		// Accumulate gradients for this embedding
		for d := 0; d < e.embedding_dim; d++ {
			e.gradWeights[weightStart+d] += grad[gradStart+d]
		}
	}

	return gradInput
}

// Params returns layer parameters (weights).
func (e *Embedding) Params() []float32 {
	total := len(e.weights)
	params := make([]float32, total)
	copy(params, e.weights)
	return params
}

// SetParams updates weights from a flattened slice.
func (e *Embedding) SetParams(params []float32) {
	copy(e.weights, params)
}

// Gradients returns layer gradients (weight gradients).
func (e *Embedding) Gradients() []float32 {
	total := len(e.gradWeights)
	gradients := make([]float32, total)
	copy(gradients, e.gradWeights)
	return gradients
}

// SetGradients sets gradients from a flattened slice (in-place).
func (e *Embedding) SetGradients(gradients []float32) {
	copy(e.gradWeights, gradients)
}

// InSize returns the number of embeddings (vocab size).
func (e *Embedding) InSize() int {
	return e.num_embeddings
}

// OutSize returns the embedding dimension.
func (e *Embedding) OutSize() int {
	return e.embedding_dim
}

// Reset resets the embedding layer.
func (e *Embedding) Reset() {
	e.savedIndices = make([]int, 0)
}

// ClearGradients zeroes out the accumulated gradients.
func (e *Embedding) ClearGradients() {
	for i := range e.gradWeights {
		e.gradWeights[i] = 0
	}
}

// Clone creates a deep copy of the embedding layer.
func (e *Embedding) Clone() Layer {
	newE := NewEmbedding(e.num_embeddings, e.embedding_dim)
	copy(newE.weights, e.weights)
	newE.padding_idx = e.padding_idx
	newE.max_norm = e.max_norm
	newE.device = e.device
	return newE
}

// GetWeights returns the weights matrix.
func (e *Embedding) GetWeights() []float32 {
	return e.weights
}

// GetWeight returns the embedding vector for a specific index.
func (e *Embedding) GetWeight(idx int) []float32 {
	if idx < 0 || idx >= e.num_embeddings {
		idx = 0
	}
	start := idx * e.embedding_dim
	return e.weights[start : start+e.embedding_dim]
}

// SetWeight sets the embedding vector for a specific index.
func (e *Embedding) SetWeight(idx int, values []float32) {
	if idx < 0 || idx >= e.num_embeddings {
		return
	}
	start := idx * e.embedding_dim
	for i := 0; i < e.embedding_dim && i < len(values); i++ {
		e.weights[start+i] = values[i]
	}
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For Embedding, gradients are already accumulated in Backward, so this just calls Backward.
func (e *Embedding) AccumulateBackward(grad []float32) []float32 {
	return e.Backward(grad)
}
