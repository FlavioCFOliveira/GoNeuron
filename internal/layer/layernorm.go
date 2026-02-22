// Package layer provides neural network layer implementations.
package layer

import (
	"math"
)

// LayerNorm implements layer normalization.
// Normalizes across feature dimensions (not batch dimension).
type LayerNorm struct {
	// Normalization parameters
	normalizedShape int
	eps             float64
	elementwiseAffine bool

	// Learnable parameters (when affine is enabled)
	gamma []float64
	beta  []float64

	// Pre-allocated buffers
	outputBuf      []float64
	gradInBuf      []float64
	gradGammaBuf   []float64
	gradBetaBuf    []float64
	gradBuf        []float64 // Temporary gradient buffer
	savedInput     []float64 // Input saved for backward pass
	inputMean      float64   // Stored mean for backward
	inputStd       float64   // Stored std for backward
}

// NewLayerNorm creates a new layer normalization layer.
// normalizedShape: the shape of the input from the expected input dimension
// eps: small value for numerical stability (default 1e-5)
// elementwiseAffine: whether to learn scale (gamma) and shift (beta) parameters
func NewLayerNorm(normalizedShape int, eps float64, elementwiseAffine bool) *LayerNorm {
	l := &LayerNorm{
		normalizedShape:   normalizedShape,
		eps:               eps,
		elementwiseAffine: elementwiseAffine,
		outputBuf:         make([]float64, normalizedShape),
		gradInBuf:         make([]float64, normalizedShape),
		gradBuf:           make([]float64, normalizedShape),
	}

	if elementwiseAffine {
		l.gamma = make([]float64, normalizedShape)
		l.beta = make([]float64, normalizedShape)
		l.gradGammaBuf = make([]float64, normalizedShape)
		l.gradBetaBuf = make([]float64, normalizedShape)

		// Initialize gamma to 1 and beta to 0
		for i := 0; i < normalizedShape; i++ {
			l.gamma[i] = 1.0
			l.beta[i] = 0.0
		}
	}

	return l
}

// Forward performs a forward pass through the layer normalization layer.
// x: input tensor of shape [..., normalizedShape]
// Returns: normalized output with same shape as input
func (l *LayerNorm) Forward(x []float64) []float64 {
	// Assume input is [..., normalizedShape]
	// We normalize across the last dimension

	// Compute mean and std for each sample in the batch
	numSamples := len(x) / l.normalizedShape

	// Ensure output buffer is sized correctly
	if len(l.outputBuf) != len(x) {
		l.outputBuf = make([]float64, len(x))
	}
	if len(l.gradInBuf) != len(x) {
		l.gradInBuf = make([]float64, len(x))
	}

	output := l.outputBuf

	// Save input for backward pass
	if cap(l.savedInput) < len(x) {
		l.savedInput = make([]float64, len(x))
	}
	copy(l.savedInput, x)

	for s := 0; s < numSamples; s++ {
		start := s * l.normalizedShape
		end := start + l.normalizedShape

		// Compute mean
		sum := 0.0
		for i := start; i < end; i++ {
			sum += x[i]
		}
		mean := sum / float64(l.normalizedShape)
		l.inputMean = mean

		// Compute std
		sumSquares := 0.0
		for i := start; i < end; i++ {
			diff := x[i] - mean
			sumSquares += diff * diff
		}
		variance := sumSquares / float64(l.normalizedShape)
		std := math.Sqrt(variance + l.eps)
		l.inputStd = std

		// Normalize and apply affine transformation
		for i := start; i < end; i++ {
			normalized := (x[i] - mean) / std
			if l.elementwiseAffine {
				output[i] = l.gamma[i-start]*normalized + l.beta[i-start]
			} else {
				output[i] = normalized
			}
		}
	}

	return output
}

// Backward performs backpropagation through the layer normalization layer.
func (l *LayerNorm) Backward(grad []float64) []float64 {
	numSamples := len(grad) / l.normalizedShape

	// Ensure gradInBuf is sized correctly
	if len(l.gradInBuf) != len(grad) {
		l.gradInBuf = make([]float64, len(grad))
	}

	mean := l.inputMean
	std := l.inputStd

	for s := 0; s < numSamples; s++ {
		start := s * l.normalizedShape
		end := start + l.normalizedShape

		// Compute gradient w.r.t. input
		// dL/dx = (dL/dy * gamma) / std - mean(dL/dy * gamma) / std
		//       - (x - mean) * mean((dL/dy * gamma) * (x - mean)) / std^3

		// First compute dL/dy * gamma
		gammaProd := l.gradBuf
		for i := start; i < end; i++ {
			if l.elementwiseAffine {
				gammaProd[i-start] = grad[i] * l.gamma[i-start]
			} else {
				gammaProd[i-start] = grad[i]
			}
		}

		// Compute sum of gamma*grad and sum of gamma*grad*(x-mean)
		sumGrad := 0.0
		sumGradXMean := 0.0
		for i := start; i < end; i++ {
			diff := (l.savedInput[i] - mean)
			sumGrad += gammaProd[i-start]
			sumGradXMean += gammaProd[i-start] * diff
		}

		// Compute gradient for each input
		for i := start; i < end; i++ {
			diff := (l.savedInput[i] - mean)
			gradInput := (gammaProd[i-start] - sumGrad/float64(l.normalizedShape)) / std
			gradInput -= (diff * sumGradXMean) / (float64(l.normalizedShape) * std * std)
			l.gradInBuf[i] = gradInput
		}

		// Accumulate parameter gradients
		if l.elementwiseAffine {
			for i := start; i < end; i++ {
				normalized := (l.savedInput[i] - mean) / std
				l.gradGammaBuf[i-start] += grad[i] * normalized
				l.gradBetaBuf[i-start] += grad[i]
			}
		}
	}

	return l.gradInBuf
}

// Params returns layer parameters (gamma and beta when affine is enabled).
func (l *LayerNorm) Params() []float64 {
	if !l.elementwiseAffine {
		return make([]float64, 0)
	}

	total := len(l.gamma) + len(l.beta)
	params := make([]float64, total)
	copy(params, l.gamma)
	copy(params[len(l.gamma):], l.beta)
	return params
}

// SetParams updates gamma and beta from a flattened slice.
func (l *LayerNorm) SetParams(params []float64) {
	if !l.elementwiseAffine {
		return
	}

	totalGamma := len(l.gamma)
	copy(l.gamma, params[:totalGamma])
	copy(l.beta, params[totalGamma:])
}

// Gradients returns layer gradients (gamma and beta gradients when affine is enabled).
func (l *LayerNorm) Gradients() []float64 {
	if !l.elementwiseAffine {
		return make([]float64, 0)
	}

	total := len(l.gradGammaBuf) + len(l.gradBetaBuf)
	gradients := make([]float64, total)
	copy(gradients, l.gradGammaBuf)
	copy(gradients[len(l.gradGammaBuf):], l.gradBetaBuf)
	return gradients
}

// SetGradients sets gradients from a flattened slice.
func (l *LayerNorm) SetGradients(gradients []float64) {
	if !l.elementwiseAffine {
		return
	}

	totalGamma := len(l.gradGammaBuf)
	copy(l.gradGammaBuf, gradients[:totalGamma])
	copy(l.gradBetaBuf, gradients[totalGamma:])
}

// InSize returns the input size.
func (l *LayerNorm) InSize() int {
	return l.normalizedShape
}

// OutSize returns the output size.
func (l *LayerNorm) OutSize() int {
	return l.normalizedShape
}

// Reset resets the layer normalization layer.
func (l *LayerNorm) Reset() {
	// No state to reset
}

// ClearGradients zeroes out the accumulated gradients.
func (l *LayerNorm) ClearGradients() {
	if l.elementwiseAffine {
		for i := range l.gradGammaBuf {
			l.gradGammaBuf[i] = 0
		}
		for i := range l.gradBetaBuf {
			l.gradBetaBuf[i] = 0
		}
	}
}

// Clone creates a deep copy of the layer normalization layer.
func (l *LayerNorm) Clone() Layer {
	newL := NewLayerNorm(l.normalizedShape, l.eps, l.elementwiseAffine)
	if l.elementwiseAffine {
		copy(newL.gamma, l.gamma)
		copy(newL.beta, l.beta)
	}
	return newL
}

// GetGamma returns the gamma parameters (for affine layers).
func (l *LayerNorm) GetGamma() []float64 {
	if !l.elementwiseAffine {
		return nil
	}
	return l.gamma
}

// GetBeta returns the beta parameters (for affine layers).
func (l *LayerNorm) GetBeta() []float64 {
	if !l.elementwiseAffine {
		return nil
	}
	return l.beta
}

// GetEps returns the epsilon value for numerical stability.
func (l *LayerNorm) GetEps() float64 {
	return l.eps
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For LayerNorm, gradients are already accumulated in Backward, so this just calls Backward.
func (l *LayerNorm) AccumulateBackward(grad []float64) []float64 {
	return l.Backward(grad)
}
