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
	eps             float32
	elementwiseAffine bool

	// Learnable parameters (when affine is enabled)
	gamma []float32
	beta  []float32

	// Pre-allocated buffers
	outputBuf      []float32
	gradInBuf      []float32
	gradGammaBuf   []float32
	gradBetaBuf    []float32
	gradBuf        []float32 // Temporary gradient buffer
	savedInput     []float32 // Input saved for backward pass
	inputMeans     []float32 // Stored means for backward
	inputStds      []float32 // Stored stds for backward

	device Device
}

// NewLayerNorm creates a new layer normalization layer.
// normalizedShape: the shape of the input from the expected input dimension
// eps: small value for numerical stability (default 1e-5)
// elementwiseAffine: whether to learn scale (gamma) and shift (beta) parameters
func NewLayerNorm(normalizedShape int, eps float32, elementwiseAffine bool) *LayerNorm {
	l := &LayerNorm{
		normalizedShape:   normalizedShape,
		eps:               eps,
		elementwiseAffine: elementwiseAffine,
		outputBuf:         make([]float32, normalizedShape),
		gradInBuf:         make([]float32, normalizedShape),
		gradBuf:           make([]float32, normalizedShape),
		device:            &CPUDevice{},
	}

	if elementwiseAffine {
		l.gamma = make([]float32, normalizedShape)
		l.beta = make([]float32, normalizedShape)
		l.gradGammaBuf = make([]float32, normalizedShape)
		l.gradBetaBuf = make([]float32, normalizedShape)

		// Initialize gamma to 1 and beta to 0
		for i := 0; i < normalizedShape; i++ {
			l.gamma[i] = 1.0
			l.beta[i] = 0.0
		}
	}

	return l
}

// SetDevice sets the computation device.
func (l *LayerNorm) SetDevice(device Device) {
	l.device = device
}

// Forward performs a forward pass through the layer normalization layer.
// x: input tensor of shape [..., normalizedShape]
// Returns: normalized output with same shape as input
func (l *LayerNorm) Forward(x []float32) []float32 {
	numSamples := len(x) / l.normalizedShape

	if len(l.outputBuf) != len(x) {
		l.outputBuf = make([]float32, len(x))
	}
	if len(l.inputMeans) != numSamples {
		l.inputMeans = make([]float32, numSamples)
		l.inputStds = make([]float32, numSamples)
	}

	output := l.outputBuf
	if cap(l.savedInput) < len(x) {
		l.savedInput = make([]float32, len(x))
	}
	copy(l.savedInput, x)

	for s := 0; s < numSamples; s++ {
		start := s * l.normalizedShape
		end := start + l.normalizedShape

		sum := float32(0.0)
		for i := start; i < end; i++ {
			sum += x[i]
		}
		mean := sum / float32(l.normalizedShape)
		l.inputMeans[s] = mean

		sumSquares := float32(0.0)
		for i := start; i < end; i++ {
			diff := x[i] - mean
			sumSquares += diff * diff
		}
		variance := sumSquares / float32(l.normalizedShape)
		std := float32(math.Sqrt(float64(variance + l.eps)))
		l.inputStds[s] = std

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

func (l *LayerNorm) Backward(grad []float32) []float32 {
	numSamples := len(grad) / l.normalizedShape

	if len(l.gradInBuf) != len(grad) {
		l.gradInBuf = make([]float32, len(grad))
	}

	for s := 0; s < numSamples; s++ {
		start := s * l.normalizedShape
		end := start + l.normalizedShape

		mean := l.inputMeans[s]
		std := l.inputStds[s]

		gammaProd := l.gradBuf
		for i := start; i < end; i++ {
			if l.elementwiseAffine {
				gammaProd[i-start] = grad[i] * l.gamma[i-start]
			} else {
				gammaProd[i-start] = grad[i]
			}
		}

		sumGrad := float32(0.0)
		sumGradXMean := float32(0.0)
		for i := start; i < end; i++ {
			diff := (l.savedInput[i] - mean)
			sumGrad += gammaProd[i-start]
			sumGradXMean += gammaProd[i-start] * diff
		}

		for i := start; i < end; i++ {
			diff := (l.savedInput[i] - mean)
			gradInput := (gammaProd[i-start] - sumGrad/float32(l.normalizedShape)) / std
			gradInput -= (diff * sumGradXMean) / (float32(l.normalizedShape) * std * std)
			l.gradInBuf[i] = gradInput
		}

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
func (l *LayerNorm) Params() []float32 {
	if !l.elementwiseAffine {
		return make([]float32, 0)
	}

	total := len(l.gamma) + len(l.beta)
	params := make([]float32, total)
	copy(params, l.gamma)
	copy(params[len(l.gamma):], l.beta)
	return params
}

// SetParams updates gamma and beta from a flattened slice.
func (l *LayerNorm) SetParams(params []float32) {
	if !l.elementwiseAffine {
		return
	}

	totalGamma := len(l.gamma)
	copy(l.gamma, params[:totalGamma])
	copy(l.beta, params[totalGamma:])
}

// Gradients returns layer gradients (gamma and beta gradients when affine is enabled).
func (l *LayerNorm) Gradients() []float32 {
	if !l.elementwiseAffine {
		return make([]float32, 0)
	}

	total := len(l.gradGammaBuf) + len(l.gradBetaBuf)
	gradients := make([]float32, total)
	copy(gradients, l.gradGammaBuf)
	copy(gradients[len(l.gradGammaBuf):], l.gradBetaBuf)
	return gradients
}

// SetGradients sets gradients from a flattened slice.
func (l *LayerNorm) SetGradients(gradients []float32) {
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
	newL.device = l.device
	return newL
}

// GetGamma returns the gamma parameters (for affine layers).
func (l *LayerNorm) GetGamma() []float32 {
	if !l.elementwiseAffine {
		return nil
	}
	return l.gamma
}

// GetBeta returns the beta parameters (for affine layers).
func (l *LayerNorm) GetBeta() []float32 {
	if !l.elementwiseAffine {
		return nil
	}
	return l.beta
}

// GetEps returns the epsilon value for numerical stability.
func (l *LayerNorm) GetEps() float32 {
	return l.eps
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For LayerNorm, gradients are already accumulated in Backward, so this just calls Backward.
func (l *LayerNorm) AccumulateBackward(grad []float32) []float32 {
	return l.Backward(grad)
}
