// Package layer provides neural network layer implementations.
package layer

import (
	"math"
)

// LayerNorm implements layer normalization.
type LayerNorm struct {
	// Normalization parameters
	normalizedShape   int
	eps               float32
	elementwiseAffine bool

	// Learnable parameters
	params []float32 // Contiguous gamma + beta
	gamma  []float32 // View of params
	beta   []float32 // View of params

	// Gradient buffers
	grads        []float32 // Contiguous gradGamma + gradBeta
	gradGammaBuf []float32 // View of grads
	gradBetaBuf  []float32 // View of grads
	gradInBuf    []float32
	gradBuf      []float32 // Temporary gradient buffer

	// Buffers
	outputBuf  []float32
	savedInput []float32
	inputMeans []float32
	inputStds  []float32

	training bool

	device Device
}

// NewLayerNorm creates a new layer normalization layer.
func NewLayerNorm(normalizedShape int, eps float32, elementwiseAffine bool) *LayerNorm {
	params := make([]float32, 0)
	var gamma, beta []float32
	grads := make([]float32, 0)
	var gradGamma, gradBeta []float32

	if elementwiseAffine {
		params = make([]float32, normalizedShape*2)
		gamma = params[:normalizedShape]
		beta = params[normalizedShape:]
		for i := 0; i < normalizedShape; i++ {
			gamma[i] = 1.0
			beta[i] = 0.0
		}
		grads = make([]float32, normalizedShape*2)
		gradGamma = grads[:normalizedShape]
		gradBeta = grads[normalizedShape:]
	}

	return &LayerNorm{
		normalizedShape:   normalizedShape,
		eps:               eps,
		elementwiseAffine: elementwiseAffine,
		params:            params,
		gamma:             gamma,
		beta:              beta,
		grads:             grads,
		gradGammaBuf:      gradGamma,
		gradBetaBuf:       gradBeta,
		outputBuf:         make([]float32, normalizedShape),
		gradInBuf:         make([]float32, normalizedShape),
		gradBuf:           make([]float32, normalizedShape),
		device:            &CPUDevice{},
	}
}

// SetDevice sets the computation device.
func (l *LayerNorm) SetDevice(device Device) {
	l.device = device
}

// SetTraining sets whether the layer is in training mode.
func (l *LayerNorm) SetTraining(training bool) {
	l.training = training
}

func (l *LayerNorm) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	numSamples := len(x) / l.normalizedShape

	if len(l.outputBuf) != len(x) {
		l.outputBuf = make([]float32, len(x))
	}

	if arena != nil && offset != nil {
		// Save input, means and stds to arena
		total := len(x)
		required := total + numSamples*2
		if len(*arena) < *offset+required {
			newArena := make([]float32, (*offset+required)*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		savedIn := (*arena)[*offset : *offset+total]
		copy(savedIn, x)
		l.savedInput = savedIn
		*offset += total

		l.inputMeans = (*arena)[*offset : *offset+numSamples]
		*offset += numSamples

		l.inputStds = (*arena)[*offset : *offset+numSamples]
		*offset += numSamples
	} else {
		if len(l.inputMeans) != numSamples {
			l.inputMeans = make([]float32, numSamples)
			l.inputStds = make([]float32, numSamples)
		}
		if cap(l.savedInput) < len(x) {
			l.savedInput = make([]float32, len(x))
		}
		l.savedInput = l.savedInput[:len(x)]
		copy(l.savedInput, x)
	}

	output := l.outputBuf
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

// Forward performs a forward pass through the layer normalization layer.
func (l *LayerNorm) Forward(x []float32) []float32 {
	return l.ForwardWithArena(x, nil, nil)
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

func (l *LayerNorm) Params() []float32 {
	return l.params
}

func (l *LayerNorm) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &l.params[0] != &params[0] {
		if len(params) == len(l.params) {
			l.params = params
			l.updateViews()
		} else {
			copy(l.params, params)
		}
	}
}

func (l *LayerNorm) updateViews() {
	if l.elementwiseAffine {
		l.gamma = l.params[:l.normalizedShape]
		l.beta = l.params[l.normalizedShape:]
	}
}

func (l *LayerNorm) Gradients() []float32 {
	return l.grads
}

func (l *LayerNorm) SetGradients(gradients []float32) {
	if len(gradients) == 0 {
		return
	}
	if &l.grads[0] != &gradients[0] {
		if len(gradients) == len(l.grads) {
			l.grads = gradients
			l.updateGradViews()
		} else {
			copy(l.grads, gradients)
		}
	}
}

func (l *LayerNorm) updateGradViews() {
	if l.elementwiseAffine {
		l.gradGammaBuf = l.grads[:l.normalizedShape]
		l.gradBetaBuf = l.grads[l.normalizedShape:]
	}
}

// InSize returns the input size.
func (l *LayerNorm) InSize() int {
	return l.normalizedShape
}

// OutSize returns the output size.
func (l *LayerNorm) OutSize() int {
	return l.normalizedShape
}

func (l *LayerNorm) NamedParams() []NamedParam {
	if !l.elementwiseAffine {
		return nil
	}

	return []NamedParam{
		{
			Name:  "gamma",
			Shape: []int{l.normalizedShape},
			Data:  l.gamma,
		},
		{
			Name:  "beta",
			Shape: []int{l.normalizedShape},
			Data:  l.beta,
		},
	}
}

// Reset resets the layer normalization layer.
func (l *LayerNorm) Reset() {
	l.savedInput = nil
	l.inputMeans = nil
	l.inputStds = nil
}

// ClearGradients zeroes out the accumulated gradients.
func (l *LayerNorm) ClearGradients() {
	for i := range l.grads {
		l.grads[i] = 0
	}
}

// Clone creates a deep copy of the layer normalization layer.
func (l *LayerNorm) Clone() Layer {
	newL := NewLayerNorm(l.normalizedShape, l.eps, l.elementwiseAffine)
	if l.elementwiseAffine {
		copy(newL.params, l.params)
	}
	newL.device = l.device
	return newL
}

func (l *LayerNorm) LightweightClone(params []float32, grads []float32) Layer {
	var gamma, beta, gradGamma, gradBeta []float32
	if l.elementwiseAffine {
		gamma = params[:l.normalizedShape]
		beta = params[l.normalizedShape:]
		gradGamma = grads[:l.normalizedShape]
		gradBeta = grads[l.normalizedShape:]
	}

	newL := &LayerNorm{
		normalizedShape:   l.normalizedShape,
		eps:               l.eps,
		elementwiseAffine: l.elementwiseAffine,
		params:            params,
		gamma:             gamma,
		beta:              beta,
		grads:             grads,
		gradGammaBuf:      gradGamma,
		gradBetaBuf:       gradBeta,
		outputBuf:         make([]float32, l.normalizedShape),
		gradInBuf:         make([]float32, l.normalizedShape),
		gradBuf:           make([]float32, l.normalizedShape),
		training:          l.training,
		device:            l.device,
	}
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
