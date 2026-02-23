// Package layer provides neural network layer implementations.
package layer

import (
	"math"
)

// RMSNorm implements Root Mean Square Layer Normalization.
// It scales the input by the root mean square of the elements.
// y = (x / RMS(x)) * gamma
type RMSNorm struct {
	normalizedShape int
	eps             float32

	// Learnable parameters
	params []float32 // gamma
	gamma  []float32 // View of params

	// Gradient buffers
	grads        []float32 // gradGamma
	gradGammaBuf []float32 // View of grads
	gradInBuf    []float32

	// Buffers
	outputBuf []float32
	rmsBuf    []float32

	savedInputOffsets []int
	savedRMSOffsets   []int
	arena             []float32

	training bool
	device   Device
}

// NewRMSNorm creates a new RMS normalization layer.
func NewRMSNorm(normalizedShape int, eps float32) *RMSNorm {
	r := &RMSNorm{
		normalizedShape:   normalizedShape,
		eps:               eps,
		device:            &CPUDevice{},
		savedInputOffsets: make([]int, 0, 16),
		savedRMSOffsets:   make([]int, 0, 16),
	}

	if normalizedShape != -1 {
		r.Build(normalizedShape)
	}

	return r
}

// Build initializes the layer with the given input size.
func (r *RMSNorm) Build(normalizedShape int) {
	r.normalizedShape = normalizedShape
	r.params = make([]float32, normalizedShape)
	r.gamma = r.params
	for i := 0; i < normalizedShape; i++ {
		r.gamma[i] = 1.0
	}

	r.grads = make([]float32, normalizedShape)
	r.gradGammaBuf = r.grads

	r.outputBuf = make([]float32, normalizedShape)
	r.gradInBuf = make([]float32, normalizedShape)
}

// SetDevice sets the computation device.
func (r *RMSNorm) SetDevice(device Device) {
	r.device = device
}

// SetTraining sets whether the layer is in training mode.
func (r *RMSNorm) SetTraining(training bool) {
	r.training = training
}

func (r *RMSNorm) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	numSamples := len(x) / r.normalizedShape

	if len(r.outputBuf) < len(x) {
		r.outputBuf = make([]float32, len(x))
	}

	if arena != nil && offset != nil {
		r.arena = *arena
		total := len(x)
		required := total + numSamples
		if len(*arena) < *offset+required {
			newArena := make([]float32, (*offset+required)*2)
			copy(newArena, *arena)
			*arena = newArena
			r.arena = *arena
		}

		r.savedInputOffsets = append(r.savedInputOffsets, *offset)
		copy((*arena)[*offset:*offset+total], x)
		*offset += total

		r.savedRMSOffsets = append(r.savedRMSOffsets, *offset)
		r.rmsBuf = (*arena)[*offset : *offset+numSamples]
		*offset += numSamples
	} else {
		if len(r.rmsBuf) < numSamples {
			r.rmsBuf = make([]float32, numSamples)
		}
		// If no external arena, we don't save input here for simple Forward calls
		// but typically Forward calls ForwardWithArena(x, nil, nil)
		// We'll need a way to save input if we want Backward to work after Forward.
		// For now, let's follow LayerNorm pattern which uses an internal arena-like buffer if needed.
		r.arena = x // Temporary fallback
		r.savedInputOffsets = append(r.savedInputOffsets[:0], -2) // -2 indicates x itself
		r.savedRMSOffsets = append(r.savedRMSOffsets[:0], -1)   // -1 indicates internal rmsBuf
	}

	output := r.outputBuf
	for s := 0; s < numSamples; s++ {
		start := s * r.normalizedShape
		end := start + r.normalizedShape

		sumSq := float32(0.0)
		for i := start; i < end; i++ {
			sumSq += x[i] * x[i]
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(r.normalizedShape) + r.eps)))
		r.rmsBuf[s] = rms

		for i := start; i < end; i++ {
			output[i] = (x[i] / rms) * r.gamma[i-start]
		}
	}

	return output[:len(x)]
}

func (r *RMSNorm) Forward(x []float32) []float32 {
	return r.ForwardWithArena(x, nil, nil)
}

func (r *RMSNorm) Backward(grad []float32) []float32 {
	numSaved := len(r.savedInputOffsets)
	if numSaved == 0 {
		return nil
	}

	ts := numSaved - 1
	inOff := r.savedInputOffsets[ts]
	rmsOff := r.savedRMSOffsets[ts]

	numSamples := len(grad) / r.normalizedShape
	var savedInput []float32
	if inOff == -2 {
		savedInput = r.arena // x itself in Forward
	} else {
		savedInput = r.arena[inOff : inOff+len(grad)]
	}

	var rmsBuf []float32
	if rmsOff == -1 {
		rmsBuf = r.rmsBuf
	} else {
		rmsBuf = r.arena[rmsOff : rmsOff+numSamples]
	}

	if len(r.gradInBuf) < len(grad) {
		r.gradInBuf = make([]float32, len(grad))
	}

	for s := 0; s < numSamples; s++ {
		start := s * r.normalizedShape
		end := start + r.normalizedShape

		rms := rmsBuf[s]
		invRMS := 1.0 / rms

		// dL/dx_i = (gamma_i / RMS) * (grad_i - (x_i * sum(grad_j * gamma_j * x_j)) / (n * RMS^2))
		sumGradXGammaX := float32(0.0)
		for i := start; i < end; i++ {
			sumGradXGammaX += grad[i] * r.gamma[i-start] * savedInput[i]
		}

		factor := sumGradXGammaX / (float32(r.normalizedShape) * rms * rms)
		for i := start; i < end; i++ {
			r.gradInBuf[i] = (r.gamma[i-start] * invRMS) * (grad[i] - savedInput[i]*factor)
		}

		// Update gamma gradients
		for i := start; i < end; i++ {
			r.gradGammaBuf[i-start] += grad[i] * (savedInput[i] * invRMS)
		}
	}

	r.savedInputOffsets = r.savedInputOffsets[:ts]
	r.savedRMSOffsets = r.savedRMSOffsets[:ts]

	return r.gradInBuf[:len(grad)]
}

func (r *RMSNorm) Params() []float32 {
	return r.params
}

func (r *RMSNorm) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &r.params[0] != &params[0] {
		if len(params) == len(r.params) {
			r.params = params
			r.gamma = params
		} else {
			copy(r.params, params)
		}
	}
}

func (r *RMSNorm) Gradients() []float32 {
	return r.grads
}

func (r *RMSNorm) SetGradients(gradients []float32) {
	if len(gradients) == 0 {
		return
	}
	if &r.grads[0] != &gradients[0] {
		if len(gradients) == len(r.grads) {
			r.grads = gradients
			r.gradGammaBuf = gradients
		} else {
			copy(r.grads, gradients)
		}
	}
}

func (r *RMSNorm) InSize() int {
	return r.normalizedShape
}

func (r *RMSNorm) OutSize() int {
	return r.normalizedShape
}

func (r *RMSNorm) NamedParams() []NamedParam {
	return []NamedParam{
		{
			Name:  "gamma",
			Shape: []int{r.normalizedShape},
			Data:  r.gamma,
		},
	}
}

func (r *RMSNorm) Reset() {
	r.savedInputOffsets = r.savedInputOffsets[:0]
	r.savedRMSOffsets = r.savedRMSOffsets[:0]
}

func (r *RMSNorm) ClearGradients() {
	for i := range r.grads {
		r.grads[i] = 0
	}
}

func (r *RMSNorm) Clone() Layer {
	newR := NewRMSNorm(r.normalizedShape, r.eps)
	copy(newR.params, r.params)
	newR.device = r.device
	return newR
}

func (r *RMSNorm) LightweightClone(params []float32, grads []float32) Layer {
	newR := &RMSNorm{
		normalizedShape:   r.normalizedShape,
		eps:               r.eps,
		params:            params,
		gamma:             params,
		grads:             grads,
		gradGammaBuf:      grads,
		outputBuf:         make([]float32, r.normalizedShape),
		gradInBuf:         make([]float32, r.normalizedShape),
		savedInputOffsets: make([]int, 0, 128),
		savedRMSOffsets:   make([]int, 0, 128),
		training:          r.training,
		device:            r.device,
	}
	return newR
}

func (r *RMSNorm) AccumulateBackward(grad []float32) []float32 {
	return r.Backward(grad)
}
