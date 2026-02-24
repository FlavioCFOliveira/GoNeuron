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

	// Metal buffers
	bufIn     *MetalBuffer
	bufOut    *MetalBuffer
	bufGamma  *MetalBuffer
	bufGradIn *MetalBuffer
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
	if normalizedShape <= 0 {
		return
	}
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

	if md, ok := r.device.(*MetalDevice); ok && md.IsAvailable() {
		r.bufGamma = md.CreateBuffer(r.gamma)
		r.bufOut = md.CreateEmptyBuffer(normalizedShape)
		r.bufIn = md.CreateEmptyBuffer(normalizedShape)
		r.bufGradIn = md.CreateEmptyBuffer(normalizedShape)
	}
}

// SetDevice sets the computation device.
func (r *RMSNorm) SetDevice(device Device) {
	r.device = device
	if md, ok := device.(*MetalDevice); ok && md.IsAvailable() && r.normalizedShape > 0 {
		if r.bufGamma == nil {
			r.bufGamma = md.CreateBuffer(r.gamma)
		}
		if r.bufOut == nil {
			r.bufOut = md.CreateEmptyBuffer(r.normalizedShape)
		}
		if r.bufIn == nil {
			r.bufIn = md.CreateEmptyBuffer(r.normalizedShape)
		}
		if r.bufGradIn == nil {
			r.bufGradIn = md.CreateEmptyBuffer(r.normalizedShape)
		}
	}
}

func (r *RMSNorm) syncGPU() {
	if md, ok := r.device.(*MetalDevice); ok && md.IsAvailable() {
		if r.bufGamma == nil {
			r.bufGamma = md.CreateBuffer(r.gamma)
		} else {
			r.bufGamma.Update(r.gamma)
		}
	}
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
		if cap(r.arena) < len(x) {
			r.arena = make([]float32, len(x))
		}
		r.arena = r.arena[:len(x)]
		copy(r.arena, x)
		r.savedInputOffsets = append(r.savedInputOffsets[:0], 0)
		r.savedRMSOffsets = append(r.savedRMSOffsets[:0], -1) // -1 indicates internal rmsBuf
	}

	output := r.outputBuf
	if md, ok := r.device.(*MetalDevice); ok && md.IsAvailable() {
		total := len(x)
		if r.bufIn == nil || r.bufIn.length < total {
			if r.bufIn != nil {
				r.bufIn.Free()
			}
			r.bufIn = md.CreateBuffer(x)
		} else {
			r.bufIn.Update(x)
		}
		if r.bufOut == nil || r.bufOut.length < total {
			if r.bufOut != nil {
				r.bufOut.Free()
			}
			r.bufOut = md.CreateEmptyBuffer(total)
		}
		md.RMSNormForwardPersistent(r.bufIn, r.bufOut, r.bufGamma, numSamples, r.normalizedShape, r.eps)
		if len(output) < total {
			r.outputBuf = make([]float32, total)
			output = r.outputBuf
		}
		r.bufOut.Read(output[:total])
	}

	for s := 0; s < numSamples; s++ {
		start := s * r.normalizedShape
		end := start + r.normalizedShape

		sumSq := float32(0.0)
		for i := start; i < end; i++ {
			sumSq += x[i] * x[i]
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(r.normalizedShape) + r.eps)))
		r.rmsBuf[s] = rms

		if md, ok := r.device.(*MetalDevice); ok && md.IsAvailable() {
			continue
		}

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
	savedInput := r.arena[inOff : inOff+len(grad)]

	var rmsBuf []float32
	if rmsOff == -1 {
		rmsBuf = r.rmsBuf
	} else {
		rmsBuf = r.arena[rmsOff : rmsOff+numSamples]
	}

	if len(r.gradInBuf) < len(grad) {
		r.gradInBuf = make([]float32, len(grad))
	}
	gradIn := r.gradInBuf[:len(grad)]

	if md, ok := r.device.(*MetalDevice); ok && md.IsAvailable() {
		total := len(grad)
		if r.bufIn == nil || r.bufIn.length < total {
			if r.bufIn != nil {
				r.bufIn.Free()
			}
			r.bufIn = md.CreateBuffer(savedInput)
		} else {
			r.bufIn.Update(savedInput)
		}
		if r.bufOut == nil || r.bufOut.length < total {
			if r.bufOut != nil {
				r.bufOut.Free()
			}
			r.bufOut = md.CreateBuffer(grad)
		} else {
			r.bufOut.Update(grad)
		}
		if r.bufGradIn == nil || r.bufGradIn.length < total {
			if r.bufGradIn != nil {
				r.bufGradIn.Free()
			}
			r.bufGradIn = md.CreateEmptyBuffer(total)
		}
		md.RMSNormBackwardPersistent(r.bufOut, r.bufIn, r.bufGradIn, r.bufGamma, numSamples, r.normalizedShape, r.eps)
		r.bufGradIn.Read(gradIn)
	} else {
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
				gradIn[i] = (r.gamma[i-start] * invRMS) * (grad[i] - savedInput[i]*factor)
			}
		}
	}

	for s := 0; s < numSamples; s++ {
		start := s * r.normalizedShape
		end := start + r.normalizedShape
		rms := rmsBuf[s]
		invRMS := 1.0 / rms
		for i := start; i < end; i++ {
			r.gradGammaBuf[i-start] += grad[i] * (savedInput[i] * invRMS)
		}
	}

	r.savedInputOffsets = r.savedInputOffsets[:ts]
	r.savedRMSOffsets = r.savedRMSOffsets[:ts]

	return r.gradInBuf[:len(grad)]
}

func (r *RMSNorm) ForwardBatch(x []float32, batchSize int) []float32 {
	return r.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (r *RMSNorm) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return r.ForwardWithArena(x, arena, offset)
	}
	inSize := r.normalizedShape
	outSize := r.normalizedShape
	if len(r.outputBuf) < batchSize*outSize {
		r.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := r.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(r.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return r.outputBuf[:batchSize*outSize]
}

func (r *RMSNorm) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return r.Backward(grad)
	}
	inSize := r.normalizedShape
	outSize := r.normalizedShape
	if len(r.gradInBuf) < batchSize*inSize {
		r.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		dx := r.Backward(grad[i*outSize : (i+1)*outSize])
		copy(r.gradInBuf[i*inSize:(i+1)*inSize], dx)
	}
	return r.gradInBuf[:batchSize*inSize]
}

func (r *RMSNorm) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return r.BackwardBatch(grad, batchSize)
}

func (r *RMSNorm) Params() []float32 {
	return r.params
}

func (r *RMSNorm) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if len(r.params) > 0 && &r.params[0] != &params[0] {
		if len(params) == len(r.params) {
			r.params = params
			r.gamma = params
		} else {
			copy(r.params, params)
		}
	}
	r.syncGPU()
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

	if md, ok := r.device.(*MetalDevice); ok && md.IsAvailable() && r.normalizedShape > 0 {
		newR.bufGamma = md.CreateBuffer(newR.gamma)
		newR.bufOut = md.CreateEmptyBuffer(r.normalizedShape)
		newR.bufIn = md.CreateEmptyBuffer(r.normalizedShape)
		newR.bufGradIn = md.CreateEmptyBuffer(r.normalizedShape)
	}

	return newR
}

func (r *RMSNorm) AccumulateBackward(grad []float32) []float32 {
	return r.Backward(grad)
}
