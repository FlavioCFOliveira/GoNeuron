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

	savedInputOffsets []int
	savedMeanOffsets  []int
	savedStdOffsets   []int
	arena             []float32

	training bool

	device Device

	// Metal buffers
	bufIn     *MetalBuffer
	bufOut    *MetalBuffer
	bufGamma  *MetalBuffer
	bufBeta   *MetalBuffer
	bufGradIn *MetalBuffer
}

// NewLayerNorm creates a new layer normalization layer.
func NewLayerNorm(normalizedShape int, eps float32, elementwiseAffine bool) *LayerNorm {
	l := &LayerNorm{
		normalizedShape:   normalizedShape,
		eps:               eps,
		elementwiseAffine: elementwiseAffine,
		device:            &CPUDevice{},
		savedInputOffsets: make([]int, 0, 16),
		savedMeanOffsets:  make([]int, 0, 16),
		savedStdOffsets:   make([]int, 0, 16),
	}

	if normalizedShape != -1 {
		l.Build(normalizedShape)
	}

	return l
}

// Build initializes the layer with the given input size.
func (l *LayerNorm) Build(normalizedShape int) {
	if normalizedShape <= 0 {
		return
	}
	l.normalizedShape = normalizedShape
	if l.elementwiseAffine {
		l.params = make([]float32, normalizedShape*2)
		l.gamma = l.params[:normalizedShape]
		l.beta = l.params[normalizedShape:]
		for i := 0; i < normalizedShape; i++ {
			l.gamma[i] = 1.0
			l.beta[i] = 0.0
		}
		l.grads = make([]float32, normalizedShape*2)
		l.gradGammaBuf = l.grads[:normalizedShape]
		l.gradBetaBuf = l.grads[normalizedShape:]
	} else {
		l.params = make([]float32, 0)
		l.grads = make([]float32, 0)
	}

	l.outputBuf = make([]float32, normalizedShape)
	l.gradInBuf = make([]float32, normalizedShape)
	l.gradBuf = make([]float32, normalizedShape)

	// Metal initialization if device is GPU
	if md, ok := l.device.(*MetalDevice); ok && md.IsAvailable() && l.normalizedShape > 0 {
		if l.elementwiseAffine {
			l.bufGamma = md.CreateBuffer(l.gamma)
			l.bufBeta = md.CreateBuffer(l.beta)
		}
		l.bufIn = md.CreateEmptyBuffer(normalizedShape)
		l.bufOut = md.CreateEmptyBuffer(normalizedShape)
		l.bufGradIn = md.CreateEmptyBuffer(normalizedShape)
	}
}

// SetDevice sets the computation device.
func (l *LayerNorm) SetDevice(device Device) {
	l.device = device
	if md, ok := device.(*MetalDevice); ok && md.IsAvailable() && l.normalizedShape > 0 {
		if l.elementwiseAffine {
			if l.bufGamma == nil {
				l.bufGamma = md.CreateBuffer(l.gamma)
			}
			if l.bufBeta == nil {
				l.bufBeta = md.CreateBuffer(l.beta)
			}
		}
		if l.bufIn == nil {
			l.bufIn = md.CreateEmptyBuffer(l.normalizedShape)
		}
		if l.bufOut == nil {
			l.bufOut = md.CreateEmptyBuffer(l.normalizedShape)
		}
		if l.bufGradIn == nil {
			l.bufGradIn = md.CreateEmptyBuffer(l.normalizedShape)
		}
	}
}

func (l *LayerNorm) syncGPU() {
	if md, ok := l.device.(*MetalDevice); ok && md.IsAvailable() && l.elementwiseAffine {
		if l.bufGamma == nil {
			l.bufGamma = md.CreateBuffer(l.gamma)
		} else {
			l.bufGamma.Update(l.gamma)
		}
		if l.bufBeta == nil {
			l.bufBeta = md.CreateBuffer(l.beta)
		} else {
			l.bufBeta.Update(l.beta)
		}
	}
}

// SetTraining sets whether the layer is in training mode.
func (l *LayerNorm) SetTraining(training bool) {
	l.training = training
}

func (l *LayerNorm) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	numSamples := len(x) / l.normalizedShape

	if len(l.outputBuf) < len(x) {
		l.outputBuf = make([]float32, len(x))
	}

	if arena != nil && offset != nil {
		l.arena = *arena
		// Save input, means and stds to arena
		total := len(x)
		required := total + numSamples*2
		if len(*arena) < *offset+required {
			newArena := make([]float32, (*offset+required)*2)
			copy(newArena, *arena)
			*arena = newArena
			l.arena = *arena
		}

		l.savedInputOffsets = append(l.savedInputOffsets, *offset)
		copy((*arena)[*offset:*offset+total], x)
		*offset += total

		l.savedMeanOffsets = append(l.savedMeanOffsets, *offset)
		l.inputMeans = (*arena)[*offset : *offset+numSamples]
		*offset += numSamples

		l.savedStdOffsets = append(l.savedStdOffsets, *offset)
		l.inputStds = (*arena)[*offset : *offset+numSamples]
		*offset += numSamples
	} else {
		if len(l.inputMeans) < numSamples {
			l.inputMeans = make([]float32, numSamples)
			l.inputStds = make([]float32, numSamples)
		}
		if cap(l.savedInput) < len(x) {
			l.savedInput = make([]float32, len(x))
		}
		l.savedInput = l.savedInput[:len(x)]
		copy(l.savedInput, x)

		l.savedInputOffsets = append(l.savedInputOffsets[:0], 0)
		l.savedMeanOffsets = append(l.savedMeanOffsets[:0], -1) // -1 indicates not in arena
		l.savedStdOffsets = append(l.savedStdOffsets[:0], -1)
		l.arena = l.savedInput
	}

	output := l.outputBuf
	if md, ok := l.device.(*MetalDevice); ok && md.IsAvailable() {
		total := len(x)
		if l.bufIn == nil || l.bufIn.length < total {
			if l.bufIn != nil {
				l.bufIn.Free()
			}
			l.bufIn = md.CreateBuffer(x)
		} else {
			l.bufIn.Update(x)
		}
		if l.bufOut == nil || l.bufOut.length < total {
			if l.bufOut != nil {
				l.bufOut.Free()
			}
			l.bufOut = md.CreateEmptyBuffer(total)
		}
		md.LayerNormForwardPersistent(l.bufIn, l.bufOut, l.bufGamma, l.bufBeta, numSamples, l.normalizedShape, l.eps)
		if len(output) < total {
			l.outputBuf = make([]float32, total)
			output = l.outputBuf
		}
		l.bufOut.Read(output[:total])
		// We still need to compute means and stds for backward if training
	}

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

		if md, ok := l.device.(*MetalDevice); ok && md.IsAvailable() {
			// Skip actual computation if Metal was used
			continue
		}

		for i := start; i < end; i++ {
			normalized := (x[i] - mean) / std
			if l.elementwiseAffine {
				output[i] = l.gamma[i-start]*normalized + l.beta[i-start]
			} else {
				output[i] = normalized
			}
		}
	}

	return output[:len(x)]
}

// Forward performs a forward pass through the layer normalization layer.
func (l *LayerNorm) Forward(x []float32) []float32 {
	return l.ForwardWithArena(x, nil, nil)
}

func (l *LayerNorm) ForwardBatch(x []float32, batchSize int) []float32 {
	return l.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (l *LayerNorm) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return l.ForwardWithArena(x, arena, offset)
	}
	inSize := l.normalizedShape
	outSize := l.normalizedShape
	if len(l.outputBuf) < batchSize*outSize {
		l.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := l.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(l.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return l.outputBuf[:batchSize*outSize]
}

func (l *LayerNorm) Backward(grad []float32) []float32 {
	numSaved := len(l.savedInputOffsets)
	if numSaved == 0 {
		return nil
	}

	ts := numSaved - 1
	inOff := l.savedInputOffsets[ts]
	meanOff := l.savedMeanOffsets[ts]
	stdOff := l.savedStdOffsets[ts]

	numSamples := len(grad) / l.normalizedShape
	savedInput := l.arena[inOff : inOff+len(grad)]

	var inputMeans, inputStds []float32
	if meanOff == -1 {
		inputMeans = l.inputMeans
		inputStds = l.inputStds
	} else {
		inputMeans = l.arena[meanOff : meanOff+numSamples]
		inputStds = l.arena[stdOff : stdOff+numSamples]
	}

	if len(l.gradInBuf) < len(grad) {
		l.gradInBuf = make([]float32, len(grad))
	}
	gradIn := l.gradInBuf[:len(grad)]

	if md, ok := l.device.(*MetalDevice); ok && md.IsAvailable() {
		total := len(grad)
		if l.bufIn == nil || l.bufIn.length < total {
			if l.bufIn != nil {
				l.bufIn.Free()
			}
			l.bufIn = md.CreateBuffer(savedInput)
		} else {
			l.bufIn.Update(savedInput)
		}
		if l.bufOut == nil || l.bufOut.length < total {
			if l.bufOut != nil {
				l.bufOut.Free()
			}
			l.bufOut = md.CreateBuffer(grad)
		} else {
			l.bufOut.Update(grad)
		}
		if l.bufGradIn == nil || l.bufGradIn.length < total {
			if l.bufGradIn != nil {
				l.bufGradIn.Free()
			}
			l.bufGradIn = md.CreateEmptyBuffer(total)
		}
		md.LayerNormBackwardPersistent(l.bufOut, l.bufIn, l.bufGradIn, l.bufGamma, numSamples, l.normalizedShape, l.eps)
		l.bufGradIn.Read(gradIn)
	} else {
		for s := 0; s < numSamples; s++ {
			start := s * l.normalizedShape
			end := start + l.normalizedShape

			mean := inputMeans[s]
			std := inputStds[s]

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
				diff := (savedInput[i] - mean)
				sumGrad += gammaProd[i-start]
				sumGradXMean += gammaProd[i-start] * diff
			}

			for i := start; i < end; i++ {
				diff := (savedInput[i] - mean)
				gradIn[i] = (gammaProd[i-start] - sumGrad/float32(l.normalizedShape)) / std
				gradIn[i] -= (diff * sumGradXMean) / (float32(l.normalizedShape) * std * std)
			}
		}
	}

	if l.elementwiseAffine {
		for s := 0; s < numSamples; s++ {
			start := s * l.normalizedShape
			end := start + l.normalizedShape
			mean := inputMeans[s]
			std := inputStds[s]
			for i := start; i < end; i++ {
				normalized := (savedInput[i] - mean) / std
				l.gradGammaBuf[i-start] += grad[i] * normalized
				l.gradBetaBuf[i-start] += grad[i]
			}
		}
	}

	// Pop offsets
	l.savedInputOffsets = l.savedInputOffsets[:ts]
	l.savedMeanOffsets = l.savedMeanOffsets[:ts]
	l.savedStdOffsets = l.savedStdOffsets[:ts]

	return l.gradInBuf[:len(grad)]
}

func (l *LayerNorm) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return l.Backward(grad)
	}
	inSize := l.normalizedShape
	outSize := l.normalizedShape
	if len(l.gradInBuf) < batchSize*inSize {
		l.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		dx := l.Backward(grad[i*outSize : (i+1)*outSize])
		copy(l.gradInBuf[i*inSize:(i+1)*inSize], dx)
	}
	return l.gradInBuf[:batchSize*inSize]
}

func (l *LayerNorm) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return l.BackwardBatch(grad, batchSize)
}

func (l *LayerNorm) Params() []float32 {
	return l.params
}

func (l *LayerNorm) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if len(l.params) > 0 && &l.params[0] != &params[0] {
		if len(params) == len(l.params) {
			l.params = params
			l.updateViews()
		} else {
			copy(l.params, params)
		}
	}
	l.syncGPU()
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
	if len(l.grads) > 0 && &l.grads[0] != &gradients[0] {
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
	l.savedInputOffsets = l.savedInputOffsets[:0]
	l.savedMeanOffsets = l.savedMeanOffsets[:0]
	l.savedStdOffsets = l.savedStdOffsets[:0]
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
		savedInputOffsets: make([]int, 0, 128),
		savedMeanOffsets:  make([]int, 0, 128),
		savedStdOffsets:   make([]int, 0, 128),
		training:          l.training,
		device:            l.device,
	}

	if md, ok := l.device.(*MetalDevice); ok && md.IsAvailable() && l.normalizedShape > 0 {
		if l.elementwiseAffine {
			newL.bufGamma = md.CreateBuffer(newL.gamma)
			newL.bufBeta = md.CreateBuffer(newL.beta)
		}
		newL.bufIn = md.CreateEmptyBuffer(l.normalizedShape)
		newL.bufOut = md.CreateEmptyBuffer(l.normalizedShape)
		newL.bufGradIn = md.CreateEmptyBuffer(l.normalizedShape)
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
