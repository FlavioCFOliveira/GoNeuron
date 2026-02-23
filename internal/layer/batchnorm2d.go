// Package layer provides neural network layer implementations.
package layer

import (
	"math"
)

// BatchNorm2D implements 2D batch normalization.
// Normalizes across batch and spatial dimensions, learns mean/std per channel.
type BatchNorm2D struct {
	// Normalization parameters
	numFeatures int
	eps         float32
	momentum    float32
	affine      bool

	// Training mode
	training bool

	// Learnable parameters (when affine is enabled)
	params []float32 // Contiguous gamma + beta
	gamma  []float32 // View of params
	beta   []float32 // View of params

	// Running statistics (for inference)
	runningMean []float32
	runningVar  []float32

	// Gradient buffers
	grads        []float32 // Contiguous gradGamma + gradBeta
	gradGammaBuf []float32 // View of grads
	gradBetaBuf  []float32 // View of grads
	gradInBuf    []float32
	outputBuf    []float32
	savedInput   []float32
	savedMean    []float32
	savedVar     []float32
	savedStd     []float32

	savedInputOffsets []int
	savedMeanOffsets  []int
	savedStdOffsets   []int
	arena             []float32

	inputHeight     int
	inputWidth      int
	numelPerChannel int
	device          Device
}

// NewBatchNorm2D creates a new 2D batch normalization layer.
func NewBatchNorm2D(numFeatures int, eps float32, momentum float32, affine bool) *BatchNorm2D {
	l := &BatchNorm2D{
		numFeatures:       numFeatures,
		eps:               eps,
		momentum:          momentum,
		affine:            affine,
		training:          true,
		device:            &CPUDevice{},
		savedInputOffsets: make([]int, 0, 16),
		savedMeanOffsets:  make([]int, 0, 16),
		savedStdOffsets:   make([]int, 0, 16),
	}

	if numFeatures != -1 {
		l.Build(numFeatures)
	}

	return l
}

// Build initializes the layer with the given input size (channels).
func (b *BatchNorm2D) Build(numFeatures int) {
	b.numFeatures = numFeatures
	if b.affine {
		b.params = make([]float32, numFeatures*2)
		b.gamma = b.params[:numFeatures]
		b.beta = b.params[numFeatures:]
		for i := 0; i < numFeatures; i++ {
			b.gamma[i] = 1.0
			b.beta[i] = 0.0
		}
		b.grads = make([]float32, numFeatures*2)
		b.gradGammaBuf = b.grads[:numFeatures]
		b.gradBetaBuf = b.grads[numFeatures:]
	} else {
		b.params = make([]float32, 0)
		b.grads = make([]float32, 0)
	}

	b.runningMean = make([]float32, numFeatures)
	b.runningVar = make([]float32, numFeatures)
	for i := range b.runningVar {
		b.runningVar[i] = 1.0
	}

	b.outputBuf = make([]float32, 0)
	b.gradInBuf = make([]float32, 0)
	b.savedMean = make([]float32, numFeatures)
	b.savedVar = make([]float32, numFeatures)
	b.savedStd = make([]float32, numFeatures)
}

// SetInputDimensions sets the spatial dimensions for the layer.
func (b *BatchNorm2D) SetInputDimensions(height, width int) {
	b.inputHeight = height
	b.inputWidth = width
	b.numelPerChannel = height * width
}

// GetOutputDimensions returns the spatial dimensions of the output.
func (b *BatchNorm2D) GetOutputDimensions() (int, int) {
	return b.inputHeight, b.inputWidth
}

// SetDevice sets the computation device.
func (b *BatchNorm2D) SetDevice(device Device) {
	b.device = device
}

func (b *BatchNorm2D) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	if len(x) == 0 {
		return make([]float32, 0)
	}

	// Infer dimensions
	total := len(x)
	spatialSize := total / (b.numFeatures)
	b.numelPerChannel = spatialSize

	// Ensure output buffer is sized correctly
	if len(b.outputBuf) != total {
		b.outputBuf = make([]float32, total)
	}
	if len(b.gradInBuf) != total {
		b.gradInBuf = make([]float32, total)
	}

	output := b.outputBuf

	// Save input for backward pass
	if b.training {
		if arena != nil && offset != nil {
			b.arena = *arena
			totalRequired := total + b.numFeatures*2
			if len(*arena) < *offset+totalRequired {
				newArena := make([]float32, (*offset+totalRequired)*2)
				copy(newArena, *arena)
				*arena = newArena
				b.arena = *arena
			}
			b.savedInputOffsets = append(b.savedInputOffsets, *offset)
			copy((*arena)[*offset:*offset+total], x)
			*offset += total

			b.savedMeanOffsets = append(b.savedMeanOffsets, *offset)
			b.savedMean = (*arena)[*offset : *offset+b.numFeatures]
			*offset += b.numFeatures

			b.savedStdOffsets = append(b.savedStdOffsets, *offset)
			b.savedStd = (*arena)[*offset : *offset+b.numFeatures]
			*offset += b.numFeatures
		} else {
			if cap(b.savedInput) < total {
				b.savedInput = make([]float32, total)
			}
			b.savedInput = b.savedInput[:total]
			copy(b.savedInput, x)

			b.savedInputOffsets = append(b.savedInputOffsets[:0], 0)
			b.savedMeanOffsets = append(b.savedMeanOffsets[:0], -1)
			b.savedStdOffsets = append(b.savedStdOffsets[:0], -1)
			b.arena = b.savedInput
		}
	}

	numel := b.numelPerChannel
	numFeatures := b.numFeatures

	if b.training {
		for f := 0; f < numFeatures; f++ {
			// Compute batch mean
			sum := float32(0.0)
			for s := 0; s < numel; s++ {
				idx := f*numel + s
				sum += x[idx]
			}
			mean := sum / float32(numel)
			b.savedMean[f] = mean
			b.runningMean[f] = (1-b.momentum)*b.runningMean[f] + b.momentum*mean

			// Compute batch variance
			sumSquares := float32(0.0)
			for s := 0; s < numel; s++ {
				idx := f*numel + s
				diff := x[idx] - mean
				sumSquares += diff * diff
			}
			variance := sumSquares / float32(numel)
			std := float32(math.Sqrt(float64(variance + b.eps)))
			b.savedVar[f] = variance
			b.savedStd[f] = std
			b.runningVar[f] = (1-b.momentum)*b.runningVar[f] + b.momentum*variance

			// Normalize and apply affine transformation
			for s := 0; s < numel; s++ {
				idx := f*numel + s
				normalized := (x[idx] - mean) / std
				if b.affine {
					output[idx] = b.gamma[f]*normalized + b.beta[f]
				} else {
					output[idx] = normalized
				}
			}
		}
	} else {
		// Inference mode: use running statistics
		for f := 0; f < numFeatures; f++ {
			mean := b.runningMean[f]
			variance := b.runningVar[f]
			std := float32(math.Sqrt(float64(variance + b.eps)))

			for s := 0; s < numel; s++ {
				idx := f*numel + s
				normalized := (x[idx] - mean) / std
				if b.affine {
					output[idx] = b.gamma[f]*normalized + b.beta[f]
				} else {
					output[idx] = normalized
				}
			}
		}
	}

	return output
}

// Forward performs a forward pass through the batch normalization layer.
func (b *BatchNorm2D) Forward(x []float32) []float32 {
	return b.ForwardWithArena(x, nil, nil)
}


// Backward performs backpropagation through the batch normalization layer.
func (b *BatchNorm2D) Backward(grad []float32) []float32 {
	if !b.training {
		// Gradient in inference mode: simple scaling
		numel := b.numelPerChannel
		numFeatures := b.numFeatures
		if len(b.gradInBuf) != len(grad) {
			b.gradInBuf = make([]float32, len(grad))
		}
		for f := 0; f < numFeatures; f++ {
			variance := b.runningVar[f]
			std := float32(math.Sqrt(float64(variance + b.eps)))
			for s := 0; s < numel; s++ {
				idx := f*numel + s
				g := grad[idx]
				if b.affine {
					g *= b.gamma[f]
				}
				b.gradInBuf[idx] = g / std
			}
		}
		return b.gradInBuf
	}

	numSaved := len(b.savedInputOffsets)
	if numSaved == 0 {
		return nil
	}

	ts := numSaved - 1
	inOff := b.savedInputOffsets[ts]
	meanOff := b.savedMeanOffsets[ts]
	stdOff := b.savedStdOffsets[ts]

	total := len(grad)
	savedInput := b.arena[inOff : inOff+total]

	numel := b.numelPerChannel
	numFeatures := b.numFeatures

	var savedMean, savedStd []float32
	if meanOff == -1 {
		savedMean = b.savedMean
		savedStd = b.savedStd
	} else {
		savedMean = b.arena[meanOff : meanOff+numFeatures]
		savedStd = b.arena[stdOff : stdOff+numFeatures]
	}

	// Ensure gradInBuf is sized correctly
	if len(b.gradInBuf) < len(grad) {
		b.gradInBuf = make([]float32, len(grad))
	}

	for f := 0; f < numFeatures; f++ {
		mean := savedMean[f]
		std := savedStd[f]

		// Compute sum of gradients and sum of gradients * (x - mean)
		sumGrad := float32(0.0)
		sumGradXMean := float32(0.0)
		for s := 0; s < numel; s++ {
			idx := f*numel + s
			diff := savedInput[idx] - mean
			sumGrad += grad[idx]
			sumGradXMean += grad[idx] * diff
		}

		// Compute gradient for each input
		for s := 0; s < numel; s++ {
			idx := f*numel + s
			diff := savedInput[idx] - mean

			gradInput := grad[idx]
			g := float32(1.0)
			if b.affine {
				g = b.gamma[f]
				gradInput *= g
			}
			gradInput = gradInput/std - (sumGrad*g)/float32(numel)/std - diff*(sumGradXMean*g)/float32(numel)/std/std/std

			b.gradInBuf[idx] = gradInput
		}

		// Accumulate parameter gradients
		if b.affine {
			for s := 0; s < numel; s++ {
				idx := f*numel + s
				normalized := (savedInput[idx] - mean) / std
				b.gradGammaBuf[f] += grad[idx] * normalized
				b.gradBetaBuf[f] += grad[idx]
			}
		}
	}

	// Pop offsets
	b.savedInputOffsets = b.savedInputOffsets[:ts]
	b.savedMeanOffsets = b.savedMeanOffsets[:ts]
	b.savedStdOffsets = b.savedStdOffsets[:ts]

	return b.gradInBuf[:len(grad)]
}

func (b *BatchNorm2D) Params() []float32 {
	return b.params
}

func (b *BatchNorm2D) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &b.params[0] != &params[0] {
		if len(params) == len(b.params) {
			b.params = params
			b.updateViews()
		} else {
			copy(b.params, params)
		}
	}
}

func (b *BatchNorm2D) updateViews() {
	if b.affine {
		b.gamma = b.params[:b.numFeatures]
		b.beta = b.params[b.numFeatures:]
	}
}

func (b *BatchNorm2D) Gradients() []float32 {
	return b.grads
}

func (b *BatchNorm2D) SetGradients(gradients []float32) {
	if len(gradients) == 0 {
		return
	}
	if &b.grads[0] != &gradients[0] {
		if len(gradients) == len(b.grads) {
			b.grads = gradients
			b.updateGradViews()
		} else {
			copy(b.grads, gradients)
		}
	}
}

func (b *BatchNorm2D) updateGradViews() {
	if b.affine {
		b.gradGammaBuf = b.grads[:b.numFeatures]
		b.gradBetaBuf = b.grads[b.numFeatures:]
	}
}

// InSize returns the input size.
func (b *BatchNorm2D) InSize() int {
	return b.numFeatures
}

// OutSize returns the output size.
func (b *BatchNorm2D) OutSize() int {
	if b.numelPerChannel > 0 {
		return b.numFeatures * b.numelPerChannel
	}
	return b.numFeatures
}

func (b *BatchNorm2D) NamedParams() []NamedParam {
	params := []NamedParam{
		{
			Name:  "running_mean",
			Shape: []int{b.numFeatures},
			Data:  b.runningMean,
		},
		{
			Name:  "running_var",
			Shape: []int{b.numFeatures},
			Data:  b.runningVar,
		},
	}

	if b.affine {
		params = append(params,
			NamedParam{
				Name:  "gamma",
				Shape: []int{b.numFeatures},
				Data:  b.gamma,
			},
			NamedParam{
				Name:  "beta",
				Shape: []int{b.numFeatures},
				Data:  b.beta,
			},
		)
	}

	return params
}

// Reset resets the batch normalization layer.
func (b *BatchNorm2D) Reset() {
	b.savedInputOffsets = b.savedInputOffsets[:0]
	b.savedMeanOffsets = b.savedMeanOffsets[:0]
	b.savedStdOffsets = b.savedStdOffsets[:0]
}

// ClearGradients zeroes out the accumulated gradients.
func (b *BatchNorm2D) ClearGradients() {
	for i := range b.grads {
		b.grads[i] = 0
	}
}

// Clone creates a deep copy of the batch normalization layer.
func (b *BatchNorm2D) Clone() Layer {
	newB := NewBatchNorm2D(b.numFeatures, b.eps, b.momentum, b.affine)
	newB.training = b.training
	if b.affine {
		copy(newB.params, b.params)
	}
	copy(newB.runningMean, b.runningMean)
	copy(newB.runningVar, b.runningVar)
	newB.device = b.device
	return newB
}

func (b *BatchNorm2D) LightweightClone(params []float32, grads []float32) Layer {
	var gamma, beta, gradGamma, gradBeta []float32
	if b.affine {
		gamma = params[:b.numFeatures]
		beta = params[b.numFeatures:]
		gradGamma = grads[:b.numFeatures]
		gradBeta = grads[b.numFeatures:]
	}

	newB := &BatchNorm2D{
		numFeatures:     b.numFeatures,
		eps:             b.eps,
		momentum:        b.momentum,
		affine:          b.affine,
		training:        b.training,
		params:          params,
		gamma:           gamma,
		beta:            beta,
		runningMean:     b.runningMean, // shared statistics
		runningVar:      b.runningVar,  // shared statistics
		grads:           grads,
		gradGammaBuf:    gradGamma,
		gradBetaBuf:     gradBeta,
		outputBuf:       make([]float32, 0),
		gradInBuf:       make([]float32, 0),
		savedMean:         make([]float32, b.numFeatures),
		savedVar:          make([]float32, b.numFeatures),
		savedStd:          make([]float32, b.numFeatures),
		savedInputOffsets: make([]int, 0, 128),
		savedMeanOffsets:  make([]int, 0, 128),
		savedStdOffsets:   make([]int, 0, 128),
		device:            b.device,
		numelPerChannel:   b.numelPerChannel,
	}
	return newB
}

// GetGamma returns the gamma parameters (for affine layers).
func (b *BatchNorm2D) GetGamma() []float32 {
	if !b.affine {
		return nil
	}
	return b.gamma
}

// GetBeta returns the beta parameters (for affine layers).
func (b *BatchNorm2D) GetBeta() []float32 {
	if !b.affine {
		return nil
	}
	return b.beta
}

// GetRunningMean returns the running mean statistics.
func (b *BatchNorm2D) GetRunningMean() []float32 {
	return b.runningMean
}

// GetRunningVar returns the running variance statistics.
func (b *BatchNorm2D) GetRunningVar() []float32 {
	return b.runningVar
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For BatchNorm2D, gradients are already accumulated in Backward, so this just calls Backward.
func (b *BatchNorm2D) AccumulateBackward(grad []float32) []float32 {
	return b.Backward(grad)
}

// GetEps returns the epsilon value for numerical stability.
func (b *BatchNorm2D) GetEps() float32 {
	return b.eps
}

// GetMomentum returns the momentum value.
func (b *BatchNorm2D) GetMomentum() float32 {
	return b.momentum
}

// SetTraining sets whether the layer should be in training or inference mode.
func (b *BatchNorm2D) SetTraining(training bool) {
	b.training = training
}

// IsTraining returns whether the layer is in training mode.
func (b *BatchNorm2D) IsTraining() bool {
	return b.training
}
