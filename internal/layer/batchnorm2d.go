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

	numelPerChannel int
	device          Device
}

// NewBatchNorm2D creates a new 2D batch normalization layer.
func NewBatchNorm2D(numFeatures int, eps float32, momentum float32, affine bool) *BatchNorm2D {
	params := make([]float32, 0)
	var gamma, beta []float32
	grads := make([]float32, 0)
	var gradGamma, gradBeta []float32

	if affine {
		params = make([]float32, numFeatures*2)
		gamma = params[:numFeatures]
		beta = params[numFeatures:]
		for i := 0; i < numFeatures; i++ {
			gamma[i] = 1.0
			beta[i] = 0.0
		}
		grads = make([]float32, numFeatures*2)
		gradGamma = grads[:numFeatures]
		gradBeta = grads[numFeatures:]
	}

	l := &BatchNorm2D{
		numFeatures:  numFeatures,
		eps:          eps,
		momentum:     momentum,
		affine:       affine,
		training:     true,
		params:       params,
		gamma:        gamma,
		beta:         beta,
		runningMean:  make([]float32, numFeatures),
		runningVar:   make([]float32, numFeatures),
		grads:        grads,
		gradGammaBuf: gradGamma,
		gradBetaBuf:  gradBeta,
		outputBuf:    make([]float32, 0),
		gradInBuf:    make([]float32, 0),
		savedMean:    make([]float32, numFeatures),
		savedVar:     make([]float32, numFeatures),
		savedStd:     make([]float32, numFeatures),
		device:       &CPUDevice{},
	}

	// Initialize running variance to 1
	for i := range l.runningVar {
		l.runningVar[i] = 1.0
	}

	return l
}

// SetDevice sets the computation device.
func (b *BatchNorm2D) SetDevice(device Device) {
	b.device = device
}

// Forward performs a forward pass through the batch normalization layer.
// x: input tensor of shape [batchSize * numFeatures * spatialSize]
// Returns: normalized output with same shape as input
func (b *BatchNorm2D) Forward(x []float32) []float32 {
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
	if cap(b.savedInput) < total {
		b.savedInput = make([]float32, total)
	}
	copy(b.savedInput, x)

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

	numel := b.numelPerChannel
	numFeatures := b.numFeatures

	// Ensure gradInBuf is sized correctly
	if len(b.gradInBuf) != len(grad) {
		b.gradInBuf = make([]float32, len(grad))
	}

	for f := 0; f < numFeatures; f++ {
		mean := b.savedMean[f]
		std := b.savedStd[f]

		// Compute sum of gradients and sum of gradients * (x - mean)
		sumGrad := float32(0.0)
		sumGradXMean := float32(0.0)
		for s := 0; s < numel; s++ {
			idx := f*numel + s
			diff := b.savedInput[idx] - mean
			sumGrad += grad[idx]
			sumGradXMean += grad[idx] * diff
		}

		// Compute gradient for each input
		for s := 0; s < numel; s++ {
			idx := f*numel + s
			diff := b.savedInput[idx] - mean

			gradInput := grad[idx]
			if b.affine {
				gradInput *= b.gamma[f]
			}
			gradInput = gradInput/std - sumGrad/float32(numel)/std - diff*sumGradXMean/float32(numel)/std/std/std

			b.gradInBuf[idx] = gradInput // This is gradient w.r.t. input, no need to accumulate across samples in the SAME pass
		}

		// Accumulate parameter gradients
		if b.affine {
			for s := 0; s < numel; s++ {
				idx := f*numel + s
				normalized := (b.savedInput[idx] - mean) / std
				b.gradGammaBuf[f] += grad[idx] * normalized
				b.gradBetaBuf[f] += grad[idx]
			}
		}
	}

	return b.gradInBuf
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
	// No state to reset
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
