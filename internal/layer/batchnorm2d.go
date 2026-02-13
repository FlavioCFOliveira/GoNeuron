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
	eps         float64
	momentum    float64
	affine      bool

	// Learnable parameters (when affine is enabled)
	gamma []float64
	beta  []float64

	// Running statistics (for inference)
	runningMean []float64
	runningVar  []float64

	// Pre-allocated buffers
	outputBuf         []float64
	gradInBuf         []float64
	gradGammaBuf      []float64
	gradBetaBuf       []float64
	savedInput        []float64
	savedMean         []float64
	savedVar          []float64
	savedStd          []float64
	numelPerChannel   int
}

// NewBatchNorm2D creates a new 2D batch normalization layer.
// numFeatures: number of feature channels
// eps: small value for numerical stability (default 1e-5)
// momentum: for running average of mean/var (default 0.1)
// affine: whether to learn scale (gamma) and shift (beta) parameters
func NewBatchNorm2D(numFeatures int, eps float64, momentum float64, affine bool) *BatchNorm2D {
	l := &BatchNorm2D{
		numFeatures: numFeatures,
		eps:         eps,
		momentum:    momentum,
		affine:      affine,
		outputBuf:   make([]float64, 0),
		gradInBuf:   make([]float64, 0),
		runningMean: make([]float64, numFeatures),
		runningVar:  make([]float64, numFeatures),
		savedMean:   make([]float64, numFeatures),
		savedVar:    make([]float64, numFeatures),
		savedStd:    make([]float64, numFeatures),
	}

	if affine {
		l.gamma = make([]float64, numFeatures)
		l.beta = make([]float64, numFeatures)
		l.gradGammaBuf = make([]float64, numFeatures)
		l.gradBetaBuf = make([]float64, numFeatures)

		// Initialize gamma to 1 and beta to 0
		for i := 0; i < numFeatures; i++ {
			l.gamma[i] = 1.0
			l.beta[i] = 0.0
		}
	}

	return l
}

// Forward performs a forward pass through the batch normalization layer.
// x: input tensor of shape [batchSize * numFeatures * spatialSize]
// Returns: normalized output with same shape as input
func (b *BatchNorm2D) Forward(x []float64) []float64 {
	if len(x) == 0 {
		return make([]float64, 0)
	}

	// Infer dimensions
	total := len(x)
	spatialSize := total / (b.numFeatures)
	b.numelPerChannel = spatialSize

	// Ensure output buffer is sized correctly
	if len(b.outputBuf) != total {
		b.outputBuf = make([]float64, total)
	}
	if len(b.gradInBuf) != total {
		b.gradInBuf = make([]float64, total)
	}

	output := b.outputBuf

	// Save input for backward pass
	if cap(b.savedInput) < total {
		b.savedInput = make([]float64, total)
	}
	copy(b.savedInput, x)

	numel := b.numelPerChannel
	numFeatures := b.numFeatures

	for f := 0; f < numFeatures; f++ {
		// Compute batch mean
		sum := 0.0
		for s := 0; s < numel; s++ {
			idx := s*numFeatures + f
			sum += x[idx]
		}
		mean := sum / float64(numel)
		b.savedMean[f] = mean
		b.runningMean[f] = (1-b.momentum)*b.runningMean[f] + b.momentum*mean

		// Compute batch variance
		sumSquares := 0.0
		for s := 0; s < numel; s++ {
			idx := s*numFeatures + f
			diff := x[idx] - mean
			sumSquares += diff * diff
		}
		variance := sumSquares / float64(numel)
		std := math.Sqrt(variance + b.eps)
		b.savedVar[f] = variance
		b.savedStd[f] = std
		b.runningVar[f] = (1-b.momentum)*b.runningVar[f] + b.momentum*variance

		// Normalize and apply affine transformation
		for s := 0; s < numel; s++ {
			idx := s*numFeatures + f
			normalized := (x[idx] - mean) / std
			if b.affine {
				output[idx] = b.gamma[f]*normalized + b.beta[f]
			} else {
				output[idx] = normalized
			}
		}
	}

	return output
}

// Backward performs backpropagation through the batch normalization layer.
func (b *BatchNorm2D) Backward(grad []float64) []float64 {
	numel := b.numelPerChannel
	numFeatures := b.numFeatures

	// Clear input gradient buffer
	for i := range b.gradInBuf {
		b.gradInBuf[i] = 0
	}

	// Clear parameter gradients (if affine)
	if b.affine {
		for i := range b.gradGammaBuf {
			b.gradGammaBuf[i] = 0
		}
		for i := range b.gradBetaBuf {
			b.gradBetaBuf[i] = 0
		}
	}

	for f := 0; f < numFeatures; f++ {
		mean := b.savedMean[f]
		std := b.savedStd[f]

		// Compute sum of gradients and sum of gradients * (x - mean)
		sumGrad := 0.0
		sumGradXMean := 0.0
		for s := 0; s < numel; s++ {
			idx := s*numFeatures + f
			diff := b.savedInput[idx] - mean
			sumGrad += grad[idx]
			sumGradXMean += grad[idx] * diff
		}

		// Compute gradient for each input
		for s := 0; s < numel; s++ {
			idx := s*numFeatures + f
			diff := b.savedInput[idx] - mean

			gradInput := grad[idx]
			if b.affine {
				gradInput *= b.gamma[f]
			}
			gradInput = gradInput/std - sumGrad/float64(numel)/std - diff*sumGradXMean/float64(numel)/std/std/std

			b.gradInBuf[idx] += gradInput
		}

		// Accumulate parameter gradients
		if b.affine {
			for s := 0; s < numel; s++ {
				idx := s*numFeatures + f
				normalized := (b.savedInput[idx] - mean) / std
				b.gradGammaBuf[f] += grad[idx] * normalized
				b.gradBetaBuf[f] += grad[idx]
			}
		}
	}

	return b.gradInBuf
}

// Params returns layer parameters (gamma and beta when affine is enabled).
func (b *BatchNorm2D) Params() []float64 {
	if !b.affine {
		return make([]float64, 0)
	}

	total := len(b.gamma) + len(b.beta)
	params := make([]float64, total)
	copy(params, b.gamma)
	copy(params[len(b.gamma):], b.beta)
	return params
}

// SetParams updates gamma and beta from a flattened slice.
func (b *BatchNorm2D) SetParams(params []float64) {
	if !b.affine {
		return
	}

	totalGamma := len(b.gamma)
	copy(b.gamma, params[:totalGamma])
	copy(b.beta, params[totalGamma:])
}

// Gradients returns layer gradients (gamma and beta gradients when affine is enabled).
func (b *BatchNorm2D) Gradients() []float64 {
	if !b.affine {
		return make([]float64, 0)
	}

	total := len(b.gradGammaBuf) + len(b.gradBetaBuf)
	gradients := make([]float64, total)
	copy(gradients, b.gradGammaBuf)
	copy(gradients[len(b.gradGammaBuf):], b.gradBetaBuf)
	return gradients
}

// SetGradients sets gradients from a flattened slice.
func (b *BatchNorm2D) SetGradients(gradients []float64) {
	if !b.affine {
		return
	}

	totalGamma := len(b.gradGammaBuf)
	copy(b.gradGammaBuf, gradients[:totalGamma])
	copy(b.gradBetaBuf, gradients[totalGamma:])
}

// InSize returns the input size.
func (b *BatchNorm2D) InSize() int {
	return b.numFeatures
}

// OutSize returns the output size.
func (b *BatchNorm2D) OutSize() int {
	return b.numFeatures
}

// Reset resets the batch normalization layer.
func (b *BatchNorm2D) Reset() {
	// No state to reset
}

// GetGamma returns the gamma parameters (for affine layers).
func (b *BatchNorm2D) GetGamma() []float64 {
	if !b.affine {
		return nil
	}
	return b.gamma
}

// GetBeta returns the beta parameters (for affine layers).
func (b *BatchNorm2D) GetBeta() []float64 {
	if !b.affine {
		return nil
	}
	return b.beta
}

// GetRunningMean returns the running mean statistics.
func (b *BatchNorm2D) GetRunningMean() []float64 {
	return b.runningMean
}

// GetRunningVar returns the running variance statistics.
func (b *BatchNorm2D) GetRunningVar() []float64 {
	return b.runningVar
}

// GetEps returns the epsilon value for numerical stability.
func (b *BatchNorm2D) GetEps() float64 {
	return b.eps
}

// GetMomentum returns the momentum value.
func (b *BatchNorm2D) GetMomentum() float64 {
	return b.momentum
}
