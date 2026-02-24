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

	// Metal buffers
	bufIn          *MetalBuffer
	bufOut         *MetalBuffer
	bufRunningMean *MetalBuffer
	bufRunningVar  *MetalBuffer
	bufSavedMean   *MetalBuffer
	bufSavedStd    *MetalBuffer
	bufGamma       *MetalBuffer
	bufBeta        *MetalBuffer
	bufGradIn      *MetalBuffer
	bufGradGamma   *MetalBuffer
	bufGradBeta    *MetalBuffer
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
	if numFeatures <= 0 {
		return
	}
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

	// Metal initialization if device is GPU
	if md, ok := b.device.(*MetalDevice); ok && md.IsAvailable() {
		b.bufRunningMean = md.CreateBuffer(b.runningMean)
		b.bufRunningVar = md.CreateBuffer(b.runningVar)
		if b.affine {
			b.bufGamma = md.CreateBuffer(b.gamma)
			b.bufBeta = md.CreateBuffer(b.beta)
			b.bufGradGamma = md.CreateEmptyBuffer(numFeatures)
			b.bufGradBeta = md.CreateEmptyBuffer(numFeatures)
		}
		b.bufSavedMean = md.CreateEmptyBuffer(numFeatures)
		b.bufSavedStd = md.CreateEmptyBuffer(numFeatures)
		b.bufGradIn = md.CreateEmptyBuffer(numFeatures) // Placeholder, will be resized
	}
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
	if md, ok := device.(*MetalDevice); ok && md.IsAvailable() && b.numFeatures > 0 {
		if b.bufRunningMean == nil {
			b.bufRunningMean = md.CreateBuffer(b.runningMean)
		}
		if b.bufRunningVar == nil {
			b.bufRunningVar = md.CreateBuffer(b.runningVar)
		}
		if b.affine {
			if b.bufGamma == nil {
				b.bufGamma = md.CreateBuffer(b.gamma)
			}
			if b.bufBeta == nil {
				b.bufBeta = md.CreateBuffer(b.beta)
			}
			if b.bufGradGamma == nil {
				b.bufGradGamma = md.CreateEmptyBuffer(b.numFeatures)
			}
			if b.bufGradBeta == nil {
				b.bufGradBeta = md.CreateEmptyBuffer(b.numFeatures)
			}
		}
		if b.bufSavedMean == nil {
			b.bufSavedMean = md.CreateEmptyBuffer(b.numFeatures)
		}
		if b.bufSavedStd == nil {
			b.bufSavedStd = md.CreateEmptyBuffer(b.numFeatures)
		}
		if b.bufGradIn == nil {
			b.bufGradIn = md.CreateEmptyBuffer(b.numFeatures)
		}
	}
}

func (b *BatchNorm2D) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	if len(x) == 0 {
		return make([]float32, 0)
	}

	total := len(x)
	batchSize := 1
	spatialSize := total / b.numFeatures
	if b.inputHeight > 0 && b.inputWidth > 0 {
		spatialSize = b.inputHeight * b.inputWidth
		batchSize = total / (b.numFeatures * spatialSize)
	}
	b.numelPerChannel = spatialSize

	if len(b.outputBuf) < total {
		b.outputBuf = make([]float32, total)
	}
	output := b.outputBuf[:total]

	if b.training {
		// Save state for backward pass
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

		// Compute mean and variance
		if md, ok := b.device.(*MetalDevice); ok && md.IsAvailable() {
			if b.bufIn == nil || b.bufIn.length < total {
				if b.bufIn != nil {
					b.bufIn.Free()
				}
				b.bufIn = md.CreateBuffer(x)
			} else {
				b.bufIn.Update(x)
			}
			md.BatchNorm2DStatsPersistent(b.bufIn, b.bufSavedMean, b.bufSavedStd, batchSize, b.numFeatures, spatialSize, b.eps)
			b.bufSavedMean.Read(b.savedMean)
			b.bufSavedStd.Read(b.savedVar) // varBuf in kernel, we store it in savedVar
			for f := 0; f < b.numFeatures; f++ {
				mean := b.savedMean[f]
				variance := b.savedVar[f]
				b.savedStd[f] = float32(math.Sqrt(float64(variance + b.eps)))
				b.runningMean[f] = (1-b.momentum)*b.runningMean[f] + b.momentum*mean
				b.runningVar[f] = (1-b.momentum)*b.runningVar[f] + b.momentum*variance
			}
		} else {
			for f := 0; f < b.numFeatures; f++ {
				sum := float32(0.0)
				for i := 0; i < batchSize; i++ {
					base := i * b.numFeatures * spatialSize + f * spatialSize
					for s := 0; s < spatialSize; s++ {
						sum += x[base+s]
					}
				}
				mean := sum / float32(batchSize*spatialSize)
				b.savedMean[f] = mean
				b.runningMean[f] = (1-b.momentum)*b.runningMean[f] + b.momentum*mean

				sumSq := float32(0.0)
				for i := 0; i < batchSize; i++ {
					base := i * b.numFeatures * spatialSize + f * spatialSize
					for s := 0; s < spatialSize; s++ {
						diff := x[base+s] - mean
						sumSq += diff * diff
					}
				}
				variance := sumSq / float32(batchSize*spatialSize)
				std := float32(math.Sqrt(float64(variance + b.eps)))
				b.savedVar[f] = variance
				b.savedStd[f] = std
				b.runningVar[f] = (1-b.momentum)*b.runningVar[f] + b.momentum*variance
			}
		}

		if md, ok := b.device.(*MetalDevice); ok && md.IsAvailable() {
			if b.bufOut == nil || b.bufOut.length < total {
				if b.bufOut != nil {
					b.bufOut.Free()
				}
				b.bufOut = md.CreateEmptyBuffer(total)
			}
			b.bufSavedMean.Update(b.savedMean)
			b.bufSavedStd.Update(b.savedStd)
			b.bufRunningMean.Update(b.runningMean)
			b.bufRunningVar.Update(b.runningVar)

			md.BatchNorm2DForwardPersistent(b.bufIn, b.bufOut, b.bufRunningMean, b.bufRunningVar, b.bufSavedMean, b.bufSavedStd, b.bufGamma, b.bufBeta, batchSize, b.numFeatures, spatialSize, b.eps, b.momentum, true, b.affine)
			b.bufOut.Read(output)
			return output
		}

		// CPU normalization
		for i := 0; i < batchSize; i++ {
			for f := 0; f < b.numFeatures; f++ {
				base := i * b.numFeatures * spatialSize + f * spatialSize
				mean := b.savedMean[f]
				std := b.savedStd[f]
				for s := 0; s < spatialSize; s++ {
					norm := (x[base+s] - mean) / std
					if b.affine {
						output[base+s] = b.gamma[f]*norm + b.beta[f]
					} else {
						output[base+s] = norm
					}
				}
			}
		}
	} else {
		// Inference mode
		if md, ok := b.device.(*MetalDevice); ok && md.IsAvailable() {
			if b.bufIn == nil || b.bufIn.length < total {
				if b.bufIn != nil {
					b.bufIn.Free()
				}
				b.bufIn = md.CreateBuffer(x)
			} else {
				b.bufIn.Update(x)
			}
			if b.bufOut == nil || b.bufOut.length < total {
				if b.bufOut != nil {
					b.bufOut.Free()
				}
				b.bufOut = md.CreateEmptyBuffer(total)
			}
			md.BatchNorm2DForwardPersistent(b.bufIn, b.bufOut, b.bufRunningMean, b.bufRunningVar, b.bufSavedMean, b.bufSavedStd, b.bufGamma, b.bufBeta, batchSize, b.numFeatures, spatialSize, b.eps, b.momentum, false, b.affine)
			b.bufOut.Read(output)
			return output
		}

		for i := 0; i < batchSize; i++ {
			for f := 0; f < b.numFeatures; f++ {
				base := i * b.numFeatures * spatialSize + f * spatialSize
				mean := b.runningMean[f]
				std := float32(math.Sqrt(float64(b.runningVar[f] + b.eps)))
				for s := 0; s < spatialSize; s++ {
					norm := (x[base+s] - mean) / std
					if b.affine {
						output[base+s] = b.gamma[f]*norm + b.beta[f]
					} else {
						output[base+s] = norm
					}
				}
			}
		}
	}
	return output
}

func (b *BatchNorm2D) Forward(x []float32) []float32 {
	return b.ForwardWithArena(x, nil, nil)
}

func (b *BatchNorm2D) ForwardBatch(x []float32, batchSize int) []float32 {
	return b.ForwardWithArena(x, nil, nil)
}

func (b *BatchNorm2D) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	return b.ForwardWithArena(x, arena, offset)
}

func (b *BatchNorm2D) Backward(grad []float32) []float32 {
	if !b.training {
		total := len(grad)
		if len(b.gradInBuf) < total {
			b.gradInBuf = make([]float32, total)
		}
		gradIn := b.gradInBuf[:total]
		batchSize := 1
		spatialSize := total / b.numFeatures
		if b.inputHeight > 0 && b.inputWidth > 0 {
			spatialSize = b.inputHeight * b.inputWidth
			batchSize = total / (b.numFeatures * spatialSize)
		}
		for i := 0; i < batchSize; i++ {
			for f := 0; f < b.numFeatures; f++ {
				base := i * b.numFeatures * spatialSize + f * spatialSize
				std := float32(math.Sqrt(float64(b.runningVar[f] + b.eps)))
				for s := 0; s < spatialSize; s++ {
					g := grad[base+s]
					if b.affine {
						g *= b.gamma[f]
					}
					gradIn[base+s] = g / std
				}
			}
		}
		return gradIn
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
	batchSize := 1
	spatialSize := total / b.numFeatures
	if b.inputHeight > 0 && b.inputWidth > 0 {
		spatialSize = b.inputHeight * b.inputWidth
		batchSize = total / (b.numFeatures * spatialSize)
	}

	var savedMean, savedStd []float32
	if meanOff == -1 {
		savedMean = b.savedMean
		savedStd = b.savedStd
	} else {
		savedMean = b.arena[meanOff : meanOff+b.numFeatures]
		savedStd = b.arena[stdOff : stdOff+b.numFeatures]
	}

	if len(b.gradInBuf) < total {
		b.gradInBuf = make([]float32, total)
	}
	gradIn := b.gradInBuf[:total]

	if md, ok := b.device.(*MetalDevice); ok && md.IsAvailable() {
		if b.bufIn == nil || b.bufIn.length < total {
			if b.bufIn != nil {
				b.bufIn.Free()
			}
			b.bufIn = md.CreateBuffer(savedInput)
		} else {
			b.bufIn.Update(savedInput)
		}
		if b.bufOut == nil || b.bufOut.length < total {
			if b.bufOut != nil {
				b.bufOut.Free()
			}
			b.bufOut = md.CreateBuffer(grad)
		} else {
			b.bufOut.Update(grad)
		}
		if b.bufGradIn == nil || b.bufGradIn.length < total {
			if b.bufGradIn != nil {
				b.bufGradIn.Free()
			}
			b.bufGradIn = md.CreateEmptyBuffer(total)
		}
		b.bufSavedMean.Update(savedMean)
		b.bufSavedStd.Update(savedStd)

		md.BatchNorm2DBackwardPersistent(b.bufOut, b.bufIn, b.bufGradIn, b.bufSavedMean, b.bufSavedStd, b.bufGamma, b.bufGradGamma, b.bufGradBeta, batchSize, b.numFeatures, spatialSize, b.eps, b.momentum, b.affine)
		b.bufGradIn.Read(gradIn)
		if b.affine {
			b.bufGradGamma.Read(b.gradGammaBuf)
			b.bufGradBeta.Read(b.gradBetaBuf)
		}
		b.savedInputOffsets = b.savedInputOffsets[:ts]
		b.savedMeanOffsets = b.savedMeanOffsets[:ts]
		b.savedStdOffsets = b.savedStdOffsets[:ts]
		return gradIn
	}

	// CPU backward
	m := float32(batchSize * spatialSize)
	for f := 0; f < b.numFeatures; f++ {
		mean := savedMean[f]
		std := savedStd[f]
		sumGrad := float32(0.0)
		sumGradXMean := float32(0.0)
		for i := 0; i < batchSize; i++ {
			base := i * b.numFeatures * spatialSize + f * spatialSize
			for s := 0; s < spatialSize; s++ {
				diff := savedInput[base+s] - mean
				sumGrad += grad[base+s]
				sumGradXMean += grad[base+s] * diff
			}
		}

		g := float32(1.0)
		if b.affine {
			g = b.gamma[f]
		}

		for i := 0; i < batchSize; i++ {
			base := i * b.numFeatures * spatialSize + f * spatialSize
			for s := 0; s < spatialSize; s++ {
				diff := savedInput[base+s] - mean
				norm := diff / std
				gi := grad[base+s] * g
				gradIn[base+s] = (gi/std - (sumGrad*g)/m/std - diff*(sumGradXMean*g)/m/(std*std*std))
				if b.affine {
					b.gradGammaBuf[f] += grad[base+s] * norm
					b.gradBetaBuf[f] += grad[base+s]
				}
			}
		}
	}
	b.savedInputOffsets = b.savedInputOffsets[:ts]
	b.savedMeanOffsets = b.savedMeanOffsets[:ts]
	b.savedStdOffsets = b.savedStdOffsets[:ts]
	return gradIn
}

func (b *BatchNorm2D) BackwardBatch(grad []float32, batchSize int) []float32 {
	return b.Backward(grad)
}

func (b *BatchNorm2D) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return b.Backward(grad)
}

func (b *BatchNorm2D) Params() []float32 { return b.params }

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
	b.syncGPU()
}

func (b *BatchNorm2D) syncGPU() {
	if md, ok := b.device.(*MetalDevice); ok && md.IsAvailable() && b.affine {
		if b.bufGamma == nil {
			b.bufGamma = md.CreateBuffer(b.gamma)
		} else {
			b.bufGamma.Update(b.gamma)
		}
		if b.bufBeta == nil {
			b.bufBeta = md.CreateBuffer(b.beta)
		} else {
			b.bufBeta.Update(b.beta)
		}
	}
}

func (b *BatchNorm2D) updateViews() {
	if b.affine {
		b.gamma = b.params[:b.numFeatures]
		b.beta = b.params[b.numFeatures:]
	}
}

func (b *BatchNorm2D) Gradients() []float32 { return b.grads }

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

func (b *BatchNorm2D) InSize() int { return b.numFeatures }
func (b *BatchNorm2D) OutSize() int {
	if b.numelPerChannel > 0 {
		return b.numFeatures * b.numelPerChannel
	}
	return b.numFeatures
}

func (b *BatchNorm2D) NamedParams() []NamedParam {
	params := []NamedParam{
		{Name: "running_mean", Shape: []int{b.numFeatures}, Data: b.runningMean},
		{Name: "running_var", Shape: []int{b.numFeatures}, Data: b.runningVar},
	}
	if b.affine {
		params = append(params,
			NamedParam{Name: "gamma", Shape: []int{b.numFeatures}, Data: b.gamma},
			NamedParam{Name: "beta", Shape: []int{b.numFeatures}, Data: b.beta},
		)
	}
	return params
}

func (b *BatchNorm2D) Reset() {
	b.savedInputOffsets = b.savedInputOffsets[:0]
	b.savedMeanOffsets = b.savedMeanOffsets[:0]
	b.savedStdOffsets = b.savedStdOffsets[:0]
}

func (b *BatchNorm2D) ClearGradients() {
	for i := range b.grads {
		b.grads[i] = 0
	}
}

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

func (b *BatchNorm2D) LightweightClone(params, grads []float32) Layer {
	var gamma, beta, gradGamma, gradBeta []float32
	if b.affine {
		gamma, beta = params[:b.numFeatures], params[b.numFeatures:]
		gradGamma, gradBeta = grads[:b.numFeatures], grads[b.numFeatures:]
	}
	newB := &BatchNorm2D{
		numFeatures: b.numFeatures, eps: b.eps, momentum: b.momentum, affine: b.affine,
		training: b.training, params: params, gamma: gamma, beta: beta,
		runningMean: b.runningMean, runningVar: b.runningVar,
		grads: grads, gradGammaBuf: gradGamma, gradBetaBuf: gradBeta,
		outputBuf: make([]float32, 0), gradInBuf: make([]float32, 0),
		savedMean: make([]float32, b.numFeatures), savedVar: make([]float32, b.numFeatures), savedStd: make([]float32, b.numFeatures),
		savedInputOffsets: make([]int, 0, 128), savedMeanOffsets: make([]int, 0, 128), savedStdOffsets: make([]int, 0, 128),
		device: b.device, numelPerChannel: b.numelPerChannel,
	}

	if md, ok := b.device.(*MetalDevice); ok && md.IsAvailable() && b.numFeatures > 0 {
		newB.bufRunningMean = md.CreateBuffer(newB.runningMean)
		newB.bufRunningVar = md.CreateBuffer(newB.runningVar)
		if b.affine {
			newB.bufGamma = md.CreateBuffer(newB.gamma)
			newB.bufBeta = md.CreateBuffer(newB.beta)
			newB.bufGradGamma = md.CreateEmptyBuffer(b.numFeatures)
			newB.bufGradBeta = md.CreateEmptyBuffer(b.numFeatures)
		}
		newB.bufSavedMean = md.CreateEmptyBuffer(b.numFeatures)
		newB.bufSavedStd = md.CreateEmptyBuffer(b.numFeatures)
		newB.bufGradIn = md.CreateEmptyBuffer(b.numFeatures)
	}

	return newB
}

func (b *BatchNorm2D) GetGamma() []float32          { if !b.affine { return nil }; return b.gamma }
func (b *BatchNorm2D) GetBeta() []float32           { if !b.affine { return nil }; return b.beta }
func (b *BatchNorm2D) GetRunningMean() []float32    { return b.runningMean }
func (b *BatchNorm2D) GetRunningVar() []float32     { return b.runningVar }
func (b *BatchNorm2D) AccumulateBackward(grad []float32) []float32 { return b.Backward(grad) }
func (b *BatchNorm2D) GetEps() float32              { return b.eps }
func (b *BatchNorm2D) GetMomentum() float32         { return b.momentum }
func (b *BatchNorm2D) SetTraining(training bool)    { b.training = training }
func (b *BatchNorm2D) IsTraining() bool             { return b.training }
