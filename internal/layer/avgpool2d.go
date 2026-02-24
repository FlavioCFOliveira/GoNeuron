// Package layer provides neural network layer implementations.
package layer

import (
	"math"
)

// AvgPool2D implements 2D average pooling.
// Downsamples by taking the average over sliding windows.
type AvgPool2D struct {
	// Pooling parameters
	inChannels int
	kernelSize int
	stride     int
	padding    int

	// Input dimensions (inferred during forward)
	inputHeight int
	inputWidth  int

	// Output dimensions
	outputHeight int
	outputWidth  int

	// Pre-allocated buffers
	outputBuf []float32
	gradInBuf []float32
	countBuf  []int // Count of inputs contributing to each output position

	// Saved input for backward pass
	savedInput        []float32
	savedInputOffsets []int
	arena             []float32

	training bool

	device Device
}

// NewAvgPool2D creates a new 2D average pooling layer.
// inChannels: number of input channels
// kernelSize: size of pooling window (square)
// stride: stride for pooling (defaults to kernelSize)
// padding: zero padding size
func NewAvgPool2D(inChannels, kernelSize, stride, padding int) *AvgPool2D {
	return &AvgPool2D{
		inChannels:        inChannels,
		kernelSize:        kernelSize,
		stride:            stride,
		padding:           padding,
		outputBuf:         make([]float32, 0),
		gradInBuf:         make([]float32, 0),
		countBuf:          make([]int, 0),
		savedInput:        make([]float32, 0),
		savedInputOffsets: make([]int, 0, 16),
		device:            &CPUDevice{},
	}
}

// Build initializes the layer.
func (a *AvgPool2D) Build(inChannels int) {
	a.inChannels = inChannels
}

// SetDevice sets the computation device.
func (a *AvgPool2D) SetDevice(device Device) {
	a.device = device
}

// computeOutputSize calculates the output spatial dimensions
func (a *AvgPool2D) computeOutputSize(inputHeight, inputWidth int) (int, int) {
	// Output size: (input + 2*padding - kernel) / stride + 1
	outH := (inputHeight + 2*a.padding - a.kernelSize) / a.stride + 1
	outW := (inputWidth + 2*a.padding - a.kernelSize) / a.stride + 1
	return outH, outW
}

// SetTraining sets whether the layer is in training mode.
func (a *AvgPool2D) SetTraining(training bool) {
	a.training = training
}

func (a *AvgPool2D) ForwardWithArena(input []float32, arena *[]float32, offset *int) []float32 {
	totalInput := len(input)
	if totalInput == 0 {
		return make([]float32, 0)
	}

	if a.inChannels <= 0 {
		// Fallback for legacy code
		a.inChannels = 1
	}

	channelSize := totalInput / a.inChannels
	// Infer input dimensions
	a.inputHeight = int(math.Sqrt(float64(channelSize)))
	if a.inputHeight*a.inputHeight == channelSize {
		a.inputWidth = a.inputHeight
	} else {
		a.inputWidth = channelSize / a.inputHeight
	}

	// Compute output dimensions
	a.outputHeight, a.outputWidth = a.computeOutputSize(a.inputHeight, a.inputWidth)

	outSize := a.outputHeight * a.outputWidth
	requiredOutput := a.inChannels * outSize

	// Ensure buffers are sized correctly
	if len(a.outputBuf) < requiredOutput {
		a.outputBuf = make([]float32, requiredOutput)
	}
	if len(a.countBuf) < outSize {
		a.countBuf = make([]int, outSize)
	}
	if len(a.gradInBuf) < totalInput {
		a.gradInBuf = make([]float32, totalInput)
	}

	// Save input for backward pass
	if arena != nil && offset != nil {
		a.arena = *arena
		if len(*arena) < *offset+totalInput {
			newArena := make([]float32, (*offset+totalInput)*2)
			copy(newArena, *arena)
			*arena = newArena
			a.arena = *arena
		}
		saved := (*arena)[*offset : *offset+totalInput]
		copy(saved, input)
		a.savedInputOffsets = append(a.savedInputOffsets, *offset)
		*offset += totalInput
	} else {
		if cap(a.savedInput) < totalInput {
			a.savedInput = make([]float32, totalInput)
		}
		a.savedInput = a.savedInput[:totalInput]
		copy(a.savedInput, input)
		a.savedInputOffsets = append(a.savedInputOffsets[:0], 0)
		a.arena = a.savedInput
	}

	// GPU acceleration
	if m, ok := a.device.(*MetalDevice); ok && m.IsAvailable() {
		inBuf := m.CreateBuffer(input)
		outBuf := m.CreateBuffer(a.outputBuf[:requiredOutput])
		defer inBuf.Free()
		defer outBuf.Free()

		m.AvgPool2DPersistent(inBuf, outBuf, 1, a.inChannels, a.inputHeight, a.inputWidth, a.outputHeight, a.outputWidth, a.kernelSize, a.stride, a.padding)
		outBuf.Read(a.outputBuf[:requiredOutput])
		return a.outputBuf[:requiredOutput]
	}

	// Average pooling logic
	outH := a.outputHeight
	outW := a.outputWidth
	kernelSize := a.kernelSize
	stride := a.stride
	padding := a.padding
	inputHeight := a.inputHeight
	inputWidth := a.inputWidth

	// Reset countBuf for spatial positions (it's the same for all channels)
	for i := range a.countBuf {
		a.countBuf[i] = 0
	}

	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			pos := oh*outW + ow
			count := 0
			// Calculate contributing inputs for this output position
			for kh := 0; kh < kernelSize; kh++ {
				for kw := 0; kw < kernelSize; kw++ {
					inH := oh*stride + kh - padding
					inW := ow*stride + kw - padding
					if inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth {
						count++
					}
				}
			}
			a.countBuf[pos] = count

			if count > 0 {
				invCount := 1.0 / float32(count)
				for c := 0; c < a.inChannels; c++ {
					sum := float32(0.0)
					inChannelOffset := c * inputHeight * inputWidth
					outChannelOffset := c * outH * outW
					for kh := 0; kh < kernelSize; kh++ {
						for kw := 0; kw < kernelSize; kw++ {
							inH := oh*stride + kh - padding
							inW := ow*stride + kw - padding
							if inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth {
								idx := inChannelOffset + inH*inputWidth + inW
								sum += input[idx]
							}
						}
					}
					a.outputBuf[outChannelOffset+pos] = sum * invCount
				}
			} else {
				for c := 0; c < a.inChannels; c++ {
					outChannelOffset := c * outH * outW
					a.outputBuf[outChannelOffset+pos] = 0
				}
			}
		}
	}

	return a.outputBuf[:requiredOutput]
}

// Forward performs a forward pass through the average pooling layer.
func (a *AvgPool2D) Forward(input []float32) []float32 {
	return a.ForwardWithArena(input, nil, nil)
}

func (a *AvgPool2D) ForwardBatch(input []float32, batchSize int) []float32 {
	return a.ForwardBatchWithArena(input, batchSize, nil, nil)
}

func (a *AvgPool2D) ForwardBatchWithArena(input []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return a.ForwardWithArena(input, arena, offset)
	}

	inSize := len(input) / batchSize
	if a.inChannels <= 0 {
		a.inChannels = 1
	}

	channelSize := inSize / a.inChannels
	// Infer dimensions if needed
	if a.inputHeight == 0 || a.inputWidth == 0 {
		a.inputHeight = int(math.Sqrt(float64(channelSize)))
		if a.inputHeight*a.inputHeight == channelSize {
			a.inputWidth = a.inputHeight
		} else {
			a.inputWidth = channelSize / a.inputHeight
		}
		a.outputHeight, a.outputWidth = a.computeOutputSize(a.inputHeight, a.inputWidth)
	}

	outSize := a.outputHeight * a.outputWidth
	requiredOutput := batchSize * a.inChannels * outSize

	if len(a.outputBuf) < requiredOutput {
		a.outputBuf = make([]float32, requiredOutput)
	}

	// GPU acceleration
	if m, ok := a.device.(*MetalDevice); ok && m.IsAvailable() {
		// Save input to arena if provided
		if arena != nil && offset != nil {
			if len(*arena) < *offset+len(input) {
				newArena := make([]float32, (*offset+len(input))*2)
				copy(newArena, *arena)
				*arena = newArena
			}
			copy((*arena)[*offset:*offset+len(input)], input)
			a.arena = *arena
			for i := 0; i < batchSize; i++ {
				a.savedInputOffsets = append(a.savedInputOffsets, *offset+i*inSize)
			}
			*offset += len(input)
		} else {
			if cap(a.savedInput) < len(input) {
				a.savedInput = make([]float32, len(input))
			}
			a.savedInput = a.savedInput[:len(input)]
			copy(a.savedInput, input)
			a.arena = a.savedInput
			a.savedInputOffsets = a.savedInputOffsets[:0]
			for i := 0; i < batchSize; i++ {
				a.savedInputOffsets = append(a.savedInputOffsets, i*inSize)
			}
		}

		inBuf := m.CreateBuffer(input)
		outBuf := m.CreateBuffer(a.outputBuf[:requiredOutput])
		defer inBuf.Free()
		defer outBuf.Free()

		m.AvgPool2DPersistent(inBuf, outBuf, batchSize, a.inChannels, a.inputHeight, a.inputWidth, a.outputHeight, a.outputWidth, a.kernelSize, a.stride, a.padding)
		outBuf.Read(a.outputBuf[:requiredOutput])
		return a.outputBuf[:requiredOutput]
	}

	for i := 0; i < batchSize; i++ {
		out := a.ForwardWithArena(input[i*inSize:(i+1)*inSize], arena, offset)
		copy(a.outputBuf[i*a.inChannels*outSize:(i+1)*a.inChannels*outSize], out)
	}

	return a.outputBuf[:requiredOutput]
}

// Backward performs backpropagation through the average pooling layer.
func (a *AvgPool2D) Backward(grad []float32) []float32 {
	numSaved := len(a.savedInputOffsets)
	if numSaved == 0 {
		return nil
	}

	ts := numSaved - 1
	inSize := a.inChannels * a.inputHeight * a.inputWidth
	outSize := a.inChannels * a.outputHeight * a.outputWidth

	if len(a.gradInBuf) < inSize {
		a.gradInBuf = make([]float32, inSize)
	}
	gradIn := a.gradInBuf[:inSize]

	// GPU acceleration
	if m, ok := a.device.(*MetalDevice); ok && m.IsAvailable() {
		gOutBuf := m.CreateBuffer(grad[:outSize])
		gInBuf := m.CreateBuffer(gradIn)
		defer gOutBuf.Free()
		defer gInBuf.Free()

		m.AvgPool2DBackwardPersistent(gOutBuf, gInBuf, 1, a.inChannels, a.inputHeight, a.inputWidth, a.outputHeight, a.outputWidth, a.kernelSize, a.stride, a.padding)
		gInBuf.Read(gradIn)

		a.savedInputOffsets = a.savedInputOffsets[:ts]
		return gradIn
	}

	// Clear gradient buffer
	for i := range gradIn {
		gradIn[i] = 0
	}

	outH := a.outputHeight
	outW := a.outputWidth
	kernelSize := a.kernelSize
	stride := a.stride
	padding := a.padding
	inputHeight := a.inputHeight
	inputWidth := a.inputWidth

	// For each output position, distribute gradient to input positions
	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			pos := oh*outW + ow
			count := a.countBuf[pos]

			if count > 0 {
				gradPerInput := 1.0 / float32(count)
				for c := 0; c < a.inChannels; c++ {
					outChannelOffset := c * outH * outW
					inChannelOffset := c * inputHeight * inputWidth
					gradVal := grad[outChannelOffset+pos] * gradPerInput

					for kh := 0; kh < kernelSize; kh++ {
						for kw := 0; kw < kernelSize; kw++ {
							inH := oh*stride + kh - padding
							inW := ow*stride + kw - padding

							if inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth {
								inputIdx := inChannelOffset + inH*inputWidth + inW
								gradIn[inputIdx] += gradVal
							}
						}
					}
				}
			}
		}
	}

	a.savedInputOffsets = a.savedInputOffsets[:ts]
	return gradIn
}

func (a *AvgPool2D) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return a.Backward(grad)
	}

	inSize := a.inChannels * a.inputHeight * a.inputWidth
	outSize := a.inChannels * a.outputHeight * a.outputWidth
	requiredInput := batchSize * inSize

	if len(a.gradInBuf) < requiredInput {
		a.gradInBuf = make([]float32, requiredInput)
	}

	// GPU acceleration
	if m, ok := a.device.(*MetalDevice); ok && m.IsAvailable() {
		gradIn := a.gradInBuf[:requiredInput]
		gOutBuf := m.CreateBuffer(grad)
		gInBuf := m.CreateBuffer(gradIn)
		defer gOutBuf.Free()
		defer gInBuf.Free()

		m.AvgPool2DBackwardPersistent(gOutBuf, gInBuf, batchSize, a.inChannels, a.inputHeight, a.inputWidth, a.outputHeight, a.outputWidth, a.kernelSize, a.stride, a.padding)
		gInBuf.Read(gradIn)

		if len(a.savedInputOffsets) >= batchSize {
			a.savedInputOffsets = a.savedInputOffsets[:len(a.savedInputOffsets)-batchSize]
		}
		return gradIn
	}

	for i := batchSize - 1; i >= 0; i-- {
		dx := a.Backward(grad[i*outSize : (i+1)*outSize])
		copy(a.gradInBuf[i*inSize:(i+1)*inSize], dx)
	}

	return a.gradInBuf[:requiredInput]
}

func (a *AvgPool2D) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return a.BackwardBatch(grad, batchSize)
}

// Params returns layer parameters (empty for AvgPool2D).
func (a *AvgPool2D) Params() []float32 {
	return make([]float32, 0)
}

// SetParams sets layer parameters (no-op for AvgPool2D).
func (a *AvgPool2D) SetParams(params []float32) {
}

// Gradients returns layer gradients (empty for AvgPool2D).
func (a *AvgPool2D) Gradients() []float32 {
	return make([]float32, 0)
}

// SetGradients sets layer gradients (no-op for AvgPool2D).
func (a *AvgPool2D) SetGradients(gradients []float32) {
}

// InSize returns the input size (flattened).
func (a *AvgPool2D) InSize() int {
	return a.inChannels * a.inputHeight * a.inputWidth
}

// OutSize returns the output size (flattened).
func (a *AvgPool2D) OutSize() int {
	return a.inChannels * a.outputHeight * a.outputWidth
}

// Reset resets the average pooling layer.
func (a *AvgPool2D) Reset() {
	a.inputHeight = 0
	a.inputWidth = 0
	a.outputHeight = 0
	a.outputWidth = 0
}

// ClearGradients zeroes out the accumulated gradients (no-op for AvgPool2D).
func (a *AvgPool2D) ClearGradients() {
}

// Clone creates a deep copy of the average pooling layer.
func (a *AvgPool2D) Clone() Layer {
	newA := NewAvgPool2D(a.inChannels, a.kernelSize, a.stride, a.padding)
	newA.inputHeight = a.inputHeight
	newA.inputWidth = a.inputWidth
	newA.outputHeight = a.outputHeight
	newA.outputWidth = a.outputWidth
	newA.device = a.device
	return newA
}

func (a *AvgPool2D) LightweightClone(params []float32, grads []float32) Layer {
	newA := &AvgPool2D{
		inChannels:        a.inChannels,
		kernelSize:        a.kernelSize,
		stride:            a.stride,
		padding:           a.padding,
		inputHeight:       a.inputHeight,
		inputWidth:        a.inputWidth,
		outputHeight:      a.outputHeight,
		outputWidth:       a.outputWidth,
		outputBuf:         make([]float32, 0),
		gradInBuf:         make([]float32, 0),
		countBuf:          make([]int, 0),
		savedInput:        make([]float32, 0),
		savedInputOffsets: make([]int, 0, 128),
		training:          a.training,
		device:            a.device,
	}
	return newA
}

// GetKernelSize returns the kernel size.
func (a *AvgPool2D) GetKernelSize() int {
	return a.kernelSize
}

// GetStride returns the stride.
func (a *AvgPool2D) GetStride() int {
	return a.stride
}

// GetPadding returns the padding.
func (a *AvgPool2D) GetPadding() int {
	return a.padding
}

// GetInChannels returns the number of input channels.
func (a *AvgPool2D) GetInChannels() int {
	return a.inChannels
}

// AccumulateBackward performs backpropagation and accumulates gradients.
func (a *AvgPool2D) AccumulateBackward(grad []float32) []float32 {
	return a.Backward(grad)
}
