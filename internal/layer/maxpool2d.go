// Package layer provides neural network layer implementations.
package layer

import (
	"math"
)

// MaxPool2D implements 2D max pooling.
// Downsamples by taking the maximum over sliding windows.
// Stores argmax indices for correct gradient flow during backward pass.
type MaxPool2D struct {
	// Pooling parameters
	kernelSize int
	stride     int
	padding    int

	// Input/output dimensions
	inChannels  int
	inputHeight int
	inputWidth  int

	// Output dimensions
	outputHeight int
	outputWidth  int

	// Pre-allocated buffers
	outputBuf []float32
	gradInBuf []float32
	argmaxBuf []int32 // Stores index of max value for each output position

	// Saved input for backward pass
	savedInput []float32

	training bool

	device Device
}

// NewMaxPool2D creates a new 2D max pooling layer.
// inChannels: number of input channels
// kernelSize: size of pooling window (square)
// stride: stride for pooling (defaults to kernelSize)
// padding: zero padding size
func NewMaxPool2D(inChannels, kernelSize, stride, padding int) *MaxPool2D {
	m := &MaxPool2D{
		inChannels: inChannels,
		kernelSize: kernelSize,
		stride:     stride,
		padding:    padding,
		outputBuf:  make([]float32, 0),
		gradInBuf:  make([]float32, 0),
		argmaxBuf:  make([]int32, 0),
		savedInput: make([]float32, 0),
		device:     &CPUDevice{},
	}

	if inChannels != -1 {
		m.Build(inChannels)
	}

	return m
}

// Build initializes the layer with the given input size (channels).
func (m *MaxPool2D) Build(inChannels int) {
	m.inChannels = inChannels
}

// SetInputDimensions explicitly sets the input dimensions.
func (m *MaxPool2D) SetInputDimensions(height, width int) {
	m.inputHeight = height
	m.inputWidth = width
	m.outputHeight, m.outputWidth = m.computeOutputSize(height, width)
}

// SetDevice sets the computation device.
func (m *MaxPool2D) SetDevice(device Device) {
	m.device = device
}

// computeOutputSize calculates the output spatial dimensions
func (m *MaxPool2D) computeOutputSize(inputHeight, inputWidth int) (int, int) {
	// Output size: (input + 2*padding - kernel) / stride + 1
	outH := (inputHeight + 2*m.padding - m.kernelSize) / m.stride + 1
	outW := (inputWidth + 2*m.padding - m.kernelSize) / m.stride + 1
	return outH, outW
}

// SetTraining sets whether the layer is in training mode.
func (m *MaxPool2D) SetTraining(training bool) {
	m.training = training
}

func (m *MaxPool2D) ForwardWithArena(input []float32, arena *[]float32, offset *int) []float32 {
	totalInput := len(input)
	if totalInput == 0 {
		return make([]float32, 0)
	}

	if m.inChannels <= 0 {
		m.inChannels = 1
	}

	channelSize := totalInput / m.inChannels

	// Infer input dimensions
	if m.inputHeight == 0 || m.inputWidth == 0 {
		m.inputHeight = int(math.Sqrt(float64(channelSize)))
		if m.inputHeight*m.inputHeight == channelSize {
			m.inputWidth = m.inputHeight
		} else {
			m.inputWidth = channelSize / m.inputHeight
		}
	}

	// Compute output dimensions
	m.outputHeight, m.outputWidth = m.computeOutputSize(m.inputHeight, m.inputWidth)

	outChannels := m.inChannels
	outSize := m.outputHeight * m.outputWidth

	// Ensure buffers are sized correctly
	requiredOutput := outChannels * outSize
	if len(m.outputBuf) < requiredOutput {
		m.outputBuf = make([]float32, requiredOutput)
	}
	if cap(m.argmaxBuf) < requiredOutput {
		m.argmaxBuf = make([]int32, requiredOutput)
	}
	m.argmaxBuf = m.argmaxBuf[:requiredOutput]

	if cap(m.gradInBuf) < totalInput {
		m.gradInBuf = make([]float32, totalInput)
	}

	// Save input for backward pass
	if arena != nil && offset != nil {
		if len(*arena) < *offset+totalInput {
			newArena := make([]float32, (*offset+totalInput)*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		saved := (*arena)[*offset : *offset+totalInput]
		copy(saved, input)
		m.savedInput = saved
		*offset += totalInput
	} else {
		if cap(m.savedInput) < totalInput {
			m.savedInput = make([]float32, totalInput)
		}
		m.savedInput = m.savedInput[:totalInput]
		copy(m.savedInput, input)
	}

	// GPU acceleration
	if met, ok := m.device.(*MetalDevice); ok && met.IsAvailable() {
		inBuf := met.CreateBuffer(input)
		outBuf := met.CreateBuffer(m.outputBuf[:requiredOutput])
		argBuf := met.CreateIntBuffer(m.argmaxBuf[:requiredOutput])
		defer inBuf.Free()
		defer outBuf.Free()
		defer argBuf.Free()

		met.MaxPool2DPersistent(inBuf, outBuf, argBuf, 1, m.inChannels, m.inputHeight, m.inputWidth, m.outputHeight, m.outputWidth, m.kernelSize, m.stride, m.padding)
		outBuf.Read(m.outputBuf[:requiredOutput])
		argBuf.ReadInt(m.argmaxBuf[:requiredOutput])
		return m.outputBuf[:requiredOutput]
	}

	// Max pooling logic
	outH := m.outputHeight
	outW := m.outputWidth
	kernelSize := m.kernelSize
	stride := m.stride
	padding := m.padding
	inputHeight := m.inputHeight
	inputWidth := m.inputWidth

	channelStride := inputHeight * inputWidth
	outputChannelStride := outH * outW

	for c := 0; c < outChannels; c++ {
		channelOffset := c * channelStride
		outputOffset := c * outputChannelStride

		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				maxVal := float32(math.Inf(-1))
				maxIdx := -1

				for kh := 0; kh < kernelSize; kh++ {
					for kw := 0; kw < kernelSize; kw++ {
						inH := oh*stride + kh - padding
						inW := ow*stride + kw - padding

						if inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth {
							idx := channelOffset + inH*inputWidth + inW
							if input[idx] > maxVal {
								maxVal = input[idx]
								maxIdx = idx
							}
						}
					}
				}

				pos := outputOffset + oh*outW + ow
				m.outputBuf[pos] = maxVal
				m.argmaxBuf[pos] = int32(maxIdx)
			}
		}
	}

	return m.outputBuf[:requiredOutput]
}

// Forward performs a forward pass through the max pooling layer.
func (m *MaxPool2D) Forward(input []float32) []float32 {
	return m.ForwardWithArena(input, nil, nil)
}

func (m *MaxPool2D) ForwardBatch(input []float32, batchSize int) []float32 {
	return m.ForwardBatchWithArena(input, batchSize, nil, nil)
}

func (m *MaxPool2D) ForwardBatchWithArena(input []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return m.ForwardWithArena(input, arena, offset)
	}

	inSize := len(input) / batchSize
	if m.inChannels <= 0 {
		m.inChannels = 1
	}

	channelSize := inSize / m.inChannels
	// Infer dimensions if needed
	if m.inputHeight == 0 || m.inputWidth == 0 {
		m.inputHeight = int(math.Sqrt(float64(channelSize)))
		if m.inputHeight*m.inputHeight == channelSize {
			m.inputWidth = m.inputHeight
		} else {
			m.inputWidth = channelSize / m.inputHeight
		}
		m.outputHeight, m.outputWidth = m.computeOutputSize(m.inputHeight, m.inputWidth)
	}

	outSize := m.outputHeight * m.outputWidth
	requiredOutput := batchSize * m.inChannels * outSize

	if len(m.outputBuf) < requiredOutput {
		m.outputBuf = make([]float32, requiredOutput)
	}
	if cap(m.argmaxBuf) < requiredOutput {
		m.argmaxBuf = make([]int32, requiredOutput)
	}
	m.argmaxBuf = m.argmaxBuf[:requiredOutput]

	// Save input to arena if provided
	if arena != nil && offset != nil {
		if len(*arena) < *offset+len(input) {
			newArena := make([]float32, (*offset+len(input))*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		saved := (*arena)[*offset : *offset+len(input)]
		copy(saved, input)
		m.savedInput = saved
		*offset += len(input)
	} else {
		if cap(m.savedInput) < len(input) {
			m.savedInput = make([]float32, len(input))
		}
		m.savedInput = m.savedInput[:len(input)]
		copy(m.savedInput, input)
	}

	// GPU acceleration
	if met, ok := m.device.(*MetalDevice); ok && met.IsAvailable() {
		inBuf := met.CreateBuffer(input)
		outBuf := met.CreateBuffer(m.outputBuf[:requiredOutput])
		argBuf := met.CreateIntBuffer(m.argmaxBuf[:requiredOutput])
		defer inBuf.Free()
		defer outBuf.Free()
		defer argBuf.Free()

		met.MaxPool2DPersistent(inBuf, outBuf, argBuf, batchSize, m.inChannels, m.inputHeight, m.inputWidth, m.outputHeight, m.outputWidth, m.kernelSize, m.stride, m.padding)
		outBuf.Read(m.outputBuf[:requiredOutput])
		argBuf.ReadInt(m.argmaxBuf[:requiredOutput])
		return m.outputBuf[:requiredOutput]
	}

	for i := 0; i < batchSize; i++ {
		// Use a temporary slice for the output of each sample to avoid overwriting m.outputBuf
		sampleInput := input[i*inSize : (i+1)*inSize]

		// We need to manually perform pooling here or fix ForwardWithArena to accept an offset
		// For simplicity and correctness, let's just use the logic here
		outH := m.outputHeight
		outW := m.outputWidth
		kernelSize := m.kernelSize
		stride := m.stride
		padding := m.padding
		inputHeight := m.inputHeight
		inputWidth := m.inputWidth
		inChannels := m.inChannels

		channelStride := inputHeight * inputWidth
		outputChannelStride := outH * outW
		sampleOutputOffset := i * inChannels * outputChannelStride

		for c := 0; c < inChannels; c++ {
			channelOffset := c * channelStride
			outputOffset := sampleOutputOffset + c * outputChannelStride

			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					maxVal := float32(math.Inf(-1))
					maxIdx := -1

					for kh := 0; kh < kernelSize; kh++ {
						for kw := 0; kw < kernelSize; kw++ {
							inH := oh*stride + kh - padding
							inW := ow*stride + kw - padding

							if inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth {
								idx := channelOffset + inH*inputWidth + inW
								val := sampleInput[idx]
								if val > maxVal {
									maxVal = val
									maxIdx = i*inSize + idx
								}
							}
						}
					}

					pos := outputOffset + oh*outW + ow
					m.outputBuf[pos] = maxVal
					m.argmaxBuf[pos] = int32(maxIdx)
				}
			}
		}
	}
	return m.outputBuf[:requiredOutput]
}

// Backward performs backpropagation through the max pooling layer.
func (m *MaxPool2D) Backward(grad []float32) []float32 {
	totalInput := len(m.savedInput)
	if totalInput == 0 {
		return nil
	}
	if len(m.gradInBuf) < totalInput {
		m.gradInBuf = make([]float32, totalInput)
	}
	gradIn := m.gradInBuf[:totalInput]

	// GPU acceleration
	if met, ok := m.device.(*MetalDevice); ok && met.IsAvailable() {
		// Clear gradient buffer first
		for i := range gradIn {
			gradIn[i] = 0
		}
		gOutBuf := met.CreateBuffer(grad)
		gInBuf := met.CreateBuffer(gradIn)
		argBuf := met.CreateIntBuffer(m.argmaxBuf)
		defer gOutBuf.Free()
		defer gInBuf.Free()
		defer argBuf.Free()

		met.MaxPool2DBackwardPersistent(gOutBuf, gInBuf, argBuf, 1, m.inChannels, m.inputHeight, m.inputWidth, m.outputHeight, m.outputWidth, m.kernelSize, m.stride, m.padding)
		gInBuf.Read(gradIn)
		return gradIn
	}

	// Clear gradient buffer
	for i := range gradIn {
		gradIn[i] = 0
	}

	outChannels := m.inChannels
	outH := m.outputHeight
	outW := m.outputWidth
	outputChannelStride := outH * outW

	// For each output position, pass gradient only to the max input position
	for c := 0; c < outChannels; c++ {
		outputOffset := c * outputChannelStride

		for pos := 0; pos < outputChannelStride; pos++ {
			gradVal := grad[outputOffset+pos]
			maxIdx := m.argmaxBuf[outputOffset+pos]

			if maxIdx >= 0 {
				gradIn[maxIdx] += gradVal
			}
		}
	}

	return gradIn
}

func (m *MaxPool2D) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return m.Backward(grad)
	}

	inSize := m.InSize()
	outSize := m.OutSize()
	requiredInput := batchSize * inSize

	if len(m.gradInBuf) < requiredInput {
		m.gradInBuf = make([]float32, requiredInput)
	}
	gradIn := m.gradInBuf[:requiredInput]

	// GPU acceleration
	if met, ok := m.device.(*MetalDevice); ok && met.IsAvailable() {
		// Clear gradient buffer first
		for i := range gradIn {
			gradIn[i] = 0
		}
		gOutBuf := met.CreateBuffer(grad)
		gInBuf := met.CreateBuffer(gradIn)
		argBuf := met.CreateIntBuffer(m.argmaxBuf)
		defer gOutBuf.Free()
		defer gInBuf.Free()
		defer argBuf.Free()

		met.MaxPool2DBackwardPersistent(gOutBuf, gInBuf, argBuf, batchSize, m.inChannels, m.inputHeight, m.inputWidth, m.outputHeight, m.outputWidth, m.kernelSize, m.stride, m.padding)
		gInBuf.Read(gradIn)
		return gradIn
	}

	for i := 0; i < batchSize; i++ {
		sampleGrad := grad[i*outSize : (i+1)*outSize]
		// Use absolute indexing from argmaxBuf
		sampleArgmax := m.argmaxBuf[i*outSize : (i+1)*outSize]

		for j := range sampleGrad {
			maxIdx := sampleArgmax[j]
			if maxIdx >= 0 {
				gradIn[maxIdx] += sampleGrad[j]
			}
		}
	}
	return gradIn
}

func (m *MaxPool2D) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return m.BackwardBatch(grad, batchSize)
}

// Params returns layer parameters (empty for MaxPool2D).
func (m *MaxPool2D) Params() []float32 {
	return make([]float32, 0)
}

// SetParams sets layer parameters (no-op for MaxPool2D).
func (m *MaxPool2D) SetParams(params []float32) {
}

// Gradients returns layer gradients (empty for MaxPool2D).
func (m *MaxPool2D) Gradients() []float32 {
	return make([]float32, 0)
}

// SetGradients sets layer gradients (no-op for MaxPool2D).
func (m *MaxPool2D) SetGradients(gradients []float32) {
}

// InSize returns the total input size (channels * height * width).
func (m *MaxPool2D) InSize() int {
	if m.inputHeight == 0 || m.inputWidth == 0 {
		return 0
	}
	return m.inChannels * m.inputHeight * m.inputWidth
}

// OutSize returns the total output size (channels * outputHeight * outputWidth).
func (m *MaxPool2D) OutSize() int {
	if m.outputHeight == 0 || m.outputWidth == 0 {
		return 0
	}
	return m.inChannels * m.outputHeight * m.outputWidth
}

// Reset resets the max pooling layer.
func (m *MaxPool2D) Reset() {
	m.inputHeight = 0
	m.inputWidth = 0
	m.outputHeight = 0
	m.outputWidth = 0
}

// ClearGradients zeroes out the accumulated gradients (no-op for MaxPool2D).
func (m *MaxPool2D) ClearGradients() {
}

// Clone creates a deep copy of the max pooling layer.
func (m *MaxPool2D) Clone() Layer {
	newM := NewMaxPool2D(m.inChannels, m.kernelSize, m.stride, m.padding)
	newM.inputHeight = m.inputHeight
	newM.inputWidth = m.inputWidth
	newM.outputHeight = m.outputHeight
	newM.outputWidth = m.outputWidth
	newM.device = m.device
	return newM
}

func (m *MaxPool2D) LightweightClone(params []float32, grads []float32) Layer {
	newM := &MaxPool2D{
		inChannels:   m.inChannels,
		kernelSize:   m.kernelSize,
		stride:       m.stride,
		padding:      m.padding,
		inputHeight:  m.inputHeight,
		inputWidth:   m.inputWidth,
		outputHeight: m.outputHeight,
		outputWidth:  m.outputWidth,
		outputBuf:    make([]float32, 0),
		gradInBuf:    make([]float32, 0),
		argmaxBuf:    make([]int32, 0),
		savedInput:   make([]float32, 0),
		training:     m.training,
		device:       m.device,
	}
	return newM
}

// GetKernelSize returns the kernel size.
func (m *MaxPool2D) GetKernelSize() int {
	return m.kernelSize
}

// GetStride returns the stride.
func (m *MaxPool2D) GetStride() int {
	return m.stride
}

// GetPadding returns the padding.
func (m *MaxPool2D) GetPadding() int {
	return m.padding
}

// GetArgmax returns the argmax indices buffer (for testing/verification).
func (m *MaxPool2D) GetArgmax() []int32 {
	return m.argmaxBuf
}

// GetOutputDimensions returns the spatial dimensions of the output.
func (m *MaxPool2D) GetOutputDimensions() (int, int) {
	if m.inputHeight == 0 || m.inputWidth == 0 {
		return 0, 0
	}
	return m.computeOutputSize(m.inputHeight, m.inputWidth)
}

// AccumulateBackward performs backpropagation and accumulates gradients.
func (m *MaxPool2D) AccumulateBackward(grad []float32) []float32 {
	return m.Backward(grad)
}
