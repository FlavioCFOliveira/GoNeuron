// Package layer provides neural network layer implementations.
package layer

import (
	"fmt"
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
	outputBuf    []float32
	gradInBuf    []float32
	argmaxBuf    []int // Stores index of max value for each output position

	// Saved input for backward pass
	savedInput []float32

	device Device
}

// NewMaxPool2D creates a new 2D max pooling layer.
// inChannels: number of input channels
// kernelSize: size of pooling window (square)
// stride: stride for pooling (defaults to kernelSize)
// padding: zero padding size
func NewMaxPool2D(inChannels, kernelSize, stride, padding int) *MaxPool2D {
	return &MaxPool2D{
		inChannels:   inChannels,
		kernelSize:   kernelSize,
		stride:       stride,
		padding:      padding,
		outputBuf:    make([]float32, 0),
		gradInBuf:    make([]float32, 0),
		argmaxBuf:    make([]int, 0),
		savedInput:   make([]float32, 0),
		device:       &CPUDevice{},
	}
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

// Forward performs a forward pass through the max pooling layer.
// input: flattened [inChannels, inputHeight, inputWidth]
// Returns: flattened [outChannels, outputHeight, outputWidth]
func (m *MaxPool2D) Forward(input []float32) []float32 {
	totalInput := len(input)
	if totalInput == 0 {
		return make([]float32, 0)
	}

	if m.inChannels <= 0 {
		panic("MaxPool2D: inChannels not set. Use NewMaxPool2D with inChannels parameter.")
	}

	if totalInput%m.inChannels != 0 {
		panic(fmt.Sprintf("MaxPool2D: input length %d not divisible by inChannels %d", totalInput, m.inChannels))
	}

	channelSize := totalInput / m.inChannels

	// Infer input dimensions (assuming square for now)
	m.inputHeight = int(math.Sqrt(float64(channelSize)))
	if m.inputHeight*m.inputHeight == channelSize {
		m.inputWidth = m.inputHeight
	} else {
		m.inputWidth = channelSize / m.inputHeight
		if m.inputHeight*m.inputWidth != channelSize {
			// Fall back
			m.inputWidth = channelSize / m.inputHeight
		}
	}

	// Compute output dimensions
	m.outputHeight, m.outputWidth = m.computeOutputSize(m.inputHeight, m.inputWidth)

	outChannels := m.inChannels // Each channel is pooled independently
	outSize := m.outputHeight * m.outputWidth

	// Ensure buffers are sized correctly
	requiredOutput := outChannels * outSize
	if len(m.outputBuf) < requiredOutput {
		m.outputBuf = make([]float32, requiredOutput)
	}
	if cap(m.argmaxBuf) < requiredOutput {
		m.argmaxBuf = make([]int, requiredOutput)
	} else {
		m.argmaxBuf = m.argmaxBuf[:requiredOutput]
	}
	if cap(m.gradInBuf) < totalInput {
		m.gradInBuf = make([]float32, totalInput)
	}

	if cap(m.savedInput) < totalInput {
		m.savedInput = make([]float32, totalInput)
	}
	m.savedInput = m.savedInput[:totalInput]
	copy(m.savedInput, input)

	// Max pooling: for each channel, then for each output position
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
						// Calculate input position with padding
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
				m.argmaxBuf[pos] = maxIdx
			}
		}
	}

	return m.outputBuf[:requiredOutput]
}

// Backward performs backpropagation through the max pooling layer.
func (m *MaxPool2D) Backward(grad []float32) []float32 {
	totalInput := len(m.savedInput)
	gradIn := m.gradInBuf[:totalInput]

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

// Params returns layer parameters (empty for MaxPool2D).
func (m *MaxPool2D) Params() []float32 {
	return make([]float32, 0)
}

// SetParams sets layer parameters (no-op for MaxPool2D).
func (m *MaxPool2D) SetParams(params []float32) {
	// No parameters to set
}

// Gradients returns layer gradients (empty for MaxPool2D).
func (m *MaxPool2D) Gradients() []float32 {
	return make([]float32, 0)
}

// SetGradients sets layer gradients (no-op for MaxPool2D).
func (m *MaxPool2D) SetGradients(gradients []float32) {
	// No parameters to set
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
	// No parameters to clear
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
func (m *MaxPool2D) GetArgmax() []int {
	return m.argmaxBuf
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For MaxPool2D, gradients are already accumulated in Backward, so this just calls Backward.
func (m *MaxPool2D) AccumulateBackward(grad []float32) []float32 {
	return m.Backward(grad)
}
