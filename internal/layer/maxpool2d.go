// Package layer provides neural network layer implementations.
package layer

import "math"

// MaxPool2D implements 2D max pooling.
// Downsamples by taking the maximum over sliding windows.
// Stores argmax indices for correct gradient flow during backward pass.
type MaxPool2D struct {
	// Pooling parameters
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
	outputBuf    []float64
	gradInBuf    []float64
	argmaxBuf    []int // Stores index of max value for each output position

	// Saved input for backward pass
	savedInput []float64
}

// NewMaxPool2D creates a new 2D max pooling layer.
// kernelSize: size of pooling window (square)
// stride: stride for pooling (defaults to kernelSize)
// padding: zero padding size
func NewMaxPool2D(kernelSize, stride, padding int) *MaxPool2D {
	return &MaxPool2D{
		kernelSize:   kernelSize,
		stride:       stride,
		padding:      padding,
		outputBuf:    make([]float64, 0),
		gradInBuf:    make([]float64, 0),
		argmaxBuf:    make([]int, 0),
		savedInput:   make([]float64, 0),
	}
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
func (m *MaxPool2D) Forward(input []float64) []float64 {
	totalInput := len(input)
	if totalInput == 0 {
		return make([]float64, 0)
	}

	// Infer input dimensions
	channelSize := totalInput
	m.inputHeight = int(math.Sqrt(float64(channelSize)))
	if m.inputHeight*m.inputHeight == channelSize {
		m.inputWidth = m.inputHeight
	} else {
		m.inputWidth = channelSize / m.inputHeight
	}

	// Compute output dimensions
	m.outputHeight, m.outputWidth = m.computeOutputSize(m.inputHeight, m.inputWidth)

	outChannels := 1 // For now, assume single channel
	outSize := m.outputHeight * m.outputWidth

	// Ensure buffers are sized correctly
	requiredOutput := outChannels * outSize
	if len(m.outputBuf) < requiredOutput {
		m.outputBuf = make([]float64, requiredOutput)
		m.argmaxBuf = make([]int, requiredOutput)
		m.gradInBuf = make([]float64, totalInput)
	}

	if cap(m.savedInput) < len(input) {
		m.savedInput = make([]float64, len(input))
	}
	copy(m.savedInput, input)

	// Max pooling: for each output position
	outH := m.outputHeight
	outW := m.outputWidth
	kernelSize := m.kernelSize
	stride := m.stride
	padding := m.padding
	inputHeight := m.inputHeight
	inputWidth := m.inputWidth

	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			maxVal := math.Inf(-1)
			maxIdx := -1

			for kh := 0; kh < kernelSize; kh++ {
				for kw := 0; kw < kernelSize; kw++ {
					// Calculate input position with padding
					inH := oh*stride + kh - padding
					inW := ow*stride + kw - padding

					if inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth {
						idx := inH*inputWidth + inW
						if input[idx] > maxVal {
							maxVal = input[idx]
							maxIdx = idx
						}
					}
				}
			}

			pos := oh*outW + ow
			m.outputBuf[pos] = maxVal
			m.argmaxBuf[pos] = maxIdx
		}
	}

	return m.outputBuf[:requiredOutput]
}

// Backward performs backpropagation through the max pooling layer.
func (m *MaxPool2D) Backward(grad []float64) []float64 {
	totalInput := len(m.savedInput)
	if len(m.gradInBuf) != totalInput {
		m.gradInBuf = make([]float64, totalInput)
	}

	// Clear gradient buffer
	for i := range m.gradInBuf {
		m.gradInBuf[i] = 0
	}

	// For each output position, pass gradient only to the max input position
	for pos := 0; pos < len(m.argmaxBuf); pos++ {
		gradVal := grad[pos]
		maxIdx := m.argmaxBuf[pos]

		if maxIdx >= 0 {
			m.gradInBuf[maxIdx] += gradVal
		}
	}

	return m.gradInBuf
}

// Params returns layer parameters (empty for MaxPool2D).
func (m *MaxPool2D) Params() []float64 {
	return make([]float64, 0)
}

// SetParams sets layer parameters (no-op for MaxPool2D).
func (m *MaxPool2D) SetParams(params []float64) {
	// No parameters to set
}

// Gradients returns layer gradients (empty for MaxPool2D).
func (m *MaxPool2D) Gradients() []float64 {
	return make([]float64, 0)
}

// SetGradients sets layer gradients (no-op for MaxPool2D).
func (m *MaxPool2D) SetGradients(gradients []float64) {
	// No parameters to set
}

// InSize returns the input size (flattened).
func (m *MaxPool2D) InSize() int {
	return m.inputHeight * m.inputWidth
}

// OutSize returns the output size (flattened).
func (m *MaxPool2D) OutSize() int {
	return m.outputHeight * m.outputWidth
}

// Reset resets the max pooling layer.
func (m *MaxPool2D) Reset() {
	m.inputHeight = 0
	m.inputWidth = 0
	m.outputHeight = 0
	m.outputWidth = 0
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
