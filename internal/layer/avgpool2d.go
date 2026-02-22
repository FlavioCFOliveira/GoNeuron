// Package layer provides neural network layer implementations.
package layer

import "math"

// AvgPool2D implements 2D average pooling.
// Downsamples by taking the average over sliding windows.
type AvgPool2D struct {
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
	countBuf     []int // Count of inputs contributing to each output position

	// Saved input for backward pass
	savedInput []float64

	device Device
}

// NewAvgPool2D creates a new 2D average pooling layer.
// kernelSize: size of pooling window (square)
// stride: stride for pooling (defaults to kernelSize)
// padding: zero padding size
func NewAvgPool2D(kernelSize, stride, padding int) *AvgPool2D {
	return &AvgPool2D{
		kernelSize: kernelSize,
		stride:     stride,
		padding:    padding,
		outputBuf:  make([]float64, 0),
		gradInBuf:  make([]float64, 0),
		countBuf:   make([]int, 0),
		savedInput: make([]float64, 0),
		device:     &CPUDevice{},
	}
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

// Forward performs a forward pass through the average pooling layer.
// input: flattened [inChannels, inputHeight, inputWidth]
// Returns: flattened [outChannels, outputHeight, outputWidth]
func (a *AvgPool2D) Forward(input []float64) []float64 {
	totalInput := len(input)
	if totalInput == 0 {
		return make([]float64, 0)
	}

	// Infer input dimensions
	channelSize := totalInput
	a.inputHeight = int(math.Sqrt(float64(channelSize)))
	if a.inputHeight*a.inputHeight == channelSize {
		a.inputWidth = a.inputHeight
	} else {
		a.inputWidth = channelSize / a.inputHeight
	}

	// Compute output dimensions
	a.outputHeight, a.outputWidth = a.computeOutputSize(a.inputHeight, a.inputWidth)

	outChannels := 1 // For now, assume single channel
	outSize := a.outputHeight * a.outputWidth

	// Ensure buffers are sized correctly
	requiredOutput := outChannels * outSize
	if len(a.outputBuf) < requiredOutput {
		a.outputBuf = make([]float64, requiredOutput)
		a.countBuf = make([]int, requiredOutput)
		a.gradInBuf = make([]float64, totalInput)
	}

	if cap(a.savedInput) < len(input) {
		a.savedInput = make([]float64, len(input))
	}
	copy(a.savedInput, input)

	// Average pooling: for each output position
	outH := a.outputHeight
	outW := a.outputWidth
	kernelSize := a.kernelSize
	stride := a.stride
	padding := a.padding
	inputHeight := a.inputHeight
	inputWidth := a.inputWidth

	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			sum := 0.0
			count := 0

			for kh := 0; kh < kernelSize; kh++ {
				for kw := 0; kw < kernelSize; kw++ {
					// Calculate input position with padding
					inH := oh*stride + kh - padding
					inW := ow*stride + kw - padding

					if inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth {
						idx := inH*inputWidth + inW
						sum += input[idx]
						count++
					}
				}
			}

			pos := oh*outW + ow
			if count > 0 {
				a.outputBuf[pos] = sum / float64(count)
				a.countBuf[pos] = count
			} else {
				a.outputBuf[pos] = 0
				a.countBuf[pos] = 0
			}
		}
	}

	return a.outputBuf[:requiredOutput]
}

// Backward performs backpropagation through the average pooling layer.
func (a *AvgPool2D) Backward(grad []float64) []float64 {
	totalInput := len(a.savedInput)
	gradIn := a.gradInBuf[:totalInput]

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
			gradVal := grad[pos]
			count := a.countBuf[pos]

			if count > 0 {
				// Distribute gradient equally among contributing inputs
				gradPerInput := gradVal / float64(count)

				for kh := 0; kh < kernelSize; kh++ {
					for kw := 0; kw < kernelSize; kw++ {
						inH := oh*stride + kh - padding
						inW := ow*stride + kw - padding

						if inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth {
							inputIdx := inH*inputWidth + inW
							gradIn[inputIdx] += gradPerInput
						}
					}
				}
			}
		}
	}

	return gradIn
}

// Params returns layer parameters (empty for AvgPool2D).
func (a *AvgPool2D) Params() []float64 {
	return make([]float64, 0)
}

// SetParams sets layer parameters (no-op for AvgPool2D).
func (a *AvgPool2D) SetParams(params []float64) {
	// No parameters to set
}

// Gradients returns layer gradients (empty for AvgPool2D).
func (a *AvgPool2D) Gradients() []float64 {
	return make([]float64, 0)
}

// SetGradients sets layer gradients (no-op for AvgPool2D).
func (a *AvgPool2D) SetGradients(gradients []float64) {
	// No parameters to set
}

// InSize returns the input size (flattened).
func (a *AvgPool2D) InSize() int {
	return a.inputHeight * a.inputWidth
}

// OutSize returns the output size (flattened).
func (a *AvgPool2D) OutSize() int {
	return a.outputHeight * a.outputWidth
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
	// No parameters to clear
}

// Clone creates a deep copy of the average pooling layer.
func (a *AvgPool2D) Clone() Layer {
	newA := NewAvgPool2D(a.kernelSize, a.stride, a.padding)
	newA.inputHeight = a.inputHeight
	newA.inputWidth = a.inputWidth
	newA.outputHeight = a.outputHeight
	newA.outputWidth = a.outputWidth
	newA.device = a.device
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

// AccumulateBackward performs backpropagation and accumulates gradients.
// For AvgPool2D, gradients are already accumulated in Backward, so this just calls Backward.
func (a *AvgPool2D) AccumulateBackward(grad []float64) []float64 {
	return a.Backward(grad)
}
