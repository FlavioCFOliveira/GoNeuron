// Package layer provides neural network layer implementations.
package layer

import (
	"fmt"
	"math"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

// Conv2D implements a 2D convolutional layer.
// Uses direct convolution computation for correctness.
type Conv2D struct {
	// Input/output dimensions
	inChannels  int
	outChannels int
	kernelSize  int
	stride      int
	padding     int

	// Input dimensions (set during forward if not explicitly set)
	inputHeight int
	inputWidth  int

	// Explicitly set dimensions (if set, overrides automatic inference)
	setInputHeight int
	setInputWidth  int

	// Parameters
	params  []float32 // Contiguous weights + biases
	weights []float32 // View of params
	biases  []float32 // View of params

	// Activation function
	activation activations.Activation

	// Pre-allocated buffers
	grads       []float32 // Contiguous gradWeights + gradBiases
	preActBuf   []float32 // Contains pre-activation values (z = w*x + b)
	outputBuf   []float32 // Contains post-activation values (activation(z))
	gradWeights []float32
	gradBiases  []float32
	gradInBuf   []float32

	// Saved input for backward pass
	savedInput []float32

	training bool

	device Device
}

// NewConv2D creates a new 2D convolutional layer.
func NewConv2D(inChannels, outChannels, kernelSize, stride, padding int,
	activation activations.Activation) *Conv2D {

	c := &Conv2D{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelSize:  kernelSize,
		stride:      stride,
		padding:     padding,
		activation:  activation,
		device:      &CPUDevice{},
	}

	if inChannels != -1 {
		c.Build(inChannels)
	}

	return c
}

// Build initializes the layer with the given input size (channels).
func (c *Conv2D) Build(inChannels int) {
	c.inChannels = inChannels
	weightSize := c.outChannels * inChannels * c.kernelSize * c.kernelSize
	biasSize := c.outChannels
	c.params = make([]float32, weightSize+biasSize)
	c.weights = c.params[:weightSize]
	c.biases = c.params[weightSize:]

	// He initialization
	scale := float32(math.Sqrt(2.0 / float64(inChannels*c.kernelSize*c.kernelSize)))
	rng := NewRNG(42)

	for i := range c.weights {
		c.weights[i] = rng.RandFloat()*2*scale - scale
	}
	for i := range c.biases {
		c.biases[i] = rng.RandFloat()*0.2 - 0.1
	}

	c.grads = make([]float32, weightSize+biasSize)
	c.gradWeights = c.grads[:weightSize]
	c.gradBiases = c.grads[weightSize:]
	c.outputBuf = make([]float32, 0)
	c.gradInBuf = make([]float32, 0)
	c.savedInput = make([]float32, 0)
}

// SetDevice sets the computation device for the convolutional layer.
func (c *Conv2D) SetDevice(device Device) {
	c.device = device
}

// computeOutputSize calculates the output spatial dimensions
func (c *Conv2D) computeOutputSize(inputHeight, inputWidth int) (int, int) {
	// Output size: (input + 2*padding - kernel) / stride + 1
	outH := (inputHeight + 2*c.padding - c.kernelSize) / c.stride + 1
	outW := (inputWidth + 2*c.padding - c.kernelSize) / c.stride + 1
	return outH, outW
}

// SetInputDimensions explicitly sets the input dimensions for the next forward pass.
// This allows non-square inputs and avoids automatic inference.
func (c *Conv2D) SetInputDimensions(height, width int) {
	c.setInputHeight = height
	c.setInputWidth = width
	c.inputHeight = height
	c.inputWidth = width
}

// SetTraining sets whether the layer is in training mode.
func (c *Conv2D) SetTraining(training bool) {
	c.training = training
}

func (c *Conv2D) ForwardWithArena(input []float32, arena *[]float32, offset *int) []float32 {
	// Infer input dimensions from length
	totalInput := len(input)
	if totalInput%c.inChannels != 0 {
		panic("Conv2D: input length not divisible by inChannels")
	}
	channelSize := totalInput / c.inChannels

	// Use explicitly set dimensions if available, otherwise infer
	var inputHeight, inputWidth int
	if c.setInputHeight > 0 && c.setInputWidth > 0 {
		inputHeight = c.setInputHeight
		inputWidth = c.setInputWidth
		// Verify the dimensions match the input length
		if inputHeight*inputWidth != channelSize {
			panic(fmt.Sprintf("Conv2D: input dimensions %dx%d don't match channelSize %d", inputHeight, inputWidth, channelSize))
		}
	} else {
		// Infer dimensions - try square first, then rectangular
		inputHeight = int(math.Sqrt(float64(channelSize)))
		if inputHeight*inputHeight == channelSize {
			inputWidth = inputHeight
		} else {
			// Try to find rectangular dimensions
			inputWidth = channelSize / inputHeight
			if inputHeight*inputWidth != channelSize {
				// Fall back to a reasonable approximation
				inputWidth = channelSize / inputHeight
			}
		}
		c.inputHeight = inputHeight
		c.inputWidth = inputWidth
	}

	// Compute output dimensions
	outH, outW := c.computeOutputSize(inputHeight, inputWidth)

	// Ensure buffers are sized correctly
	requiredOutput := c.outChannels * outH * outW
	if len(c.preActBuf) < requiredOutput {
		c.preActBuf = make([]float32, requiredOutput)
	}
	if len(c.outputBuf) < requiredOutput {
		c.outputBuf = make([]float32, requiredOutput)
	}

	// Ensure gradInBuf is sized correctly for backward pass
	if len(c.gradInBuf) < totalInput {
		c.gradInBuf = make([]float32, totalInput)
	}

	// Save input for backward pass
	if arena != nil && offset != nil {
		if len(*arena) < *offset+len(input) {
			newArena := make([]float32, (*offset+len(input))*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		saved := (*arena)[*offset : *offset+len(input)]
		copy(saved, input)
		c.savedInput = saved
		*offset += len(input)
	} else {
		if cap(c.savedInput) < len(input) {
			c.savedInput = make([]float32, len(input))
		}
		copy(c.savedInput, input)
	}

	// Cache dimensions for backward pass
	c.inputHeight = inputHeight
	c.inputWidth = inputWidth

	// Convolution logic
	outSize := outH * outW
	kernelSize := c.kernelSize
	stride := c.stride
	padding := c.padding
	inChannels := c.inChannels
	outChannels := c.outChannels

	// Clear pre-activation buffer
	for i := 0; i < requiredOutput; i++ {
		c.preActBuf[i] = 0
	}

	// Pre-compute weight stride values
	icWeightStride := kernelSize * kernelSize
	ocWeightStride := inChannels * icWeightStride

	for oc := 0; oc < outChannels; oc++ {
		ocWeightBase := oc * ocWeightStride
		ocOutBase := oc * outSize

		for ic := 0; ic < inChannels; ic++ {
			icWeightBase := ocWeightBase + ic*icWeightStride
			inputChannelOffset := ic * inputHeight * inputWidth

			for kh := 0; kh < kernelSize; kh++ {
				khWeightBase := icWeightBase + kh*kernelSize

				for kw := 0; kw < kernelSize; kw++ {
					weightIdx := khWeightBase + kw
					wVal := c.weights[weightIdx]

					for oh := 0; oh < outH; oh++ {
						inH := oh*stride + kh - padding
						if inH >= 0 && inH < inputHeight {
							inHOffset := inputChannelOffset + inH*inputWidth
							ohOffset := ocOutBase + oh*outW
							for ow := 0; ow < outW; ow++ {
								inW := ow*stride + kw - padding
								if inW >= 0 && inW < inputWidth {
									inputIdx := inHOffset + inW
									pos := ohOffset + ow
									c.preActBuf[pos] += wVal * input[inputIdx]
								}
							}
						}
					}
				}
			}
		}

		// Add bias and apply activation for this output channel
		biasVal := c.biases[oc]
		for oh := 0; oh < outH; oh++ {
			ohOffset := ocOutBase + oh*outW
			for ow := 0; ow < outW; ow++ {
				pos := ohOffset + ow
				sum := c.preActBuf[pos] + biasVal
				c.preActBuf[pos] = sum
				c.outputBuf[pos] = c.activation.Activate(sum)
			}
		}
	}

	return c.outputBuf[:requiredOutput]
}

// Forward performs a forward pass through the convolutional layer.
func (c *Conv2D) Forward(input []float32) []float32 {
	return c.ForwardWithArena(input, nil, nil)
}


// Backward performs backpropagation through the convolutional layer.
// grad: gradient of loss w.r.t. activated output (shape: [outChannels, outH, outW] flattened)
// Returns: gradient of loss w.r.t. input
func (c *Conv2D) Backward(grad []float32) []float32 {
	// Output dimensions
	outH, outW := c.computeOutputSize(c.inputHeight, c.inputWidth)
	outSize := outH * outW

	kernelSize := c.kernelSize
	inChannels := c.inChannels
	outChannels := c.outChannels
	stride := c.stride
	padding := c.padding
	inputHeight := c.inputHeight
	inputWidth := c.inputWidth

	// Use pre-allocated input gradient buffer
	gradInput := c.gradInBuf[:inChannels*inputHeight*inputWidth]

	// Clear input gradient buffer
	for i := range gradInput {
		gradInput[i] = 0
	}

	// Pre-compute weight stride values
	icWeightStride := kernelSize * kernelSize
	ocWeightStride := inChannels * icWeightStride

	// For each output channel
	for oc := 0; oc < outChannels; oc++ {
		ocWeightBase := oc * ocWeightStride
		ocOutBase := oc * outSize

		// For each output position
		for oh := 0; oh < outH; oh++ {
			ohOffset := ocOutBase + oh*outW
			for ow := 0; ow < outW; ow++ {
				pos := ohOffset + ow

				// Gradient after activation: dL/dz = dL/d(output) * activation'(z)
				gradAfterAct := grad[pos] * c.activation.Derivative(c.preActBuf[pos])

				// Accumulate gradient for bias
				c.gradBiases[oc] += gradAfterAct

				// For each input channel and kernel position
				for ic := 0; ic < inChannels; ic++ {
					icWeightBase := ocWeightBase + ic*icWeightStride
					inputChannelOffset := ic * inputHeight * inputWidth

					for kh := 0; kh < kernelSize; kh++ {
						inH := oh*stride + kh - padding
						if inH >= 0 && inH < inputHeight {
							inHOffset := inputChannelOffset + inH*inputWidth
							khWeightBase := icWeightBase + kh*kernelSize

							for kw := 0; kw < kernelSize; kw++ {
								inW := ow*stride + kw - padding
								if inW >= 0 && inW < inputWidth {
									inputIdx := inHOffset + inW
									weightIdx := khWeightBase + kw

									// Accumulate weight gradient
									c.gradWeights[weightIdx] += gradAfterAct * c.savedInput[inputIdx]

									// Accumulate input gradient
									gradInput[inputIdx] += gradAfterAct * c.weights[weightIdx]
								}
							}
						}
					}
				}
			}
		}
	}

	return gradInput
}

// Params returns layer parameters flattened.
func (c *Conv2D) Params() []float32 {
	return c.params
}

// SetParams updates weights and biases from a flattened slice.
func (c *Conv2D) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &c.params[0] != &params[0] {
		if len(params) == len(c.params) {
			c.params = params
			c.updateViews()
		} else {
			copy(c.params, params)
		}
	}
}

func (c *Conv2D) updateViews() {
	weightSize := c.outChannels * c.inChannels * c.kernelSize * c.kernelSize
	c.weights = c.params[:weightSize]
	c.biases = c.params[weightSize:]
}

// Gradients returns all convolutional layer gradients flattened.
func (c *Conv2D) Gradients() []float32 {
	return c.grads
}

// SetGradients sets gradients from a flattened slice (in-place).
func (c *Conv2D) SetGradients(gradients []float32) {
	if len(gradients) == 0 {
		return
	}
	if &c.grads[0] != &gradients[0] {
		if len(gradients) == len(c.grads) {
			c.grads = gradients
			c.updateGradViews()
		} else {
			copy(c.grads, gradients)
		}
	}
}

func (c *Conv2D) updateGradViews() {
	weightSize := c.outChannels * c.inChannels * c.kernelSize * c.kernelSize
	c.gradWeights = c.grads[:weightSize]
	c.gradBiases = c.grads[weightSize:]
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For Conv2D, gradients are already accumulated in Backward, so this just calls Backward.
func (c *Conv2D) AccumulateBackward(grad []float32) []float32 {
	return c.Backward(grad)
}

// ClearGradients zeroes out the accumulated gradients.
func (c *Conv2D) ClearGradients() {
	for i := range c.grads {
		c.grads[i] = 0
	}
}

// Clone creates a deep copy of the convolutional layer.
func (c *Conv2D) Clone() Layer {
	newC := NewConv2D(c.inChannels, c.outChannels, c.kernelSize, c.stride, c.padding, c.activation)
	copy(newC.params, c.params)
	newC.setInputHeight = c.setInputHeight
	newC.setInputWidth = c.setInputWidth
	newC.device = c.device
	return newC
}

func (c *Conv2D) LightweightClone(params []float32, grads []float32) Layer {
	weightSize := c.outChannels * c.inChannels * c.kernelSize * c.kernelSize
	newC := &Conv2D{
		inChannels:     c.inChannels,
		outChannels:    c.outChannels,
		kernelSize:     c.kernelSize,
		stride:         c.stride,
		padding:        c.padding,
		setInputHeight: c.setInputHeight,
		setInputWidth:  c.setInputWidth,
		activation:     c.activation,
		params:         params,
		weights:        params[:weightSize],
		biases:         params[weightSize:],
		grads:          grads,
		gradWeights:    grads[:weightSize],
		gradBiases:     grads[weightSize:],
		outputBuf:      make([]float32, 0),
		gradInBuf:      make([]float32, 0),
		savedInput:     make([]float32, 0),
		training:       c.training,
		device:         c.device,
	}
	return newC
}

// InSize returns the number of input channels.
func (c *Conv2D) InSize() int {
	return c.inChannels
}

// OutSize returns the number of output channels.
func (c *Conv2D) OutSize() int {
	if c.inputHeight > 0 && c.inputWidth > 0 {
		h, w := c.computeOutputSize(c.inputHeight, c.inputWidth)
		return c.outChannels * h * w
	}
	// Fallback for summary before first forward
	return c.outChannels
}

func (c *Conv2D) NamedParams() []NamedParam {
	return []NamedParam{
		{
			Name:  "weights",
			Shape: []int{c.outChannels, c.inChannels, c.kernelSize, c.kernelSize},
			Data:  c.weights,
		},
		{
			Name:  "biases",
			Shape: []int{c.outChannels},
			Data:  c.biases,
		},
	}
}

// Reset clears any cached state (for compatibility with other layer types).
func (c *Conv2D) Reset() {
	c.savedInput = nil
}

// OutputBuf returns the raw output buffer for advanced use.
// The caller should not modify this buffer directly.
func (c *Conv2D) OutputBuf() []float32 {
	return c.outputBuf
}

// SetOutputBuf allows setting a custom output buffer.
// This is useful for chaining layers efficiently.
func (c *Conv2D) SetOutputBuf(buf []float32) {
	c.outputBuf = buf
}

// GetKernelSize returns the kernel size.
func (c *Conv2D) GetKernelSize() int {
	return c.kernelSize
}

// GetStride returns the stride.
func (c *Conv2D) GetStride() int {
	return c.stride
}

// GetPadding returns the padding.
func (c *Conv2D) GetPadding() int {
	return c.padding
}

// GetActivation returns the activation function.
func (c *Conv2D) GetActivation() activations.Activation {
	return c.activation
}

// GetOutputDimensions returns the spatial dimensions of the output.
func (c *Conv2D) GetOutputDimensions() (int, int) {
	if c.inputHeight == 0 || c.inputWidth == 0 {
		return 0, 0
	}
	return c.computeOutputSize(c.inputHeight, c.inputWidth)
}
