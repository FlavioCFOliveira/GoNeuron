// Package layer provides neural network layer implementations.
package layer

import (
	"fmt"
	"math"
	"unsafe"

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
	savedInput        []float32
	savedInputOffsets []int
	arenaPtr          *[]float32

	training bool

	device Device

	// Metal state
	metalState unsafe.Pointer
	bufInput   *MetalBuffer
	bufOutput  *MetalBuffer
	bufGradOut *MetalBuffer
	bufGradIn  *MetalBuffer
	bufGradW   *MetalBuffer
	bufGradB   *MetalBuffer
}

// NewConv2D creates a new 2D convolutional layer.
func NewConv2D(inChannels, outChannels, kernelSize, stride, padding int,
	activation activations.Activation) *Conv2D {

	c := &Conv2D{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelSize:  kernelSize,
		stride:      stride,
		padding:           padding,
		activation:        activation,
		device:            &CPUDevice{},
		savedInputOffsets: make([]int, 0, 16),
	}

	if inChannels != -1 {
		c.Build(inChannels)
	}

	return c
}

// Build initializes the layer with the given input size (channels).
func (c *Conv2D) Build(inChannels int) {
	if inChannels <= 0 {
		return
	}
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

	// GPU initialization if needed
	if c.device.Type() == GPU {
		if md, ok := c.device.(*MetalDevice); ok && md.IsAvailable() {
			c.metalState = md.CreateConv2DState(c.inChannels, c.outChannels, c.kernelSize, c.stride, c.padding, c.weights, c.biases)
			c.bufGradW = md.CreateEmptyBuffer(len(c.weights))
			c.bufGradB = md.CreateEmptyBuffer(len(c.biases))
			c.bufInput = md.CreateEmptyBuffer(c.inChannels * c.kernelSize * c.kernelSize)
			c.bufOutput = md.CreateEmptyBuffer(c.outChannels)
			c.bufGradOut = md.CreateEmptyBuffer(c.outChannels)
			c.bufGradIn = md.CreateEmptyBuffer(c.inChannels * c.kernelSize * c.kernelSize)
		}
	}
}

// SetDevice sets the computation device for the convolutional layer.
func (c *Conv2D) SetDevice(device Device) {
	c.device = device
	if device.Type() == GPU && c.metalState == nil && len(c.params) > 0 {
		if md, ok := device.(*MetalDevice); ok && md.IsAvailable() {
			c.metalState = md.CreateConv2DState(c.inChannels, c.outChannels, c.kernelSize, c.stride, c.padding, c.weights, c.biases)
			c.bufGradW = md.CreateEmptyBuffer(len(c.weights))
			c.bufGradB = md.CreateEmptyBuffer(len(c.biases))
			c.bufInput = md.CreateEmptyBuffer(c.inChannels * c.kernelSize * c.kernelSize)
			c.bufOutput = md.CreateEmptyBuffer(c.outChannels)
			c.bufGradOut = md.CreateEmptyBuffer(c.outChannels)
			c.bufGradIn = md.CreateEmptyBuffer(c.inChannels * c.kernelSize * c.kernelSize)
		}
	}
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

func (c *Conv2D) initMetal(batchSize, inH, inW, outH, outW int) {
	if c.metalState != nil {
		return
	}
	mDevice, ok := c.device.(*MetalDevice)
	if !ok {
		return
	}

	c.metalState = mDevice.CreateConv2DState(c.inChannels, c.outChannels, c.kernelSize, c.stride, c.padding, c.weights, c.biases)
	c.bufInput = mDevice.CreateEmptyBuffer(batchSize * c.inChannels * inH * inW)
	c.bufOutput = mDevice.CreateEmptyBuffer(batchSize * c.outChannels * outH * outW)
	c.bufGradOut = mDevice.CreateEmptyBuffer(batchSize * c.outChannels * outH * outW)
	c.bufGradIn = mDevice.CreateEmptyBuffer(batchSize * c.inChannels * inH * inW)
	c.bufGradW = mDevice.CreateEmptyBuffer(len(c.weights))
	c.bufGradB = mDevice.CreateEmptyBuffer(len(c.biases))
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
		c.arenaPtr = arena
		if len(*arena) < *offset+len(input) {
			newArena := make([]float32, (*offset+len(input))*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		saved := (*arena)[*offset : *offset+len(input)]
		copy(saved, input)
		c.savedInputOffsets = append(c.savedInputOffsets, *offset)
		*offset += len(input)
	} else {
		if cap(c.savedInput) < len(input) {
			c.savedInput = make([]float32, len(input))
		}
		c.savedInput = c.savedInput[:len(input)]
		copy(c.savedInput, input)
		c.savedInputOffsets = append(c.savedInputOffsets[:0], 0)
		c.arenaPtr = &c.savedInput
	}

	// Cache dimensions for backward pass
	c.inputHeight = inputHeight
	c.inputWidth = inputWidth

	// Convolution logic
	if md, ok := c.device.(*MetalDevice); ok && c.metalState != nil {
		outH, outW := c.computeOutputSize(inputHeight, inputWidth)
		requiredOutput := c.outChannels * outH * outW

		// Ensure buffers
		if c.bufInput == nil || c.bufInput.length < len(input) {
			if c.bufInput != nil {
				c.bufInput.Free()
			}
			c.bufInput = md.CreateBuffer(input)
		} else {
			c.bufInput.Update(input)
		}

		if c.bufOutput == nil || c.bufOutput.length < requiredOutput {
			if c.bufOutput != nil {
				c.bufOutput.Free()
			}
			c.bufOutput = md.CreateEmptyBuffer(requiredOutput)
		}

		md.Conv2DForward(c.metalState, c.bufInput, c.bufOutput, 1, inputHeight, inputWidth, outH, outW)

		// Check if we need to apply activation (MPS kernel handles convolution but we might need fused activation)
		// Actually, our current createConv2DState doesn't fuse activation yet, so we apply it manually or update createConv2DState.
		// For now, let's read back and apply activation if needed, or better, update the Metal implementation to support activation.
		// Wait, I didn't add activation to the Metal Conv2D kernel. I should do that.

		c.bufOutput.Read(c.outputBuf[:requiredOutput])

		// If activation is ReLU, we can use the fused kernel or just do it on CPU/GPU
		// For simplicity now, let's do it on CPU as before if not fused.
		for i := 0; i < requiredOutput; i++ {
			c.outputBuf[i] = c.activation.Activate(c.outputBuf[i])
		}

		return c.outputBuf[:requiredOutput]
	}

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

func (c *Conv2D) ForwardBatch(input []float32, batchSize int) []float32 {
	return c.ForwardBatchWithArena(input, batchSize, nil, nil)
}

func (c *Conv2D) ForwardBatchWithArena(input []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return c.ForwardWithArena(input, arena, offset)
	}
	inSize := len(input) / batchSize
	outSize := c.OutSize()
	if outSize <= c.outChannels {
		// Heuristic to infer output size if not already known
		c.ForwardWithArena(input[:inSize], nil, nil)
		outSize = c.OutSize()
	}
	if len(c.outputBuf) < batchSize*outSize {
		c.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := c.ForwardWithArena(input[i*inSize:(i+1)*inSize], arena, offset)
		copy(c.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return c.outputBuf[:batchSize*outSize]
}


// Backward performs backpropagation through the convolutional layer.
// grad: gradient of loss w.r.t. activated output (shape: [outChannels, outH, outW] flattened)
// Returns: gradient of loss w.r.t. input
func (c *Conv2D) Backward(grad []float32) []float32 {
	numSaved := len(c.savedInputOffsets)
	if numSaved == 0 {
		return nil
	}

	// Use last saved input
	ts := numSaved - 1
	offset := c.savedInputOffsets[ts]
	inSize := c.inChannels * c.inputHeight * c.inputWidth
	savedInput := (*c.arenaPtr)[offset : offset+inSize]

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

	if md, ok := c.device.(*MetalDevice); ok && c.metalState != nil {
		// Ensure weight gradient buffers exist
		if c.bufGradW == nil {
			c.bufGradW = md.CreateEmptyBuffer(len(c.weights))
		}
		if c.bufGradB == nil {
			c.bufGradB = md.CreateEmptyBuffer(len(c.biases))
		}

		if c.bufGradOut == nil || c.bufGradOut.length < len(grad) {
			if c.bufGradOut != nil {
				c.bufGradOut.Free()
			}
			c.bufGradOut = md.CreateBuffer(grad)
		} else {
			c.bufGradOut.Update(grad)
		}

		if c.bufGradIn == nil || c.bufGradIn.length < len(gradInput) {
			if c.bufGradIn != nil {
				c.bufGradIn.Free()
			}
			c.bufGradIn = md.CreateEmptyBuffer(len(gradInput))
		}

		// We need to handle activation derivative if not fused
		// dL/dz = dL/d(output) * activation'(z)
		// Since we didn't fuse, we do it here
		gradCopy := make([]float32, len(grad))
		copy(gradCopy, grad)
		for i := range gradCopy {
			gradCopy[i] *= c.activation.Derivative(c.preActBuf[i])
		}
		c.bufGradOut.Update(gradCopy)

		md.Conv2DBackward(c.metalState, c.bufInput, c.bufGradOut, c.bufGradIn, c.bufGradW, c.bufGradB, 1, inputHeight, inputWidth, outH, outW)

		c.bufGradIn.Read(gradInput)

		// Accumulate gradients back to Go
		tempGradW := make([]float32, len(c.gradWeights))
		tempGradB := make([]float32, len(c.gradBiases))
		c.bufGradW.Read(tempGradW)
		c.bufGradB.Read(tempGradB)

		for i := range c.gradWeights {
			c.gradWeights[i] += tempGradW[i]
		}
		for i := range c.gradBiases {
			c.gradBiases[i] += tempGradB[i]
		}

		// Pop the last saved input offset
		c.savedInputOffsets = c.savedInputOffsets[:ts]

		return gradInput
	}

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
									c.gradWeights[weightIdx] += gradAfterAct * savedInput[inputIdx]

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

	// Pop the last saved input offset
	c.savedInputOffsets = c.savedInputOffsets[:ts]

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
	c.syncGPU()
}

func (c *Conv2D) syncGPU() {
	if md, ok := c.device.(*MetalDevice); ok && c.metalState != nil {
		md.Conv2DReloadWeights(c.metalState)
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

func (c *Conv2D) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return c.Backward(grad)
	}
	inSize := c.InSize()
	outSize := c.OutSize()
	if len(c.gradInBuf) < batchSize*inSize {
		c.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		dx := c.Backward(grad[i*outSize : (i+1)*outSize])
		copy(c.gradInBuf[i*inSize:(i+1)*inSize], dx)
	}
	return c.gradInBuf[:batchSize*inSize]
}

func (c *Conv2D) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return c.BackwardBatch(grad, batchSize)
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
		inChannels:        c.inChannels,
		outChannels:       c.outChannels,
		kernelSize:        c.kernelSize,
		stride:            c.stride,
		padding:           c.padding,
		setInputHeight:    c.setInputHeight,
		setInputWidth:     c.setInputWidth,
		activation:        c.activation,
		params:            params,
		weights:           params[:weightSize],
		biases:            params[weightSize:],
		grads:             grads,
		gradWeights:       grads[:weightSize],
		gradBiases:        grads[weightSize:],
		outputBuf:         make([]float32, 0),
		gradInBuf:         make([]float32, 0),
		savedInput:        make([]float32, 0),
		savedInputOffsets: make([]int, 0, 128),
		training:          c.training,
		device:            c.device,
		metalState:       c.metalState, // Shared state as it points to shared weights
	}

	if md, ok := c.device.(*MetalDevice); ok && md.IsAvailable() && len(c.params) > 0 {
		if newC.metalState == nil {
			newC.metalState = md.CreateConv2DState(c.inChannels, c.outChannels, c.kernelSize, c.stride, c.padding, newC.weights, newC.biases)
		}
		newC.bufGradW = md.CreateEmptyBuffer(len(newC.weights))
		newC.bufGradB = md.CreateEmptyBuffer(len(newC.biases))
		newC.bufInput = md.CreateEmptyBuffer(c.inChannels * c.kernelSize * c.kernelSize)
		newC.bufOutput = md.CreateEmptyBuffer(c.outChannels)
		newC.bufGradOut = md.CreateEmptyBuffer(c.outChannels)
		newC.bufGradIn = md.CreateEmptyBuffer(c.inChannels * c.kernelSize * c.kernelSize)
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
	c.savedInputOffsets = c.savedInputOffsets[:0]
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
