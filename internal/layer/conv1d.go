// Package layer provides Conv1D implementation for temporal convolution
// Conv1D is commonly used for time series analysis and 1D signal processing.
//
// Architecture:
// - Input:  [batchSize, channels, length] (NCL format)
// - Kernel: [outChannels, inChannels, kernelSize]
// - Output: [batchSize, outChannels, outputLength]
//
// Output length calculation:
// - With padding 'same': outputLength = inputLength
// - With padding 'valid': outputLength = inputLength - kernelSize + 1
// - With custom padding: outputLength = (inputLength + 2*padding - kernelSize) / stride + 1
package layer

import (
	"math"
	"sync"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

// Conv1D performs 1D convolution over input signals
type Conv1D struct {
	// Parameters
	inChannels  int
	outChannels int
	kernelSize  int
	stride      int
	padding     int

	// Weights and biases
	weights []float32 // Shape: [outChannels, inChannels, kernelSize]
	biases  []float32 // Shape: [outChannels]

	// Gradients
	gradWeights []float32
	gradBiases  []float32

	// Buffers
	inputBuf    []float32 // Cached input for backward
	outputBuf   []float32
	preActBuf   []float32 // Pre-activation values

	// Activation function
	act activations.Activation

	// Training mode
	training bool

	// Arena for memory management
	arenaPtr          *[]float32
	savedInputOffsets []int
	arenaMu           sync.Mutex // Protege operações de arena contra race conditions
}

// NewConv1D creates a new Conv1D layer
// inChannels: number of input channels
// outChannels: number of output channels (filters)
// kernelSize: size of the convolutional kernel
// stride: stride of the convolution
// padding: zero-padding added to both sides of input
// act: activation function
// Returns nil if parameters are invalid.
func NewConv1D(inChannels, outChannels, kernelSize, stride, padding int, act activations.Activation) *Conv1D {
	if inChannels <= 0 {
		return nil
	}
	if outChannels <= 0 {
		return nil
	}
	if kernelSize <= 0 {
		return nil
	}
	if stride <= 0 {
		return nil
	}
	if padding < 0 {
		return nil
	}

	c := &Conv1D{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelSize:  kernelSize,
		stride:      stride,
		padding:     padding,
		act:         act,
		training:    true,
	}

	// Initialize weights (Glorot/Xavier initialization)
	fanIn := inChannels * kernelSize
	fanOut := outChannels * kernelSize
	scale := float32(math.Sqrt(2.0 / float64(fanIn+fanOut)))

	c.weights = make([]float32, outChannels*inChannels*kernelSize)
	c.gradWeights = make([]float32, len(c.weights))

	rng := NewRNG(42)
	for i := range c.weights {
		c.weights[i] = (rng.RandFloat()*2 - 1) * scale
	}

	// Initialize biases to small values
	c.biases = make([]float32, outChannels)
	c.gradBiases = make([]float32, outChannels)
	for i := range c.biases {
		c.biases[i] = (rng.RandFloat()*2 - 1) * 0.1
	}

	return c
}

// OutputLength calculates output length for given input length
func (c *Conv1D) OutputLength(inputLength int) int {
	return (inputLength + 2*c.padding - c.kernelSize) / c.stride + 1
}

// Forward performs forward pass
// Input shape: [batchSize, inChannels, inputLength]
// Output shape: [batchSize, outChannels, outputLength]
func (c *Conv1D) Forward(x []float32) ([]float32, error) {
	return c.ForwardWithArena(x, nil, nil)
}

// ForwardWithArena performs forward pass with arena allocation
func (c *Conv1D) ForwardWithArena(x []float32, arena *[]float32, offset *int) ([]float32, error) {
	// Assuming batch size 1 for simplicity
	// Full implementation would handle batches
	batchSize := 1
	inputLength := len(x) / (batchSize * c.inChannels)
	outputLength := c.OutputLength(inputLength)

	// Ensure output buffer
	outputSize := batchSize * c.outChannels * outputLength
	if len(c.outputBuf) < outputSize {
		c.outputBuf = make([]float32, outputSize)
	}

	// Cache input for backward pass - protegido por mutex quando usando arena
	if c.training {
		if arena != nil && offset != nil {
			// Use arena for memory management
			c.arenaMu.Lock()
			c.arenaPtr = arena
			inSize := len(x)
			if len(*arena) < *offset+inSize {
				newArena := make([]float32, (*offset+inSize)*2)
				copy(newArena, *arena)
				*arena = newArena
			}
			saved := (*arena)[*offset : *offset+inSize]
			copy(saved, x)
			c.savedInputOffsets = append(c.savedInputOffsets, *offset)
			*offset += inSize
			c.arenaMu.Unlock()
		} else {
			c.inputBuf = make([]float32, len(x))
			copy(c.inputBuf, x)
		}
	}

	// Perform convolution
	c.conv1DForward(x, c.outputBuf[:outputSize], batchSize, inputLength, outputLength)

	// Apply activation
	if c.act != nil {
		for i := range c.outputBuf[:outputSize] {
			c.outputBuf[i] = c.act.Activate(c.outputBuf[i])
		}
	}

	return c.outputBuf[:outputSize], nil
}

// conv1DForward performs the core convolution operation
func (c *Conv1D) conv1DForward(input, output []float32, batchSize, inputLength, outputLength int) {
	// Zero the output
	for i := range output {
		output[i] = 0
	}

	// For each batch
	for b := 0; b < batchSize; b++ {
		batchOffset := b * c.inChannels * inputLength

		// For each output channel
		for oc := 0; oc < c.outChannels; oc++ {
			bias := c.biases[oc]
			outOffset := b*c.outChannels*outputLength + oc*outputLength

			// For each output position
			for ol := 0; ol < outputLength; ol++ {
				sum := bias

				// For each input channel
				for ic := 0; ic < c.inChannels; ic++ {
					inOffset := batchOffset + ic*inputLength
					wOffset := oc*c.inChannels*c.kernelSize + ic*c.kernelSize

					// Convolve kernel over input
					for k := 0; k < c.kernelSize; k++ {
						inputPos := ol*c.stride + k - c.padding
						if inputPos >= 0 && inputPos < inputLength {
							sum += input[inOffset+inputPos] * c.weights[wOffset+k]
						}
					}
				}

				output[outOffset+ol] = sum
			}
		}
	}
}

// Backward performs backward pass
func (c *Conv1D) Backward(grad []float32) ([]float32, error) {
	gradIn := c.BackwardInPlace(grad, nil)
	return gradIn, nil
}

// BackwardInPlace performs backward pass with optional gradient accumulation
func (c *Conv1D) BackwardInPlace(grad []float32, gradIn []float32) []float32 {
	batchSize := 1
	outputLength := len(grad) / (batchSize * c.outChannels)

	// Get cached input
	input := c.inputBuf
	if c.arenaPtr != nil && len(c.savedInputOffsets) > 0 {
		offset := c.savedInputOffsets[len(c.savedInputOffsets)-1]
		input = (*c.arenaPtr)[offset:]
	}

	inputLength := len(input) / (batchSize * c.inChannels)

	// Allocate gradient input if needed
	if gradIn == nil {
		gradIn = make([]float32, len(input))
	}

	// Clear gradients
	for i := range gradIn {
		gradIn[i] = 0
	}

	// Compute gradients
	c.conv1DBackward(input, grad, gradIn, batchSize, inputLength, outputLength)

	return gradIn
}

// conv1DBackward computes gradients for weights and input
func (c *Conv1D) conv1DBackward(input, gradOutput, gradInput []float32, batchSize, inputLength, outputLength int) {
	// Accumulate weight gradients
	for b := 0; b < batchSize; b++ {
		batchOffset := b * c.inChannels * inputLength

		for oc := 0; oc < c.outChannels; oc++ {
			outOffset := b*c.outChannels*outputLength + oc*outputLength

			for ol := 0; ol < outputLength; ol++ {
				gradOut := gradOutput[outOffset+ol]

				// Bias gradient
				c.gradBiases[oc] += gradOut

				// Weight and input gradients
				for ic := 0; ic < c.inChannels; ic++ {
					inOffset := batchOffset + ic*inputLength
					wOffset := oc*c.inChannels*c.kernelSize + ic*c.kernelSize

					for k := 0; k < c.kernelSize; k++ {
						inputPos := ol*c.stride + k - c.padding
						if inputPos >= 0 && inputPos < inputLength {
							// Weight gradient: dL/dW += dL/dOut * Input
							c.gradWeights[wOffset+k] += gradOut * input[inOffset+inputPos]

							// Input gradient: dL/dIn += dL/dOut * Weight
							gradInput[inOffset+inputPos] += gradOut * c.weights[wOffset+k]
						}
					}
				}
			}
		}
	}
}

// Params returns the layer parameters (weights and biases)
func (c *Conv1D) Params() []float32 {
	params := make([]float32, 0, len(c.weights)+len(c.biases))
	params = append(params, c.weights...)
	params = append(params, c.biases...)
	return params
}

// SetParams sets the layer parameters
func (c *Conv1D) SetParams(params []float32) {
	wSize := len(c.weights)
	copy(c.weights, params[:wSize])
	copy(c.biases, params[wSize:])
}

// Gradients returns the accumulated gradients
func (c *Conv1D) Gradients() []float32 {
	grads := make([]float32, 0, len(c.gradWeights)+len(c.gradBiases))
	grads = append(grads, c.gradWeights...)
	grads = append(grads, c.gradBiases...)
	return grads
}

// ClearGradients clears the accumulated gradients
func (c *Conv1D) ClearGradients() {
	for i := range c.gradWeights {
		c.gradWeights[i] = 0
	}
	for i := range c.gradBiases {
		c.gradBiases[i] = 0
	}
}

// AccumulateBackward accumulates gradients without clearing previous values
func (c *Conv1D) AccumulateBackward(grad []float32) ([]float32, error) {
	gradIn := c.BackwardInPlace(grad, nil)
	return gradIn, nil
}

// SetTraining sets training mode
func (c *Conv1D) SetTraining(training bool) {
	c.training = training
}

// OutputSize returns output size for given input size
func (c *Conv1D) OutputSize(inputSize int) int {
	return c.OutputLength(inputSize)
}

// InChannels returns number of input channels
func (c *Conv1D) InChannels() int { return c.inChannels }

// OutChannels returns number of output channels
func (c *Conv1D) OutChannels() int { return c.outChannels }

// KernelSize returns kernel size
func (c *Conv1D) KernelSize() int { return c.kernelSize }

// Stride returns stride
func (c *Conv1D) Stride() int { return c.stride }

// Padding returns padding
func (c *Conv1D) Padding() int { return c.padding }

// InSize returns the input size (inChannels).
// Note: For Conv1D, this is the number of input channels, not the sequence length.
func (c *Conv1D) InSize() int { return c.inChannels }

// OutSize returns the output size (outChannels).
// Note: For Conv1D, this is the number of output channels, not the sequence length.
func (c *Conv1D) OutSize() int { return c.outChannels }

// ArenaSize returns the number of float32 values needed in the activation arena.
// Conv1D saves input for backward pass. Size depends on batch and sequence length.
// Returns 0 as dimensions are dynamic per forward pass.
func (c *Conv1D) ArenaSize() int {
	return 0 // Dynamic based on input sequence length
}
