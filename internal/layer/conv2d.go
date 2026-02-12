// Package layer provides neural network layer implementations.
package layer

import (
	"math"
	"math/rand"

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

	// Input dimensions (set during forward)
	inputHeight int
	inputWidth  int

	// Weights: [outChannels, inChannels, kernelSize, kernelSize]
	// Stored as contiguous slice for cache efficiency
	weights []float64
	biases  []float64

	// Activation function
	activation activations.Activation

	// Pre-allocated buffers
	preActBuf   []float64 // Contains pre-activation values (z = w*x + b)
	outputBuf   []float64 // Contains post-activation values (activation(z))
	gradWeights []float64
	gradBiases  []float64

	// Saved input for backward pass
	savedInput []float64
}

// NewConv2D creates a new 2D convolutional layer.
// inChannels: number of input channels
// outChannels: number of output feature maps
// kernelSize: size of convolutional kernel (square)
// stride: stride for convolution
// padding: zero padding size
func NewConv2D(inChannels, outChannels, kernelSize, stride, padding int,
	activation activations.Activation) *Conv2D {

	// He initialization (better for ReLU)
	scale := math.Sqrt(2.0 / float64(inChannels*kernelSize*kernelSize))

	// Allocate weights: [outChannels, inChannels, kernelSize, kernelSize]
	// Flattened: [outChannels * inChannels * kernelSize * kernelSize]
	weights := make([]float64, outChannels*inChannels*kernelSize*kernelSize)
	biases := make([]float64, outChannels)

	// Initialize weights
	for o := 0; o < outChannels; o++ {
		for c := 0; c < inChannels; c++ {
			for kh := 0; kh < kernelSize; kh++ {
				for kw := 0; kw < kernelSize; kw++ {
					idx := o*inChannels*kernelSize*kernelSize +
						c*kernelSize*kernelSize +
						kh*kernelSize + kw
					weights[idx] = rand.Float64()*2*scale - scale
				}
			}
		}
		// Initialize bias
		biases[o] = rand.Float64()*0.2 - 0.1
	}

	return &Conv2D{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelSize:  kernelSize,
		stride:      stride,
		padding:     padding,
		activation:  activation,

		weights:     weights,
		biases:      biases,
		gradWeights: make([]float64, len(weights)),
		gradBiases:  make([]float64, len(biases)),
	}
}

// computeOutputSize calculates the output spatial dimensions
func (c *Conv2D) computeOutputSize(inputHeight, inputWidth int) (int, int) {
	// Output size: (input + 2*padding - kernel) / stride + 1
	outH := (inputHeight + 2*c.padding - c.kernelSize) / c.stride + 1
	outW := (inputWidth + 2*c.padding - c.kernelSize) / c.stride + 1
	return outH, outW
}

// Forward performs a forward pass through the convolutional layer.
// input: flattened [inChannels, inputHeight, inputWidth]
// Returns: flattened [outChannels, outputHeight, outputWidth]
func (c *Conv2D) Forward(input []float64) []float64 {
	// Infer input dimensions from length
	totalInput := len(input)
	if totalInput%c.inChannels != 0 {
		panic("Conv2D: input length not divisible by inChannels")
	}
	channelSize := totalInput / c.inChannels
	// Assume square input for now
	c.inputHeight = int(math.Sqrt(float64(channelSize)))
	c.inputWidth = c.inputHeight

	// Compute output dimensions
	outH, outW := c.computeOutputSize(c.inputHeight, c.inputWidth)

	// Ensure buffers are sized correctly
	requiredOutput := c.outChannels * outH * outW
	if len(c.preActBuf) < requiredOutput {
		c.preActBuf = make([]float64, requiredOutput)
	}
	if len(c.outputBuf) < requiredOutput {
		c.outputBuf = make([]float64, requiredOutput)
	}

	// Save input for backward pass
	if cap(c.savedInput) < len(input) {
		c.savedInput = make([]float64, len(input))
	}
	copy(c.savedInput, input)

	// Convolution: for each output position and channel
	// Input layout: [inChannels, inputHeight, inputWidth]
	// Output layout: [outChannels, outH, outW]
	outSize := outH * outW

	for oc := 0; oc < c.outChannels; oc++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				// Compute convolution at this position
				sum := c.biases[oc]

				for ic := 0; ic < c.inChannels; ic++ {
					for kh := 0; kh < c.kernelSize; kh++ {
						for kw := 0; kw < c.kernelSize; kw++ {
							// Calculate input position with padding
							inH := oh*c.stride + kh - c.padding
							inW := ow*c.stride + kw - c.padding

							var val float64
							if inH >= 0 && inH < c.inputHeight && inW >= 0 && inW < c.inputWidth {
								idx := ic*c.inputHeight*c.inputWidth + inH*c.inputWidth + inW
								val = input[idx]
							}

							weightIdx := oc*c.inChannels*c.kernelSize*c.kernelSize +
								ic*c.kernelSize*c.kernelSize +
								kh*c.kernelSize + kw
							sum += val * c.weights[weightIdx]
						}
					}
				}

				// Store pre-activation
				pos := oc*outSize + oh*outW + ow
				c.preActBuf[pos] = sum

				// Apply activation
				c.outputBuf[pos] = c.activation.Activate(sum)
			}
		}
	}

	return c.outputBuf[:requiredOutput]
}

// Backward performs backpropagation through the convolutional layer.
// grad: gradient of loss w.r.t. activated output (shape: [outChannels, outH, outW] flattened)
// Returns: gradient of loss w.r.t. input
func (c *Conv2D) Backward(grad []float64) []float64 {
	// Output dimensions
	outH, outW := c.computeOutputSize(c.inputHeight, c.inputWidth)
	outSize := outH * outW

	kernelSize := c.kernelSize
	inChannels := c.inChannels
	outChannels := c.outChannels

	// Clear gradient buffers
	for i := range c.gradWeights {
		c.gradWeights[i] = 0
	}
	for i := range c.gradBiases {
		c.gradBiases[i] = 0
	}

	// Input gradient buffer
	gradInput := make([]float64, inChannels*c.inputHeight*c.inputWidth)

	// For each output position and channel
	for oc := 0; oc < outChannels; oc++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				pos := oc*outSize + oh*outW + ow

				// Gradient after activation: dL/dz = dL/d(output) * activation'(z)
				gradAfterAct := grad[pos] * c.activation.Derivative(c.preActBuf[pos])

				// Accumulate gradient for bias
				c.gradBiases[oc] += gradAfterAct

				// For each input channel and kernel position
				for ic := 0; ic < inChannels; ic++ {
					for kh := 0; kh < kernelSize; kh++ {
						for kw := 0; kw < kernelSize; kw++ {
							// Calculate input position
							inH := oh*c.stride + kh - c.padding
							inW := ow*c.stride + kw - c.padding

							var inputVal float64
							if inH >= 0 && inH < c.inputHeight && inW >= 0 && inW < c.inputWidth {
								inputIdx := ic*c.inputHeight*c.inputWidth + inH*c.inputWidth + inW
								inputVal = c.savedInput[inputIdx]
							}

							// Weight index
							weightIdx := oc*inChannels*kernelSize*kernelSize +
								ic*kernelSize*kernelSize +
								kh*kernelSize + kw

							// Accumulate weight gradient
							c.gradWeights[weightIdx] += gradAfterAct * inputVal

							// Accumulate input gradient
							if inH >= 0 && inH < c.inputHeight && inW >= 0 && inW < c.inputWidth {
								inputGradIdx := ic*c.inputHeight*c.inputWidth + inH*c.inputWidth + inW
								gradInput[inputGradIdx] += gradAfterAct * c.weights[weightIdx]
							}
						}
					}
				}
			}
		}
	}

	return gradInput
}

// Params returns all convolutional layer parameters flattened.
func (c *Conv2D) Params() []float64 {
	total := len(c.weights) + len(c.biases)
	params := make([]float64, 0, total)
	params = append(params, c.weights...)
	params = append(params, c.biases...)
	return params
}

// SetParams updates weights and biases from a flattened slice.
func (c *Conv2D) SetParams(params []float64) {
	totalWeights := len(c.weights)
	copy(c.weights, params[:totalWeights])
	copy(c.biases, params[totalWeights:])
}

// Gradients returns all convolutional layer gradients flattened.
func (c *Conv2D) Gradients() []float64 {
	total := len(c.gradWeights) + len(c.gradBiases)
	gradients := make([]float64, 0, total)
	gradients = append(gradients, c.gradWeights...)
	gradients = append(gradients, c.gradBiases...)
	return gradients
}

// InSize returns the number of input channels.
func (c *Conv2D) InSize() int {
	return c.inChannels
}

// OutSize returns the number of output channels.
func (c *Conv2D) OutSize() int {
	return c.outChannels
}

// Reset clears any cached state (for compatibility with other layer types).
func (c *Conv2D) Reset() {
	// No state to reset for Conv2D
}

// OutputBuf returns the raw output buffer for advanced use.
// The caller should not modify this buffer directly.
func (c *Conv2D) OutputBuf() []float64 {
	return c.outputBuf
}

// SetOutputBuf allows setting a custom output buffer.
// This is useful for chaining layers efficiently.
func (c *Conv2D) SetOutputBuf(buf []float64) {
	c.outputBuf = buf
}
