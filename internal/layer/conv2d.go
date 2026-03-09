// Package layer provides neural network layer implementations.
package layer

import (
	"fmt"
	"math"
	"sync"
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
	scratchBuf  []float32 // Pre-allocated for temporary operations (e.g., grad * activation')

	// Saved input for backward pass
	savedInput        []float32
	savedInputOffsets []int
	arenaPtr          *[]float32
	arenaMu           sync.Mutex // Protege operações de arena contra race conditions

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

	// Buffer pooling for Metal - stores max allocated sizes to avoid reallocation
	maxInputSize   int
	maxOutputSize  int
	maxGradOutSize int
	maxGradInSize  int
}

// NewConv2D creates a new 2D convolutional layer.
// Returns an error if parameters are invalid.
func NewConv2D(inChannels, outChannels, kernelSize, stride, padding int,
	activation activations.Activation) (*Conv2D, error) {

	if inChannels != -1 && inChannels <= 0 {
		return nil, fmt.Errorf("invalid inChannels %d: must be > 0 or -1", inChannels)
	}
	if outChannels <= 0 && outChannels != -1 {
		return nil, fmt.Errorf("invalid outChannels %d: must be > 0 or -1", outChannels)
	}
	if kernelSize <= 0 {
		return nil, fmt.Errorf("invalid kernelSize %d: must be > 0", kernelSize)
	}
	if stride <= 0 {
		return nil, fmt.Errorf("invalid stride %d: must be > 0", stride)
	}
	if padding < 0 {
		return nil, fmt.Errorf("invalid padding %d: must be >= 0", padding)
	}

	c := &Conv2D{
		inChannels:        inChannels,
		outChannels:       outChannels,
		kernelSize:        kernelSize,
		stride:            stride,
		padding:           padding,
		activation:        activation,
		device:            &CPUDevice{},
		savedInputOffsets: make([]int, 0, 16),
	}

	if inChannels != -1 {
		c.Build(inChannels)
	}

	return c, nil
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
	c.scratchBuf = make([]float32, 0)

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

// Close releases all GPU buffers and resources held by the layer.
// This method is idempotent and safe to call multiple times.
// SEC-009: Implementar cleanup explícito de buffers Metal para prevenir memory leaks.
func (c *Conv2D) Close() {
	// Liberar buffers GPU em ordem inversa de criação
	if c.bufGradIn != nil {
		c.bufGradIn.Close()
		c.bufGradIn = nil
	}
	if c.bufGradOut != nil {
		c.bufGradOut.Close()
		c.bufGradOut = nil
	}
	if c.bufOutput != nil {
		c.bufOutput.Close()
		c.bufOutput = nil
	}
	if c.bufInput != nil {
		c.bufInput.Close()
		c.bufInput = nil
	}
	if c.bufGradB != nil {
		c.bufGradB.Close()
		c.bufGradB = nil
	}
	if c.bufGradW != nil {
		c.bufGradW.Close()
		c.bufGradW = nil
	}
	// Liberar estado Conv2D do Metal
	if c.metalState != nil {
		if md, ok := c.device.(*MetalDevice); ok && md != nil {
			md.FreeConv2DState(c.metalState)
		}
		c.metalState = nil
	}
	// Reset device state
	c.device = &CPUDevice{}
}

// computeOutputSize calculates the output spatial dimensions
func (c *Conv2D) computeOutputSize(inputHeight, inputWidth int) (int, int) {
	// Output size: (input + 2*padding - kernel) / stride + 1
	outH := (inputHeight+2*c.padding-c.kernelSize)/c.stride + 1
	outW := (inputWidth+2*c.padding-c.kernelSize)/c.stride + 1
	return outH, outW
}

// conv2D3x3Optimized implements an optimized 3x3 convolution with stride=1, padding=1
// This is a common configuration used in many CNN architectures
func (c *Conv2D) conv2D3x3Optimized(input []float32, inputHeight, inputWidth, outH, outW int) {
	outSize := outH * outW
	inChannels := c.inChannels
	outChannels := c.outChannels

	// Kernel weights are stored as: [outChannels][inChannels][3][3]
	// Pre-fetch weights into local arrays for better cache locality

	// Process output channels in groups for better cache utilization
	const tileSize = 8

	for oc := 0; oc < outChannels; oc++ {
		ocWeightBase := oc * inChannels * 9 // 9 = 3*3
		ocOutBase := oc * outSize

		// Clear output buffer for this channel
		for i := 0; i < outSize; i++ {
			c.preActBuf[ocOutBase+i] = 0
		}

		// Process input channels
		for ic := 0; ic < inChannels; ic++ {
			// Pre-fetch the 3x3 kernel weights for this input/output channel pair
			wBase := ocWeightBase + ic*9
			w00 := c.weights[wBase+0]
			w01 := c.weights[wBase+1]
			w02 := c.weights[wBase+2]
			w10 := c.weights[wBase+3]
			w11 := c.weights[wBase+4]
			w12 := c.weights[wBase+5]
			w20 := c.weights[wBase+6]
			w21 := c.weights[wBase+7]
			w22 := c.weights[wBase+8]

			icOffset := ic * inputHeight * inputWidth

			// Process output positions with loop unrolling and vector-friendly access
			// Skip boundary checks by processing only valid interior region
			// Handle boundaries separately

			// Process interior region (no boundary checks needed)
			for oh := 1; oh < outH-1; oh++ {
				outRowOffset := ocOutBase + oh*outW
				inRow := oh
				rowOffsetTop := icOffset + (inRow-1)*inputWidth
				rowOffsetMid := icOffset + inRow*inputWidth
				rowOffsetBot := icOffset + (inRow+1)*inputWidth

				// Process in chunks of 4 for better instruction pipelining
				ow := 1
				for ; ow <= outW-5; ow += 4 {
					pos0 := outRowOffset + ow
					pos1 := pos0 + 1
					pos2 := pos0 + 2
					pos3 := pos0 + 3
					inCol0 := ow
					inCol1 := ow + 1
					inCol2 := ow + 2
					inCol3 := ow + 3

					// Top row contributions
					c.preActBuf[pos0] += w00*input[rowOffsetTop+inCol0-1] + w01*input[rowOffsetTop+inCol0] + w02*input[rowOffsetTop+inCol0+1]
					c.preActBuf[pos1] += w00*input[rowOffsetTop+inCol1-1] + w01*input[rowOffsetTop+inCol1] + w02*input[rowOffsetTop+inCol1+1]
					c.preActBuf[pos2] += w00*input[rowOffsetTop+inCol2-1] + w01*input[rowOffsetTop+inCol2] + w02*input[rowOffsetTop+inCol2+1]
					c.preActBuf[pos3] += w00*input[rowOffsetTop+inCol3-1] + w01*input[rowOffsetTop+inCol3] + w02*input[rowOffsetTop+inCol3+1]

					// Middle row contributions
					c.preActBuf[pos0] += w10*input[rowOffsetMid+inCol0-1] + w11*input[rowOffsetMid+inCol0] + w12*input[rowOffsetMid+inCol0+1]
					c.preActBuf[pos1] += w10*input[rowOffsetMid+inCol1-1] + w11*input[rowOffsetMid+inCol1] + w12*input[rowOffsetMid+inCol1+1]
					c.preActBuf[pos2] += w10*input[rowOffsetMid+inCol2-1] + w11*input[rowOffsetMid+inCol2] + w12*input[rowOffsetMid+inCol2+1]
					c.preActBuf[pos3] += w10*input[rowOffsetMid+inCol3-1] + w11*input[rowOffsetMid+inCol3] + w12*input[rowOffsetMid+inCol3+1]

					// Bottom row contributions
					c.preActBuf[pos0] += w20*input[rowOffsetBot+inCol0-1] + w21*input[rowOffsetBot+inCol0] + w22*input[rowOffsetBot+inCol0+1]
					c.preActBuf[pos1] += w20*input[rowOffsetBot+inCol1-1] + w21*input[rowOffsetBot+inCol1] + w22*input[rowOffsetBot+inCol1+1]
					c.preActBuf[pos2] += w20*input[rowOffsetBot+inCol2-1] + w21*input[rowOffsetBot+inCol2] + w22*input[rowOffsetBot+inCol2+1]
					c.preActBuf[pos3] += w20*input[rowOffsetBot+inCol3-1] + w21*input[rowOffsetBot+inCol3] + w22*input[rowOffsetBot+inCol3+1]
				}

				// Handle remaining elements
				for ; ow < outW-1; ow++ {
					pos := outRowOffset + ow
					inCol := ow

					c.preActBuf[pos] += w00*input[rowOffsetTop+inCol-1] + w01*input[rowOffsetTop+inCol] + w02*input[rowOffsetTop+inCol+1]
					c.preActBuf[pos] += w10*input[rowOffsetMid+inCol-1] + w11*input[rowOffsetMid+inCol] + w12*input[rowOffsetMid+inCol+1]
					c.preActBuf[pos] += w20*input[rowOffsetBot+inCol-1] + w21*input[rowOffsetBot+inCol] + w22*input[rowOffsetBot+inCol+1]
				}
			}

			// Handle first and last rows (boundaries)
			for oh := 0; oh < outH; oh += outH - 1 {
				if oh == 0 || oh == outH-1 {
					outRowOffset := ocOutBase + oh*outW
					inRow := oh
					for ow := 0; ow < outW; ow++ {
						inCol := ow
						pos := outRowOffset + ow

						if inRow > 0 {
							rowOffset := icOffset + (inRow-1)*inputWidth
							if inCol > 0 {
								c.preActBuf[pos] += w00 * input[rowOffset+inCol-1]
							}
							c.preActBuf[pos] += w01 * input[rowOffset+inCol]
							if inCol < inputWidth-1 {
								c.preActBuf[pos] += w02 * input[rowOffset+inCol+1]
							}
						}

						{
							rowOffset := icOffset + inRow*inputWidth
							if inCol > 0 {
								c.preActBuf[pos] += w10 * input[rowOffset+inCol-1]
							}
							c.preActBuf[pos] += w11 * input[rowOffset+inCol]
							if inCol < inputWidth-1 {
								c.preActBuf[pos] += w12 * input[rowOffset+inCol+1]
							}
						}

						if inRow < inputHeight-1 {
							rowOffset := icOffset + (inRow+1)*inputWidth
							if inCol > 0 {
								c.preActBuf[pos] += w20 * input[rowOffset+inCol-1]
							}
							c.preActBuf[pos] += w21 * input[rowOffset+inCol]
							if inCol < inputWidth-1 {
								c.preActBuf[pos] += w22 * input[rowOffset+inCol+1]
							}
						}
					}
				}
			}
		}
	}
}

// conv2D5x5Optimized implements an optimized 5x5 convolution with stride=1, padding=2
// Common in early CNN layers and some modern architectures
func (c *Conv2D) conv2D5x5Optimized(input []float32, inputHeight, inputWidth, outH, outW int) {
	outSize := outH * outW
	inChannels := c.inChannels
	outChannels := c.outChannels

	// Kernel weights are stored as: [outChannels][inChannels][5][5]
	// Pre-fetch weights into local arrays for better cache locality

	for oc := 0; oc < outChannels; oc++ {
		ocWeightBase := oc * inChannels * 25 // 25 = 5*5
		ocOutBase := oc * outSize

		// Clear output buffer for this channel
		for i := 0; i < outSize; i++ {
			c.preActBuf[ocOutBase+i] = 0
		}

		// Process input channels
		for ic := 0; ic < inChannels; ic++ {
			// Pre-fetch the 5x5 kernel weights for this input/output channel pair
			wBase := ocWeightBase + ic*25
			var w [25]float32
			for i := 0; i < 25; i++ {
				w[i] = c.weights[wBase+i]
			}

			icOffset := ic * inputHeight * inputWidth

			// Process output positions with loop unrolling
			// Skip boundary checks by processing only valid interior region
			// Handle boundaries separately

			// Process interior region (no boundary checks needed for kernel positions)
			for oh := 2; oh < outH-2; oh++ {
				outRowOffset := ocOutBase + oh*outW
				inRow := oh

				// Pre-compute row offsets for the 5 rows
				rowOffset0 := icOffset + (inRow-2)*inputWidth
				rowOffset1 := icOffset + (inRow-1)*inputWidth
				rowOffset2 := icOffset + inRow*inputWidth
				rowOffset3 := icOffset + (inRow+1)*inputWidth
				rowOffset4 := icOffset + (inRow+2)*inputWidth

				// Process in chunks of 2 for better instruction pipelining
				ow := 2
				for ; ow <= outW-3; ow += 2 {
					pos0 := outRowOffset + ow
					pos1 := pos0 + 1
					inCol0 := ow
					inCol1 := ow + 1

					// Row -2 contributions
					c.preActBuf[pos0] += w[0]*input[rowOffset0+inCol0-2] + w[1]*input[rowOffset0+inCol0-1] + w[2]*input[rowOffset0+inCol0] + w[3]*input[rowOffset0+inCol0+1] + w[4]*input[rowOffset0+inCol0+2]
					c.preActBuf[pos1] += w[0]*input[rowOffset0+inCol1-2] + w[1]*input[rowOffset0+inCol1-1] + w[2]*input[rowOffset0+inCol1] + w[3]*input[rowOffset0+inCol1+1] + w[4]*input[rowOffset0+inCol1+2]

					// Row -1 contributions
					c.preActBuf[pos0] += w[5]*input[rowOffset1+inCol0-2] + w[6]*input[rowOffset1+inCol0-1] + w[7]*input[rowOffset1+inCol0] + w[8]*input[rowOffset1+inCol0+1] + w[9]*input[rowOffset1+inCol0+2]
					c.preActBuf[pos1] += w[5]*input[rowOffset1+inCol1-2] + w[6]*input[rowOffset1+inCol1-1] + w[7]*input[rowOffset1+inCol1] + w[8]*input[rowOffset1+inCol1+1] + w[9]*input[rowOffset1+inCol1+2]

					// Row 0 contributions
					c.preActBuf[pos0] += w[10]*input[rowOffset2+inCol0-2] + w[11]*input[rowOffset2+inCol0-1] + w[12]*input[rowOffset2+inCol0] + w[13]*input[rowOffset2+inCol0+1] + w[14]*input[rowOffset2+inCol0+2]
					c.preActBuf[pos1] += w[10]*input[rowOffset2+inCol1-2] + w[11]*input[rowOffset2+inCol1-1] + w[12]*input[rowOffset2+inCol1] + w[13]*input[rowOffset2+inCol1+1] + w[14]*input[rowOffset2+inCol1+2]

					// Row +1 contributions
					c.preActBuf[pos0] += w[15]*input[rowOffset3+inCol0-2] + w[16]*input[rowOffset3+inCol0-1] + w[17]*input[rowOffset3+inCol0] + w[18]*input[rowOffset3+inCol0+1] + w[19]*input[rowOffset3+inCol0+2]
					c.preActBuf[pos1] += w[15]*input[rowOffset3+inCol1-2] + w[16]*input[rowOffset3+inCol1-1] + w[17]*input[rowOffset3+inCol1] + w[18]*input[rowOffset3+inCol1+1] + w[19]*input[rowOffset3+inCol1+2]

					// Row +2 contributions
					c.preActBuf[pos0] += w[20]*input[rowOffset4+inCol0-2] + w[21]*input[rowOffset4+inCol0-1] + w[22]*input[rowOffset4+inCol0] + w[23]*input[rowOffset4+inCol0+1] + w[24]*input[rowOffset4+inCol0+2]
					c.preActBuf[pos1] += w[20]*input[rowOffset4+inCol1-2] + w[21]*input[rowOffset4+inCol1-1] + w[22]*input[rowOffset4+inCol1] + w[23]*input[rowOffset4+inCol1+1] + w[24]*input[rowOffset4+inCol1+2]
				}

				// Handle remaining elements
				for ; ow < outW-2; ow++ {
					pos := outRowOffset + ow
					inCol := ow

					// Accumulate all 25 kernel positions
					c.preActBuf[pos] += w[0]*input[rowOffset0+inCol-2] + w[1]*input[rowOffset0+inCol-1] + w[2]*input[rowOffset0+inCol] + w[3]*input[rowOffset0+inCol+1] + w[4]*input[rowOffset0+inCol+2]
					c.preActBuf[pos] += w[5]*input[rowOffset1+inCol-2] + w[6]*input[rowOffset1+inCol-1] + w[7]*input[rowOffset1+inCol] + w[8]*input[rowOffset1+inCol+1] + w[9]*input[rowOffset1+inCol+2]
					c.preActBuf[pos] += w[10]*input[rowOffset2+inCol-2] + w[11]*input[rowOffset2+inCol-1] + w[12]*input[rowOffset2+inCol] + w[13]*input[rowOffset2+inCol+1] + w[14]*input[rowOffset2+inCol+2]
					c.preActBuf[pos] += w[15]*input[rowOffset3+inCol-2] + w[16]*input[rowOffset3+inCol-1] + w[17]*input[rowOffset3+inCol] + w[18]*input[rowOffset3+inCol+1] + w[19]*input[rowOffset3+inCol+2]
					c.preActBuf[pos] += w[20]*input[rowOffset4+inCol-2] + w[21]*input[rowOffset4+inCol-1] + w[22]*input[rowOffset4+inCol] + w[23]*input[rowOffset4+inCol+1] + w[24]*input[rowOffset4+inCol+2]
				}
			}

			// Handle boundary regions with bounds checking
			// Only process first 2 rows, last 2 rows, first 2 columns, and last 2 columns
			// Top and bottom rows
			for oh := 0; oh < 2 || oh >= outH-2; oh++ {
				if oh >= outH {
					break
				}
				outRowOffset := ocOutBase + oh*outW
				for ow := 0; ow < outW; ow++ {
					pos := outRowOffset + ow

					for kh := 0; kh < 5; kh++ {
						inH := oh + kh - 2
						if inH >= 0 && inH < inputHeight {
							rowOffset := icOffset + inH*inputWidth
							for kw := 0; kw < 5; kw++ {
								inW := ow + kw - 2
								if inW >= 0 && inW < inputWidth {
									wIdx := kh*5 + kw
									c.preActBuf[pos] += w[wIdx] * input[rowOffset+inW]
								}
							}
						}
					}
				}
				if oh >= 1 && oh < outH-2 {
					// After processing oh=0 and oh=1, skip to outH-2
					oh = outH - 3
				}
			}
			// Left and right columns (excluding corners already processed)
			for oh := 2; oh < outH-2; oh++ {
				outRowOffset := ocOutBase + oh*outW
				// Process first 2 columns
				for ow := 0; ow < 2; ow++ {
					pos := outRowOffset + ow
					for kh := 0; kh < 5; kh++ {
						inH := oh + kh - 2
						if inH >= 0 && inH < inputHeight {
							rowOffset := icOffset + inH*inputWidth
							for kw := 0; kw < 5; kw++ {
								inW := ow + kw - 2
								if inW >= 0 && inW < inputWidth {
									wIdx := kh*5 + kw
									c.preActBuf[pos] += w[wIdx] * input[rowOffset+inW]
								}
							}
						}
					}
				}
				// Process last 2 columns
				for ow := outW - 2; ow < outW; ow++ {
					pos := outRowOffset + ow
					for kh := 0; kh < 5; kh++ {
						inH := oh + kh - 2
						if inH >= 0 && inH < inputHeight {
							rowOffset := icOffset + inH*inputWidth
							for kw := 0; kw < 5; kw++ {
								inW := ow + kw - 2
								if inW >= 0 && inW < inputWidth {
									wIdx := kh*5 + kw
									c.preActBuf[pos] += w[wIdx] * input[rowOffset+inW]
								}
							}
						}
					}
				}
			}
		}
	}
}

// conv2D7x7Optimized implements an optimized 7x7 convolution with stride=1, padding=3
// Common in early layers of ResNet, AlexNet, and other deep architectures
func (c *Conv2D) conv2D7x7Optimized(input []float32, inputHeight, inputWidth, outH, outW int) {
	outSize := outH * outW
	inChannels := c.inChannels
	outChannels := c.outChannels

	// Kernel weights are stored as: [outChannels][inChannels][7][7]
	// Pre-fetch weights into local arrays for better cache locality

	for oc := 0; oc < outChannels; oc++ {
		ocWeightBase := oc * inChannels * 49 // 49 = 7*7
		ocOutBase := oc * outSize

		// Clear output buffer for this channel
		for i := 0; i < outSize; i++ {
			c.preActBuf[ocOutBase+i] = 0
		}

		// Process input channels
		for ic := 0; ic < inChannels; ic++ {
			// Pre-fetch the 7x7 kernel weights for this input/output channel pair
			wBase := ocWeightBase + ic*49
			var w [49]float32
			for i := 0; i < 49; i++ {
				w[i] = c.weights[wBase+i]
			}

			icOffset := ic * inputHeight * inputWidth

			// Process output positions with loop unrolling
			// Skip boundary checks by processing only valid interior region

			// Process interior region (no boundary checks needed for kernel positions)
			for oh := 3; oh < outH-3; oh++ {
				outRowOffset := ocOutBase + oh*outW
				inRow := oh

				// Pre-compute row offsets for the 7 rows
				var rowOffset [7]int
				for i := 0; i < 7; i++ {
					rowOffset[i] = icOffset + (inRow-3+i)*inputWidth
				}

				// Process in chunks of 2 for better instruction pipelining
				ow := 3
				for ; ow <= outW-4; ow += 2 {
					pos0 := outRowOffset + ow
					pos1 := pos0 + 1
					inCol0 := ow
					inCol1 := ow + 1

					// Unroll all 7 rows
					for row := 0; row < 7; row++ {
						wIdx := row * 7
						c.preActBuf[pos0] += w[wIdx+0]*input[rowOffset[row]+inCol0-3] +
							w[wIdx+1]*input[rowOffset[row]+inCol0-2] +
							w[wIdx+2]*input[rowOffset[row]+inCol0-1] +
							w[wIdx+3]*input[rowOffset[row]+inCol0] +
							w[wIdx+4]*input[rowOffset[row]+inCol0+1] +
							w[wIdx+5]*input[rowOffset[row]+inCol0+2] +
							w[wIdx+6]*input[rowOffset[row]+inCol0+3]
						c.preActBuf[pos1] += w[wIdx+0]*input[rowOffset[row]+inCol1-3] +
							w[wIdx+1]*input[rowOffset[row]+inCol1-2] +
							w[wIdx+2]*input[rowOffset[row]+inCol1-1] +
							w[wIdx+3]*input[rowOffset[row]+inCol1] +
							w[wIdx+4]*input[rowOffset[row]+inCol1+1] +
							w[wIdx+5]*input[rowOffset[row]+inCol1+2] +
							w[wIdx+6]*input[rowOffset[row]+inCol1+3]
					}
				}

				// Handle remaining elements
				for ; ow < outW-3; ow++ {
					pos := outRowOffset + ow
					inCol := ow

					for row := 0; row < 7; row++ {
						wIdx := row * 7
						c.preActBuf[pos] += w[wIdx+0]*input[rowOffset[row]+inCol-3] +
							w[wIdx+1]*input[rowOffset[row]+inCol-2] +
							w[wIdx+2]*input[rowOffset[row]+inCol-1] +
							w[wIdx+3]*input[rowOffset[row]+inCol] +
							w[wIdx+4]*input[rowOffset[row]+inCol+1] +
							w[wIdx+5]*input[rowOffset[row]+inCol+2] +
							w[wIdx+6]*input[rowOffset[row]+inCol+3]
					}
				}
			}

			// Handle boundary regions with bounds checking
			// Only process first 3 rows, last 3 rows, first 3 columns, and last 3 columns
			// Top and bottom rows
			for oh := 0; oh < 3 || oh >= outH-3; oh++ {
				if oh >= outH {
					break
				}
				outRowOffset := ocOutBase + oh*outW
				for ow := 0; ow < outW; ow++ {
					pos := outRowOffset + ow
					for kh := 0; kh < 7; kh++ {
						inH := oh + kh - 3
						if inH >= 0 && inH < inputHeight {
							rowOffset := icOffset + inH*inputWidth
							for kw := 0; kw < 7; kw++ {
								inW := ow + kw - 3
								if inW >= 0 && inW < inputWidth {
									wIdx := kh*7 + kw
									c.preActBuf[pos] += w[wIdx] * input[rowOffset+inW]
								}
							}
						}
					}
				}
				if oh >= 2 && oh < outH-3 {
					// After processing oh=0,1,2, skip to outH-3
					oh = outH - 4
				}
			}
			// Left and right columns (excluding corners already processed)
			for oh := 3; oh < outH-3; oh++ {
				outRowOffset := ocOutBase + oh*outW
				// Process first 3 columns
				for ow := 0; ow < 3; ow++ {
					pos := outRowOffset + ow
					for kh := 0; kh < 7; kh++ {
						inH := oh + kh - 3
						if inH >= 0 && inH < inputHeight {
							rowOffset := icOffset + inH*inputWidth
							for kw := 0; kw < 7; kw++ {
								inW := ow + kw - 3
								if inW >= 0 && inW < inputWidth {
									wIdx := kh*7 + kw
									c.preActBuf[pos] += w[wIdx] * input[rowOffset+inW]
								}
							}
						}
					}
				}
				// Process last 3 columns
				for ow := outW - 3; ow < outW; ow++ {
					pos := outRowOffset + ow
					for kh := 0; kh < 7; kh++ {
						inH := oh + kh - 3
						if inH >= 0 && inH < inputHeight {
							rowOffset := icOffset + inH*inputWidth
							for kw := 0; kw < 7; kw++ {
								inW := ow + kw - 3
								if inW >= 0 && inW < inputWidth {
									wIdx := kh*7 + kw
									c.preActBuf[pos] += w[wIdx] * input[rowOffset+inW]
								}
							}
						}
					}
				}
			}
		}
	}
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

func (c *Conv2D) ForwardWithArena(input []float32, arena *[]float32, offset *int) ([]float32, error) {
	// Infer input dimensions from length
	totalInput := len(input)
	if totalInput%c.inChannels != 0 {
		return nil, fmt.Errorf("Conv2D: input length %d not divisible by inChannels %d", totalInput, c.inChannels)
	}
	channelSize := totalInput / c.inChannels

	// Use explicitly set dimensions if available, otherwise infer
	var inputHeight, inputWidth int
	if c.setInputHeight > 0 && c.setInputWidth > 0 {
		inputHeight = c.setInputHeight
		inputWidth = c.setInputWidth
		// Verify the dimensions match the input length
		if inputHeight*inputWidth != channelSize {
			return nil, fmt.Errorf("Conv2D: input dimensions %dx%d don't match channelSize %d", inputHeight, inputWidth, channelSize)
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

	// Save input for backward pass - assumes arena is pre-allocated with sufficient capacity
	if arena != nil && offset != nil {
		c.arenaMu.Lock()
		c.arenaPtr = arena
		// Assert: arena must have sufficient capacity
		if cap(*arena) < *offset+len(input) {
			c.arenaMu.Unlock()
			return nil, fmt.Errorf("%w: arena capacity %d insufficient for Conv2D offset %d + input %d; "+
				"ensure Network pre-allocates arena using CalculateTotalArenaSize()",
				ErrInvalidDimensions, cap(*arena), *offset, len(input))
		}
		// Extend length if needed (no allocation, just reslice)
		if len(*arena) < *offset+len(input) {
			*arena = (*arena)[:*offset+len(input)]
		}
		saved := (*arena)[*offset : *offset+len(input)]
		copy(saved, input)
		c.savedInputOffsets = append(c.savedInputOffsets, *offset)
		*offset += len(input)
		c.arenaMu.Unlock()
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

		// Ensure buffers with pooling - pre-allocate to max size seen
		if len(input) > c.maxInputSize {
			c.maxInputSize = len(input)
		}
		if c.bufInput == nil || c.bufInput.length < c.maxInputSize {
			if c.bufInput != nil {
				c.bufInput.Free()
			}
			c.bufInput = md.CreateEmptyBuffer(c.maxInputSize)
		}
		c.bufInput.Update(input)

		if requiredOutput > c.maxOutputSize {
			c.maxOutputSize = requiredOutput
		}
		if c.bufOutput == nil || c.bufOutput.length < c.maxOutputSize {
			if c.bufOutput != nil {
				c.bufOutput.Free()
			}
			c.bufOutput = md.CreateEmptyBuffer(c.maxOutputSize)
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

		return c.outputBuf[:requiredOutput], nil
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

	// OPTIMIZED CONVOLUTION: Loop reordering for better cache locality
	// Process output spatial positions with kernel unrolling for common kernel sizes

	if kernelSize == 3 && stride == 1 && padding == 1 {
		// Optimized 3x3 kernel with padding=1, stride=1 (most common)
		c.conv2D3x3Optimized(input, inputHeight, inputWidth, outH, outW)
	} else if kernelSize == 5 && stride == 1 && padding == 2 {
		// Optimized 5x5 kernel with padding=2, stride=1
		c.conv2D5x5Optimized(input, inputHeight, inputWidth, outH, outW)
	} else if kernelSize == 7 && stride == 1 && padding == 3 {
		// Optimized 7x7 kernel with padding=3, stride=1
		c.conv2D7x7Optimized(input, inputHeight, inputWidth, outH, outW)
	} else {
		// Generic fallback implementation
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
	}

	// Apply bias and activation after convolution is complete for optimized kernels
	// The optimized implementations (3x3, 5x5, 7x7) do not include bias/activation in the main loop
	isOptimizedKernel := (kernelSize == 3 && stride == 1 && padding == 1) ||
		(kernelSize == 5 && stride == 1 && padding == 2) ||
		(kernelSize == 7 && stride == 1 && padding == 3)

	if isOptimizedKernel {
		for oc := 0; oc < outChannels; oc++ {
			ocOutBase := oc * outSize
			biasVal := c.biases[oc]
			for oh := 0; oh < outH; oh++ {
				ohOffset := ocOutBase + oh*outW
				// Unrolled loop for better performance
				ow := 0
				for ; ow <= outW-4; ow += 4 {
					pos := ohOffset + ow
					c.preActBuf[pos] += biasVal
					c.preActBuf[pos+1] += biasVal
					c.preActBuf[pos+2] += biasVal
					c.preActBuf[pos+3] += biasVal
					c.outputBuf[pos] = c.activation.Activate(c.preActBuf[pos])
					c.outputBuf[pos+1] = c.activation.Activate(c.preActBuf[pos+1])
					c.outputBuf[pos+2] = c.activation.Activate(c.preActBuf[pos+2])
					c.outputBuf[pos+3] = c.activation.Activate(c.preActBuf[pos+3])
				}
				for ; ow < outW; ow++ {
					pos := ohOffset + ow
					c.preActBuf[pos] += biasVal
					c.outputBuf[pos] = c.activation.Activate(c.preActBuf[pos])
				}
			}
		}
	}

	return c.outputBuf[:requiredOutput], nil
}

// Forward performs a forward pass through the convolutional layer.
func (c *Conv2D) Forward(input []float32) ([]float32, error) {
	return c.ForwardWithArena(input, nil, nil)
}

func (c *Conv2D) ForwardBatch(input []float32, batchSize int) ([]float32, error) {
	return c.ForwardBatchWithArena(input, batchSize, nil, nil)
}

func (c *Conv2D) ForwardBatchWithArena(input []float32, batchSize int, arena *[]float32, offset *int) ([]float32, error) {
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
		out, err := c.ForwardWithArena(input[i*inSize:(i+1)*inSize], arena, offset)
		if err != nil {
			return nil, err
		}
		copy(c.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return c.outputBuf[:batchSize*outSize], nil
}

// Backward performs backpropagation through the convolutional layer.
// grad: gradient of loss w.r.t. activated output (shape: [outChannels, outH, outW] flattened)
// Returns: gradient of loss w.r.t. input
func (c *Conv2D) Backward(grad []float32) ([]float32, error) {
	numSaved := len(c.savedInputOffsets)
	if numSaved == 0 {
		return nil, nil
	}

	// Use last saved input
	ts := numSaved - 1
	offset := c.savedInputOffsets[ts]
	inSize := c.inChannels * c.inputHeight * c.inputWidth

	// SEC-012: Validar offsets antes de construir slices
	if c.arenaPtr == nil {
		return nil, fmt.Errorf("%w: arena pointer is nil", ErrInvalidState)
	}
	arenaLen := len(*c.arenaPtr)
	if offset < 0 || offset > arenaLen-inSize {
		return nil, fmt.Errorf("%w: input offset %d out of bounds (arena size: %d, inSize: %d)",
			ErrInvalidState, offset, arenaLen, inSize)
	}

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
		// Ensure weight gradient buffers exist with pooling
		if len(grad) > c.maxGradOutSize {
			c.maxGradOutSize = len(grad)
		}
		if c.bufGradW == nil {
			c.bufGradW = md.CreateEmptyBuffer(len(c.weights))
		}
		if c.bufGradB == nil {
			c.bufGradB = md.CreateEmptyBuffer(len(c.biases))
		}

		if c.bufGradOut == nil || c.bufGradOut.length < c.maxGradOutSize {
			if c.bufGradOut != nil {
				c.bufGradOut.Free()
			}
			c.bufGradOut = md.CreateEmptyBuffer(c.maxGradOutSize)
		}

		if len(gradInput) > c.maxGradInSize {
			c.maxGradInSize = len(gradInput)
		}
		if c.bufGradIn == nil || c.bufGradIn.length < c.maxGradInSize {
			if c.bufGradIn != nil {
				c.bufGradIn.Free()
			}
			c.bufGradIn = md.CreateEmptyBuffer(c.maxGradInSize)
		}

		// Apply activation derivative using pre-allocated scratch buffer
		// dL/dz = dL/d(output) * activation'(z)
		// Since we didn't fuse, we do it here using scratch buffer
		// PERF-020: Protected by arenaMu to prevent race conditions in multi-worker training
		c.arenaMu.Lock()
		if cap(c.scratchBuf) < len(grad) {
			c.scratchBuf = make([]float32, len(grad))
		}
		scratch := c.scratchBuf[:len(grad)]
		c.arenaMu.Unlock()
		for i := range grad {
			scratch[i] = grad[i] * c.activation.Derivative(c.preActBuf[i])
		}
		c.bufGradOut.Update(scratch)

		md.Conv2DBackward(c.metalState, c.bufInput, c.bufGradOut, c.bufGradIn, c.bufGradW, c.bufGradB, 1, inputHeight, inputWidth, outH, outW)

		c.bufGradIn.Read(gradInput)

		// Accumulate gradients back to Go using scratchBuf to avoid allocation (PERF-020)
		// Reuse scratchBuf if it has enough capacity for both weight and bias gradients
		// Thread-safety: Protected by arenaMu to prevent race conditions in multi-worker training
		c.arenaMu.Lock()
		needSize := len(c.gradWeights) + len(c.gradBiases)
		if cap(c.scratchBuf) < needSize {
			c.scratchBuf = make([]float32, needSize)
		}
		gradScratch := c.scratchBuf[:needSize]
		c.arenaMu.Unlock()

		// PERF-016: Consolidate 2 reads into single synchronization point
		ReadMultiple(
			[]*MetalBuffer{c.bufGradW, c.bufGradB},
			[][]float32{gradScratch[:len(c.gradWeights)], gradScratch[len(c.gradWeights):]},
		)

		for i := range c.gradWeights {
			c.gradWeights[i] += gradScratch[i]
		}
		for i := range c.gradBiases {
			c.gradBiases[i] += gradScratch[len(c.gradWeights)+i]
		}

		// Pop the last saved input offset
		c.savedInputOffsets = c.savedInputOffsets[:ts]

		return gradInput, nil
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

	return gradInput, nil
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
func (c *Conv2D) AccumulateBackward(grad []float32) ([]float32, error) {
	return c.Backward(grad)
}

func (c *Conv2D) BackwardBatch(grad []float32, batchSize int) ([]float32, error) {
	if batchSize <= 1 {
		return c.Backward(grad)
	}
	inSize := c.InSize()
	outSize := c.OutSize()
	if len(c.gradInBuf) < batchSize*inSize {
		c.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		dx, err := c.Backward(grad[i*outSize : (i+1)*outSize])
		if err != nil {
			return nil, err
		}
		copy(c.gradInBuf[i*inSize:(i+1)*inSize], dx)
	}
	return c.gradInBuf[:batchSize*inSize], nil
}

func (c *Conv2D) AccumulateBackwardBatch(grad []float32, batchSize int) ([]float32, error) {
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
	newC, err := NewConv2D(c.inChannels, c.outChannels, c.kernelSize, c.stride, c.padding, c.activation)
	if err != nil {
		return nil
	}
	copy(newC.params, c.params)
	newC.setInputHeight = c.setInputHeight
	newC.setInputWidth = c.setInputWidth
	newC.device = c.device
	return newC
}

func (c *Conv2D) LightweightClone(params []float32, grads []float32) Layer {
	inChannels := c.inChannels
	if inChannels <= 0 {
		inChannels = 0 // Avoid negative index panic
	}
	weightSize := c.outChannels * inChannels * c.kernelSize * c.kernelSize
	newC := &Conv2D{
		inChannels:        c.inChannels,
		outChannels:       c.outChannels,
		kernelSize:        c.kernelSize,
		stride:            c.stride,
		padding:           c.padding,
		setInputHeight:    c.setInputHeight,
		setInputWidth:     c.setInputWidth,
		inputHeight:       c.inputHeight,
		inputWidth:        c.inputWidth,
		activation:        c.activation,
		params:            params,
		weights:           nil,
		biases:            nil,
		grads:             grads,
		gradWeights:       nil,
		gradBiases:        nil,
		outputBuf:         make([]float32, 0),
		gradInBuf:         make([]float32, 0),
		savedInput:        make([]float32, 0),
		savedInputOffsets: make([]int, 0, 128),
		training:          c.training,
		device:            c.device,
		metalState:        c.metalState,
	}

	if len(params) >= weightSize {
		newC.weights = params[:weightSize]
		newC.biases = params[weightSize:]
	}
	if len(grads) >= weightSize {
		newC.gradWeights = grads[:weightSize]
		newC.gradBiases = grads[weightSize:]
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

// ArenaSize returns the number of float32 values needed in the activation arena.
// Conv2D saves input (inChannels * inputHeight * inputWidth) for backward pass.
// Uses configured dimensions if available, otherwise returns 0 (dynamic allocation).
func (c *Conv2D) ArenaSize() int {
	if c.inputHeight > 0 && c.inputWidth > 0 {
		return c.inChannels * c.inputHeight * c.inputWidth
	}
	// Return 0 if dimensions not yet known - will use dynamic allocation
	return 0
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

