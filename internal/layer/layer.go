// Package layer provides neural network layer implementations.
package layer

import (
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"math"
	"math/rand"
)

// Layer is a neural network layer.
type Layer interface {
	Forward(x []float64) []float64
	Backward(grad []float64) []float64
	Params() []float64
	SetParams([]float64)
	Gradients() []float64
}

// Dense is a fully connected layer optimized for performance.
// Uses contiguous memory layout with pre-allocated buffers for minimal allocations.
// Uses simple nested loops instead of external matrix libraries for better cache locality.
type Dense struct {
	// Weights stored as row-major contiguous slice for cache efficiency
	// Shape: [out * in] where weight for output i, input j is at weights[i*in + j]
	weights   []float64
	biases    []float64
	act       activations.Activation
	outSize   int
	inSize    int

	// Reusable buffers for gradient computation
	inputBuf  []float64
	outputBuf []float64
	preActBuf []float64
	gradWBuf  []float64
	gradBBuf  []float64
	gradInBuf []float64
	dzBuf     []float64
}

// NewDense creates a new dense layer with pre-allocated buffers.
func NewDense(in, out int, act activations.Activation) *Dense {
	weights := make([]float64, out*in)
	biases := make([]float64, out)

	// Xavier/Glorot initialization
	scale := math.Sqrt(2.0 / (float64(in) + float64(out)))
	for i := range weights {
		weights[i] = rand.Float64()*2*scale - scale
	}
	for i := range biases {
		biases[i] = rand.Float64()*0.2 - 0.1
	}

	return &Dense{
		weights:     weights,
		biases:      biases,
		act:         act,
		outSize:     out,
		inSize:      in,
		inputBuf:    make([]float64, in),
		outputBuf:   make([]float64, out),
		preActBuf:   make([]float64, out),
		gradWBuf:    make([]float64, out*in),
		gradBBuf:    make([]float64, out),
		gradInBuf:   make([]float64, in),
		dzBuf:       make([]float64, out),
	}
}

// Forward performs a forward pass through the dense layer.
// Uses simple nested loops for optimal cache locality and zero allocations.
func (d *Dense) Forward(x []float64) []float64 {
	// Copy input to reusable buffer (no allocation)
	copy(d.inputBuf, x)

	// Compute Wx + b with pre-allocated buffers
	// W is [outSize, inSize], x is [inSize], result is [outSize]
	outSize := d.outSize
	inSize := d.inSize
	weights := d.weights
	biases := d.biases
	input := d.inputBuf
	preAct := d.preActBuf
	output := d.outputBuf

	// For each output neuron
	for o := 0; o < outSize; o++ {
		// Compute dot product: sum(W[o, :] * x)
		sum := biases[o]
		wBase := o * inSize
		for i := 0; i < inSize; i++ {
			sum += weights[wBase+i] * input[i]
		}
		preAct[o] = sum
		output[o] = d.act.Activate(sum)
	}

	return output[:outSize]
}

// Backward performs backpropagation through the dense layer.
// Computes gradients for weights, biases, and input.
func (d *Dense) Backward(grad []float64) []float64 {
	outSize := d.outSize
	inSize := d.inSize
	weights := d.weights
	input := d.inputBuf
	dz := d.dzBuf
	gradW := d.gradWBuf
	gradB := d.gradBBuf
	gradIn := d.gradInBuf

	// Step 1: Compute dz = dL/dz = dL/d(output) * activation'(z)
	for o := 0; o < outSize; o++ {
		deriv := d.act.Derivative(d.preActBuf[o])
		dz[o] = grad[o] * deriv
		gradB[o] = dz[o]
	}

	// Step 2: Compute weight gradients: dL/dW[o, i] = dz[o] * input[i]
	for o := 0; o < outSize; o++ {
		dzo := dz[o]
		wBase := o * inSize
		for i := 0; i < inSize; i++ {
			gradW[wBase+i] = dzo * input[i]
		}
	}

	// Step 3: Compute input gradients: dL/dx[i] = sum_o(dz[o] * W[o, i])
	for i := 0; i < inSize; i++ {
		sum := 0.0
		for o := 0; o < outSize; o++ {
			sum += dz[o] * weights[o*inSize+i]
		}
		gradIn[i] = sum
	}

	return gradIn[:inSize]
}

// Params returns all dense layer parameters flattened.
func (d *Dense) Params() []float64 {
	total := len(d.weights) + len(d.biases)
	params := make([]float64, 0, total)
	params = append(params, d.weights...)
	params = append(params, d.biases...)
	return params
}

// SetParams updates weights and biases from a flattened slice (in-place).
func (d *Dense) SetParams(params []float64) {
	copy(d.weights, params[:len(d.weights)])
	copy(d.biases, params[len(d.weights):])
}

// Gradients returns all dense layer gradients flattened.
func (d *Dense) Gradients() []float64 {
	total := len(d.gradWBuf) + len(d.gradBBuf)
	gradients := make([]float64, 0, total)
	gradients = append(gradients, d.gradWBuf...)
	gradients = append(gradients, d.gradBBuf...)
	return gradients
}

// GetWeights returns the weights slice directly.
func (d *Dense) GetWeights() []float64 {
	return d.weights
}

// GetBiases returns the biases slice directly.
func (d *Dense) GetBiases() []float64 {
	return d.biases
}

// SetWeight sets a single weight at (row, col).
func (d *Dense) SetWeight(row, col int, val float64) {
	d.weights[row*d.inSize+col] = val
}

// SetBias sets a single bias.
func (d *Dense) SetBias(idx int, val float64) {
	d.biases[idx] = val
}

// GetWeight gets a single weight at (row, col).
func (d *Dense) GetWeight(row, col int) float64 {
	return d.weights[row*d.inSize+col]
}

// GetBias gets a single bias.
func (d *Dense) GetBias(idx int) float64 {
	return d.biases[idx]
}

// InSize returns the input size of the layer.
func (d *Dense) InSize() int {
	return d.inSize
}

// OutSize returns the output size of the layer.
func (d *Dense) OutSize() int {
	return d.outSize
}

// Activation returns the activation function used by this layer.
func (d *Dense) Activation() activations.Activation {
	return d.act
}
