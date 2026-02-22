// Package layer provides neural network layer implementations.
package layer

import (
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"math"
)

// RNG is a simple, fast random number generator using LCG for deterministic behavior.
type RNG struct {
	seed uint64
}

// NewRNG creates a new RNG with the given seed.
func NewRNG(seed uint64) *RNG {
	return &RNG{seed: seed}
}

// RandFloat returns a random float64 in [0, 1).
func (r *RNG) RandFloat() float64 {
	r.seed = r.seed*6364136223846793005 + 1
	return float64(r.seed>>33) / float64(1<<31)
}

// Layer is a neural network layer.
type Layer interface {
	Forward(x []float64) []float64
	Backward(grad []float64) []float64
	AccumulateBackward(grad []float64) []float64
	Params() []float64
	SetParams([]float64)
	Gradients() []float64
	SetGradients([]float64)

	// Reset clears any internal state for the layer (e.g., LSTM hidden states).
	Reset()

	// ClearGradients zeroes out the accumulated gradients.
	ClearGradients()

	// Clone creates a deep copy of the layer.
	Clone() Layer
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
	inputBuf      []float64
	outputBuf     []float64
	preActBuf     []float64
	gradWBuf      []float64
	gradBBuf      []float64
	gradInBuf     []float64
	dzBuf         []float64

	// Saved inputs and pre-activations for gradient accumulation (triplet loss training)
	savedInputs  [][]float64
	savedPreActs [][]float64

	// Saved output for batch activation derivative computation
	savedOutput []float64
}

// NewDense creates a new dense layer with pre-allocated buffers.
func NewDense(in, out int, act activations.Activation) *Dense {
	weights := make([]float64, out*in)
	biases := make([]float64, out)

	// Xavier/Glorot initialization
	scale := math.Sqrt(2.0 / (float64(in) + float64(out)))
	// Create deterministic RNG with layer-specific seed for different initial weights
	rng := NewRNG(uint64(in*1000 + out*100 + 42))
	for i := range weights {
		weights[i] = rng.RandFloat()*2*scale - scale
	}
	for i := range biases {
		biases[i] = rng.RandFloat()*0.2 - 0.1
	}

	return &Dense{
		weights:      weights,
		biases:       biases,
		act:          act,
		outSize:      out,
		inSize:       in,
		inputBuf:     make([]float64, in),
		outputBuf:    make([]float64, out),
		preActBuf:    make([]float64, out),
		gradWBuf:     make([]float64, out*in),
		gradBBuf:     make([]float64, out),
		gradInBuf:    make([]float64, in),
		dzBuf:        make([]float64, out),
		savedInputs:  make([][]float64, 0),
		savedPreActs: make([][]float64, 0),
	}
}

// Forward performs a forward pass through the dense layer.
// Uses simple nested loops for optimal cache locality and zero allocations.
func (d *Dense) Forward(x []float64) []float64 {
	// Copy input to reusable buffer (no allocation)
	copy(d.inputBuf, x)

	// Save input for gradient accumulation (used in AccumulateBackward)
	d.saveInput(x)

	// Compute Wx + b with pre-allocated buffers
	// W is [outSize, inSize], x is [inSize], result is [outSize]
	outSize := d.outSize
	inSize := d.inSize
	weights := d.weights
	biases := d.biases
	input := d.inputBuf
	preAct := d.preActBuf
	output := d.outputBuf

	// Check if activation supports batch processing (e.g., Softmax, LogSoftmax)
	// For batch activations, we compute all pre-activations first, then apply activation
	if batchAct, ok := d.act.(activations.BatchActivation); ok {
		// Compute all pre-activations first
		for o := 0; o < outSize; o++ {
			// Compute dot product: sum(W[o, :] * x)
			sum := biases[o]
			wBase := o * inSize
			for i := 0; i < inSize; i++ {
				sum += weights[wBase+i] * input[i]
			}
			preAct[o] = sum
		}
		// Apply batch activation (copies output, then modifies it in place)
		copy(output, preAct)
		batchAct.ActivateBatch(output)

		// Save output for backward pass (needed for batch activation derivatives)
		if cap(d.savedOutput) < outSize {
			d.savedOutput = make([]float64, outSize)
		}
		copy(d.savedOutput[:outSize], output[:outSize])
	} else {
		// Standard activation: compute and apply for each output neuron
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
	}

	// Save pre-activations for gradient accumulation
	d.savePreAct(preAct)

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

	// Check if activation is a batch activation (Softmax or LogSoftmax)
	if _, isBatchAct := d.act.(activations.BatchActivation); isBatchAct {
		// For batch activations, use the saved output to compute derivatives
		// The output was saved during Forward pass in savedOutput
		output := d.savedOutput[:outSize]

		// For LogSoftmax:
		// log_softmax[i] = pre_act[i] - log(sum_j(exp(pre_act[j])))
		// d(log_softmax[i])/d(pre_act[k]) = delta[i,k] - softmax[k]
		//
		// For NLLLoss with log-prob input:
		// loss = -sum_i(y[i] * log_prob[i]) / n
		// dloss/d(log_prob[k]) = -y[k] / n
		//
		// Combined gradient:
		// dloss/d(pre_act[k]) = sum_i(dloss/d(log_softmax[i]) * d(log_softmax[i])/d(pre_act[k]))
		// = sum_i(-y[i]/n * (delta[i,k] - softmax[k]))
		// = -y[k]/n + softmax[k] * sum_i(y[i])/n
		// For one-hot: sum_i(y[i]) = 1
		// = (softmax[k] - y[k]) / n
		//
		// Since grad = -y/n, we have y = -n * grad
		// = (softmax[k] + n * grad[k]) / n
		// = softmax[k]/n + grad[k]
		//
		// Wait, let me redo this more carefully:
		// grad[k] = -y[k] / n
		// y[k] = -n * grad[k]
		// dloss/d(pre_act[k]) = softmax[k]/n - y[k]/n
		// = softmax[k]/n + grad[k]
		//
		// So: dz[k] = grad[k] + softmax[k]/n

		// Compute softmax from saved log-softmax output
		softmax := make([]float64, outSize)
		var sumSoftmax float64
		for o := 0; o < outSize; o++ {
			// exp(log_softmax) = softmax
			softmax[o] = math.Exp(output[o])
			sumSoftmax += softmax[o]
		}
		// Normalize (should be 1, but numerical stability)
		for o := 0; o < outSize; o++ {
			softmax[o] /= sumSoftmax
		}

		// Compute dz = grad + softmax / n
		// where n = outSize (number of classes)
		n := float64(outSize)
		for o := 0; o < outSize; o++ {
			dz[o] = grad[o] + softmax[o]/n
			gradB[o] = dz[o]
		}
	} else {
		// Standard activation: compute derivative for each output neuron
		for o := 0; o < outSize; o++ {
			deriv := d.act.Derivative(d.preActBuf[o])
			dz[o] = grad[o] * deriv
			gradB[o] = dz[o]
		}
	}

	// Step 2: Compute weight gradients: dL/dW[o, i] = dz[o] * input[i]
	for o := 0; o < outSize; o++ {
		dzo := dz[o]
		wBase := o * inSize
		for i := 0; i < inSize; i++ {
			gradW[wBase+i] += dzo * input[i]
		}
	}

	// Step 3: Compute bias gradients
	for o := 0; o < outSize; o++ {
		gradB[o] += dz[o]
	}

	// Step 4: Compute input gradients: dL/dx[i] = sum_o(dz[o] * W[o, i])
	// Optimized version using chunked access for better cache utilization
	const chunkSize = 8
	for chunk := 0; chunk < inSize; chunk += chunkSize {
		limit := chunk + chunkSize
		if limit > inSize {
			limit = inSize
		}
		for i := chunk; i < limit; i++ {
			sum := 0.0
			for o := 0; o < outSize; o++ {
				sum += dz[o] * weights[o*inSize+i]
			}
			gradIn[i] = sum
		}
	}

	return gradIn[:inSize]
}

// Params returns all dense layer parameters flattened.
// This creates a copy of the parameters.
func (d *Dense) Params() []float64 {
	total := len(d.weights) + len(d.biases)
	params := make([]float64, total)
	copy(params, d.weights)
	copy(params[len(d.weights):], d.biases)
	return params
}

// SetParams updates weights and biases from a flattened slice (in-place).
func (d *Dense) SetParams(params []float64) {
	copy(d.weights, params[:len(d.weights)])
	copy(d.biases, params[len(d.weights):])
}

// Gradients returns all dense layer gradients flattened.
// This creates a copy of the gradients.
func (d *Dense) Gradients() []float64 {
	total := len(d.gradWBuf) + len(d.gradBBuf)
	gradients := make([]float64, total)
	copy(gradients, d.gradWBuf)
	copy(gradients[len(d.gradWBuf):], d.gradBBuf)
	return gradients
}

// SetGradients sets gradients from a flattened slice (in-place).
// This is used by the optimizer to set averaged gradients.
func (d *Dense) SetGradients(gradients []float64) {
	copy(d.gradWBuf, gradients[:len(d.gradWBuf)])
	copy(d.gradBBuf, gradients[len(d.gradWBuf):])
}

// GetWeights returns the weights slice directly (no copy).
func (d *Dense) GetWeights() []float64 {
	return d.weights
}

// GetBiases returns the biases slice directly (no copy).
func (d *Dense) GetBiases() []float64 {
	return d.biases
}

// GetGradWeights returns the gradient weights slice directly (no copy).
func (d *Dense) GetGradWeights() []float64 {
	return d.gradWBuf
}

// GetGradBiases returns the gradient biases slice directly (no copy).
func (d *Dense) GetGradBiases() []float64 {
	return d.gradBBuf
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

// Reset clears the dense layer's internal state (no-op for Dense).
func (d *Dense) Reset() {
	// Dense layer has no internal state to reset
	d.savedInputs = d.savedInputs[:0]
	d.savedPreActs = d.savedPreActs[:0]
	d.savedOutput = d.savedOutput[:0]
}

// GetSavedInputs returns the saved inputs buffer.
func (d *Dense) GetSavedInputs() [][]float64 {
	return d.savedInputs
}

// GetSavedPreActs returns the saved pre-activations buffer.
func (d *Dense) GetSavedPreActs() [][]float64 {
	return d.savedPreActs
}

// ClearSavedData clears the saved inputs and pre-activations.
func (d *Dense) ClearSavedData() {
	d.savedInputs = d.savedInputs[:0]
	d.savedPreActs = d.savedPreActs[:0]
}

// ClearGradients zeroes out the accumulated gradients.
func (d *Dense) ClearGradients() {
	for i := range d.gradWBuf {
		d.gradWBuf[i] = 0
	}
	for i := range d.gradBBuf {
		d.gradBBuf[i] = 0
	}
}

// Clone creates a deep copy of the dense layer.
func (d *Dense) Clone() Layer {
	newD := NewDense(d.inSize, d.outSize, d.act)
	copy(newD.weights, d.weights)
	copy(newD.biases, d.biases)
	return newD
}

// GetGradWBuf returns the gradient weights buffer.
func (d *Dense) GetGradWBuf() []float64 {
	return d.gradWBuf
}

// GetGradBBuf returns the gradient biases buffer.
func (d *Dense) GetGradBBuf() []float64 {
	return d.gradBBuf
}

// GetActivation returns the activation function.
func (d *Dense) GetActivation() activations.Activation {
	return d.act
}

// saveInput adds the input to the saved inputs buffer for later gradient accumulation.
// This is used for triplet loss training where multiple forward passes are done
// before backward passes.
func (d *Dense) saveInput(x []float64) {
	// Create a copy of the input
	saved := make([]float64, len(x))
	copy(saved, x)
	d.savedInputs = append(d.savedInputs, saved)
}

// savePreAct adds the pre-activations to the saved pre-acts buffer.
func (d *Dense) savePreAct(preAct []float64) {
	// Create a copy of the pre-activations
	saved := make([]float64, len(preAct))
	copy(saved, preAct)
	d.savedPreActs = append(d.savedPreActs, saved)
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// This is used for triplet loss training where gradients from multiple
// forward passes need to be accumulated.
// Each forward pass saves its input and pre-activation, and AccumulateBackward
// uses the correct pre-activation for each saved input.
func (d *Dense) AccumulateBackward(grad []float64) []float64 {
	outSize := d.outSize
	inSize := d.inSize
	weights := d.weights
	dz := d.dzBuf
	gradW := d.gradWBuf
	gradB := d.gradBBuf
	gradIn := d.gradInBuf

	numSaved := len(d.savedInputs)

	// Clear input gradient buffer (but NOT gradW/gradB - we accumulate)
	for i := range gradIn {
		gradIn[i] = 0
	}

	// For each saved input from forward passes, accumulate gradients
	for s := 0; s < numSaved; s++ {
		// Get the saved pre-activation for this forward pass
		savedPreAct := d.savedPreActs[s]

		// Step 1: Compute dz = dL/dz = dL/d(output) * activation'(z)
		// Use the saved pre-activation for this forward pass
		for o := 0; o < outSize; o++ {
			deriv := d.act.Derivative(savedPreAct[o])
			dz[o] = grad[o] * deriv
			gradB[o] += dz[o] // Accumulate instead of overwrite
		}

		// Step 2: Compute weight gradients: dL/dW[o, i] = dz[o] * input[i]
		// Use the saved input from the corresponding forward pass
		savedInput := d.savedInputs[s]
		for o := 0; o < outSize; o++ {
			dzo := dz[o]
			wBase := o * inSize
			for i := 0; i < inSize; i++ {
				gradW[wBase+i] += dzo * savedInput[i] // Accumulate instead of overwrite
			}
		}

		// Step 3: Compute input gradients: dL/dx[i] = sum_o(dz[o] * W[o, i])
		// For the first saved input only (we return this gradient)
		if s == 0 {
			for i := 0; i < inSize; i++ {
				sum := 0.0
				for o := 0; o < outSize; o++ {
					sum += dz[o] * weights[o*inSize+i]
				}
				gradIn[i] = sum
			}
		}
	}

	return gradIn[:inSize]
}
