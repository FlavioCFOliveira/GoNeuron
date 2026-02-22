package layer

import (
	"math"
)

// RBF is a Radial Basis Function layer.
// It uses Gaussian RBF activation: phi(x) = exp(-gamma * ||x - center||^2)
type RBF struct {
	centers  []float64 // Shape: [numCenters * inSize]
	weights  []float64 // Shape: [outSize * numCenters]
	biases   []float64 // Shape: [outSize]
	gamma    float64   // Shape parameter for RBF kernel

	inSize     int
	numCenters int
	outSize    int

	// Pre-allocated buffers
	inputBuf     []float64 // [inSize]
	distBuf      []float64 // [numCenters] - Squared distances
	phiBuf       []float64 // [numCenters] - RBF activations
	outputBuf    []float64 // [outSize]

	gradWBuf     []float64 // [outSize * numCenters]
	gradBBuf     []float64 // [outSize]
	gradCBuf     []float64 // [numCenters * inSize] - Optional: gradients for centers
	gradInBuf    []float64 // [inSize]
	dzBuf        []float64 // [outSize]
}

// NewRBF creates a new RBF layer.
func NewRBF(inSize, numCenters, outSize int, gamma float64) *RBF {
	centers := make([]float64, numCenters*inSize)
	weights := make([]float64, outSize*numCenters)
	biases := make([]float64, outSize)

	// Simple initialization
	rng := NewRNG(uint64(inSize*1000 + numCenters*100 + outSize + 7))
	for i := range centers {
		centers[i] = rng.RandFloat()*2 - 1
	}

	// Xavier-like for weights
	scale := math.Sqrt(2.0 / float64(numCenters+outSize))
	for i := range weights {
		weights[i] = rng.RandFloat()*2*scale - scale
	}

	for i := range biases {
		biases[i] = rng.RandFloat()*0.2 - 0.1
	}

	return &RBF{
		centers:    centers,
		weights:    weights,
		biases:     biases,
		gamma:      gamma,
		inSize:     inSize,
		numCenters: numCenters,
		outSize:    outSize,
		inputBuf:   make([]float64, inSize),
		distBuf:    make([]float64, numCenters),
		phiBuf:     make([]float64, numCenters),
		outputBuf:  make([]float64, outSize),
		gradWBuf:   make([]float64, outSize*numCenters),
		gradBBuf:   make([]float64, outSize),
		gradCBuf:   make([]float64, numCenters*inSize),
		gradInBuf:  make([]float64, inSize),
		dzBuf:      make([]float64, outSize),
	}
}

// Forward performs a forward pass.
func (r *RBF) Forward(x []float64) []float64 {
	copy(r.inputBuf, x)

	// 1. Compute RBF activations: phi_j = exp(-gamma * ||x - c_j||^2)
	for j := 0; j < r.numCenters; j++ {
		distSq := 0.0
		cBase := j * r.inSize
		for i := 0; i < r.inSize; i++ {
			diff := x[i] - r.centers[cBase+i]
			distSq += diff * diff
		}
		r.distBuf[j] = distSq
		r.phiBuf[j] = math.Exp(-r.gamma * distSq)
	}

	// 2. Compute linear output: y = W * phi + b
	for o := 0; o < r.outSize; o++ {
		sum := r.biases[o]
		wBase := o * r.numCenters
		for j := 0; j < r.numCenters; j++ {
			sum += r.weights[wBase+j] * r.phiBuf[j]
		}
		r.outputBuf[o] = sum
	}

	return r.outputBuf
}

// Backward performs a backward pass.
func (r *RBF) Backward(grad []float64) []float64 {
	// 1. Gradients for Linear part (weights and biases)
	// dL/db_o = grad_o
	// dL/dW_{oj} = grad_o * phi_j
	for o := 0; o < r.outSize; o++ {
		r.gradBBuf[o] += grad[o]
		wBase := o * r.numCenters
		for j := 0; j < r.numCenters; j++ {
			r.gradWBuf[wBase+j] += grad[o] * r.phiBuf[j]
		}
	}

	// 2. Gradient for RBF activations (phi)
	// dL/dphi_j = sum_o (grad_o * W_{oj})
	for j := 0; j < r.numCenters; j++ {
		dphi := 0.0
		for o := 0; o < r.outSize; o++ {
			dphi += grad[o] * r.weights[o*r.numCenters+j]
		}

		// 3. Gradient for input x
		// dphi_j/dx_i = phi_j * (-gamma * 2 * (x_i - c_ji))
		// dL/dx_i = sum_j (dL/dphi_j * dphi_j/dx_i)
		factor := dphi * r.phiBuf[j] * (-2.0 * r.gamma)
		cBase := j * r.inSize
		for i := 0; i < r.inSize; i++ {
			r.gradInBuf[i] += factor * (r.inputBuf[i] - r.centers[cBase+i])
			// 4. Gradient for centers (optional, we include it for completeness)
			// dphi_j/dc_ji = phi_j * (-gamma * 2 * (x_i - c_ji) * -1) = phi_j * gamma * 2 * (x_i - c_ji)
			r.gradCBuf[cBase+i] += dphi * r.phiBuf[j] * (2.0 * r.gamma * (r.inputBuf[i] - r.centers[cBase+i]))
		}
	}

	return r.gradInBuf
}

// AccumulateBackward is like Backward but specifically for accumulation.
func (r *RBF) AccumulateBackward(grad []float64) []float64 {
	// For RBF, we already accumulate in Backward if we don't clear.
	// But let's follow the interface pattern.
	return r.Backward(grad)
}

// Params returns all parameters flattened: weights, biases, centers.
func (r *RBF) Params() []float64 {
	total := len(r.weights) + len(r.biases) + len(r.centers)
	p := make([]float64, total)
	n := copy(p, r.weights)
	n += copy(p[n:], r.biases)
	copy(p[n:], r.centers)
	return p
}

// SetParams updates parameters from flattened slice.
func (r *RBF) SetParams(p []float64) {
	n := copy(r.weights, p)
	n += copy(r.biases, p[n:])
	copy(r.centers, p[n:])
}

// Gradients returns all gradients flattened.
func (r *RBF) Gradients() []float64 {
	total := len(r.gradWBuf) + len(r.gradBBuf) + len(r.gradCBuf)
	g := make([]float64, total)
	n := copy(g, r.gradWBuf)
	n += copy(g[n:], r.gradBBuf)
	copy(g[n:], r.gradCBuf)
	return g
}

// SetGradients sets gradients from flattened slice.
func (r *RBF) SetGradients(g []float64) {
	n := copy(r.gradWBuf, g)
	n += copy(r.gradBBuf, g[n:])
	copy(r.gradCBuf, g[n:])
}

// Reset is a no-op for RBF.
func (r *RBF) Reset() {}

// ClearGradients zeroes out buffers.
func (r *RBF) ClearGradients() {
	for i := range r.gradWBuf { r.gradWBuf[i] = 0 }
	for i := range r.gradBBuf { r.gradBBuf[i] = 0 }
	for i := range r.gradCBuf { r.gradCBuf[i] = 0 }
	for i := range r.gradInBuf { r.gradInBuf[i] = 0 }
}

// Clone creates a deep copy.
func (r *RBF) Clone() Layer {
	newR := NewRBF(r.inSize, r.numCenters, r.outSize, r.gamma)
	copy(newR.weights, r.weights)
	copy(newR.biases, r.biases)
	copy(newR.centers, r.centers)
	return newR
}
