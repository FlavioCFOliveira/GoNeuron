package layer

import (
	"math"
)

// RBF is a Radial Basis Function layer.
// It uses Gaussian RBF activation: phi(x) = exp(-gamma * ||x - center||^2)
type RBF struct {
	centers  []float32 // Shape: [numCenters * inSize]
	weights  []float32 // Shape: [outSize * numCenters]
	biases   []float32 // Shape: [outSize]
	gamma    float32   // Shape parameter for RBF kernel

	inSize     int
	numCenters int
	outSize    int

	// Pre-allocated buffers
	inputBuf     []float32 // [inSize]
	distBuf      []float32 // [numCenters] - Squared distances
	phiBuf       []float32 // [numCenters] - RBF activations
	outputBuf    []float32 // [outSize]

	gradWBuf     []float32 // [outSize * numCenters]
	gradBBuf     []float32 // [outSize]
	gradCBuf     []float32 // [numCenters * inSize] - Optional: gradients for centers
	gradInBuf    []float32 // [inSize]
	dzBuf        []float32 // [outSize]

	device Device
}

// NewRBF creates a new RBF layer.
func NewRBF(inSize, numCenters, outSize int, gamma float32) *RBF {
	centers := make([]float32, numCenters*inSize)
	weights := make([]float32, outSize*numCenters)
	biases := make([]float32, outSize)

	// Simple initialization
	rng := NewRNG(uint64(inSize*1000 + numCenters*100 + outSize + 7))
	for i := range centers {
		centers[i] = rng.RandFloat()*2 - 1
	}

	// Xavier-like for weights
	scale := float32(math.Sqrt(2.0 / float64(numCenters+outSize)))
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
		inputBuf:   make([]float32, inSize),
		distBuf:    make([]float32, numCenters),
		phiBuf:     make([]float32, numCenters),
		outputBuf:  make([]float32, outSize),
		gradWBuf:   make([]float32, outSize*numCenters),
		gradBBuf:   make([]float32, outSize),
		gradCBuf:   make([]float32, numCenters*inSize),
		gradInBuf:  make([]float32, inSize),
		dzBuf:      make([]float32, outSize),
		device:     &CPUDevice{},
	}
}

// SetDevice sets the computation device.
func (r *RBF) SetDevice(device Device) {
	r.device = device
}

// Forward performs a forward pass.
func (r *RBF) Forward(x []float32) []float32 {
	copy(r.inputBuf, x)

	// 1. Compute RBF activations: phi_j = exp(-gamma * ||x - c_j||^2)
	for j := 0; j < r.numCenters; j++ {
		distSq := float32(0.0)
		cBase := j * r.inSize
		for i := 0; i < r.inSize; i++ {
			diff := x[i] - r.centers[cBase+i]
			distSq += diff * diff
		}
		r.distBuf[j] = distSq
		r.phiBuf[j] = float32(math.Exp(float64(-r.gamma * distSq)))
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
func (r *RBF) Backward(grad []float32) []float32 {
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
		dphi := float32(0.0)
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
func (r *RBF) AccumulateBackward(grad []float32) []float32 {
	// For RBF, we already accumulate in Backward if we don't clear.
	// But let's follow the interface pattern.
	return r.Backward(grad)
}

// Params returns all parameters flattened: weights, biases, centers.
func (r *RBF) Params() []float32 {
	total := len(r.weights) + len(r.biases) + len(r.centers)
	p := make([]float32, total)
	n := copy(p, r.weights)
	n += copy(p[n:], r.biases)
	copy(p[n:], r.centers)
	return p
}

// SetParams updates parameters from flattened slice.
func (r *RBF) SetParams(p []float32) {
	n := copy(r.weights, p)
	n += copy(r.biases, p[n:])
	copy(r.centers, p[n:])
}

// Gradients returns all gradients flattened.
func (r *RBF) Gradients() []float32 {
	total := len(r.gradWBuf) + len(r.gradBBuf) + len(r.gradCBuf)
	g := make([]float32, total)
	n := copy(g, r.gradWBuf)
	n += copy(g[n:], r.gradBBuf)
	copy(g[n:], r.gradCBuf)
	return g
}

// SetGradients sets gradients from flattened slice.
func (r *RBF) SetGradients(g []float32) {
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

// InSize returns the input size.
func (r *RBF) InSize() int {
	return r.inSize
}

// OutSize returns the output size.
func (r *RBF) OutSize() int {
	return r.outSize
}

// Clone creates a deep copy.
func (r *RBF) Clone() Layer {
	newR := NewRBF(r.inSize, r.numCenters, r.outSize, r.gamma)
	copy(newR.weights, r.weights)
	copy(newR.biases, r.biases)
	copy(newR.centers, r.centers)
	newR.device = r.device
	return newR
}

// NumCenters returns the number of RBF centers.
func (r *RBF) NumCenters() int {
	return r.numCenters
}

// Gamma returns the RBF gamma parameter.
func (r *RBF) Gamma() float32 {
	return r.gamma
}
