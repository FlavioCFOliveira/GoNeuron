package layer

import (
	"math"
)

// RBF is a Radial Basis Function layer.
// It uses Gaussian RBF activation: phi(x) = exp(-gamma * ||x - center||^2)
type RBF struct {
	// Parameters stored contiguously
	params []float32 // Contiguous weights + biases + centers
	grads  []float32 // Contiguous gradWeights + gradBiases + gradCenters

	// Views into params
	centers []float32 // Shape: [numCenters * inSize]
	weights []float32 // Shape: [outSize * numCenters]
	biases  []float32 // Shape: [outSize]

	// Views into grads
	gradWBuf []float32 // [outSize * numCenters]
	gradBBuf []float32 // [outSize]
	gradCBuf []float32 // [numCenters * inSize]

	gamma float32 // Shape parameter for RBF kernel

	inSize     int
	numCenters int
	outSize    int

	// Pre-allocated buffers
	inputBuf     []float32 // [inSize]
	distBuf      []float32 // [numCenters] - Squared distances
	phiBuf       []float32 // [numCenters] - RBF activations
	outputBuf    []float32 // [outSize]

	gradInBuf    []float32 // [inSize]
	dzBuf        []float32 // [outSize]

	training bool

	device Device
}

// NewRBF creates a new RBF layer.
func NewRBF(inSize, numCenters, outSize int, gamma float32) *RBF {
	weightSize := outSize * numCenters
	biasSize := outSize
	centerSize := numCenters * inSize
	totalParams := weightSize + biasSize + centerSize

	params := make([]float32, totalParams)
	weights := params[:weightSize]
	biases := params[weightSize : weightSize+biasSize]
	centers := params[weightSize+biasSize:]

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

	grads := make([]float32, totalParams)

	return &RBF{
		params:     params,
		grads:      grads,
		centers:    centers,
		weights:    weights,
		biases:     biases,
		gradWBuf:   grads[:weightSize],
		gradBBuf:   grads[weightSize : weightSize+biasSize],
		gradCBuf:   grads[weightSize+biasSize:],
		gamma:      gamma,
		inSize:     inSize,
		numCenters: numCenters,
		outSize:    outSize,
		inputBuf:   make([]float32, inSize),
		distBuf:    make([]float32, numCenters),
		phiBuf:     make([]float32, numCenters),
		outputBuf:  make([]float32, outSize),
		gradInBuf:  make([]float32, inSize),
		dzBuf:      make([]float32, outSize),
		device:     &CPUDevice{},
	}
}

// SetDevice sets the computation device.
func (r *RBF) SetDevice(device Device) {
	r.device = device
}

// SetTraining sets whether the layer is in training mode.
func (r *RBF) SetTraining(training bool) {
	r.training = training
}

func (r *RBF) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	if len(r.inputBuf) < r.inSize {
		r.inputBuf = make([]float32, r.inSize)
	}
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

	// Save input and phi for backward pass
	if arena != nil && offset != nil {
		inSize := r.inSize
		phiSize := r.numCenters
		required := inSize + phiSize
		if len(*arena) < *offset+required {
			newArena := make([]float32, (*offset+required)*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		savedIn := (*arena)[*offset : *offset+inSize]
		copy(savedIn, x)
		r.inputBuf = savedIn
		*offset += inSize

		savedPhi := (*arena)[*offset : *offset+phiSize]
		copy(savedPhi, r.phiBuf)
		r.phiBuf = savedPhi
		*offset += phiSize
	}

	return r.outputBuf[:r.outSize]
}

// Forward performs a forward pass.
func (r *RBF) Forward(x []float32) []float32 {
	return r.ForwardWithArena(x, nil, nil)
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

	return r.gradInBuf[:r.inSize]
}

// AccumulateBackward is like Backward but specifically for accumulation.
func (r *RBF) AccumulateBackward(grad []float32) []float32 {
	return r.Backward(grad)
}

func (r *RBF) ForwardBatch(x []float32, batchSize int) []float32 {
	return r.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (r *RBF) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return r.ForwardWithArena(x, arena, offset)
	}
	inSize := r.inSize
	outSize := r.outSize
	if len(r.outputBuf) < batchSize*outSize {
		r.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := r.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(r.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return r.outputBuf[:batchSize*outSize]
}

func (r *RBF) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return r.Backward(grad)
	}
	inSize := r.inSize
	outSize := r.outSize
	dx := make([]float32, batchSize*inSize)
	for i := batchSize - 1; i >= 0; i-- {
		out := r.Backward(grad[i*outSize : (i+1)*outSize])
		copy(dx[i*inSize:(i+1)*inSize], out)
	}
	return dx
}

func (r *RBF) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return r.BackwardBatch(grad, batchSize)
}

// Params returns all parameters flattened.
func (r *RBF) Params() []float32 {
	return r.params
}

// SetParams updates parameters from flattened slice.
func (r *RBF) SetParams(p []float32) {
	if len(p) == 0 {
		return
	}
	if &r.params[0] != &p[0] {
		if len(p) == len(r.params) {
			r.params = p
			r.updateViews()
		} else {
			copy(r.params, p)
		}
	}
}

func (r *RBF) updateViews() {
	weightSize := r.outSize * r.numCenters
	biasSize := r.outSize
	r.weights = r.params[:weightSize]
	r.biases = r.params[weightSize : weightSize+biasSize]
	r.centers = r.params[weightSize+biasSize:]
}

// Gradients returns all gradients flattened.
func (r *RBF) Gradients() []float32 {
	return r.grads
}

// SetGradients sets gradients from flattened slice.
func (r *RBF) SetGradients(g []float32) {
	if len(g) == 0 {
		return
	}
	if &r.grads[0] != &g[0] {
		if len(g) == len(r.grads) {
			r.grads = g
			r.updateGradViews()
		} else {
			copy(r.grads, g)
		}
	}
}

func (r *RBF) updateGradViews() {
	weightSize := r.outSize * r.numCenters
	biasSize := r.outSize
	r.gradWBuf = r.grads[:weightSize]
	r.gradBBuf = r.grads[weightSize : weightSize+biasSize]
	r.gradCBuf = r.grads[weightSize+biasSize:]
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

func (r *RBF) NamedParams() []NamedParam {
	return []NamedParam{
		{
			Name:  "weights",
			Shape: []int{r.outSize, r.numCenters},
			Data:  r.weights,
		},
		{
			Name:  "biases",
			Shape: []int{r.outSize},
			Data:  r.biases,
		},
		{
			Name:  "centers",
			Shape: []int{r.numCenters, r.inSize},
			Data:  r.centers,
		},
	}
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

func (r *RBF) LightweightClone(params []float32, grads []float32) Layer {
	weightSize := r.outSize * r.numCenters
	biasSize := r.outSize
	newR := &RBF{
		params:     params,
		grads:      grads,
		weights:    params[:weightSize],
		biases:     params[weightSize : weightSize+biasSize],
		centers:    params[weightSize+biasSize:],
		gradWBuf:   grads[:weightSize],
		gradBBuf:   grads[weightSize : weightSize+biasSize],
		gradCBuf:   grads[weightSize+biasSize:],
		gamma:      r.gamma,
		inSize:     r.inSize,
		numCenters: r.numCenters,
		outSize:    r.outSize,
		inputBuf:   make([]float32, r.inSize),
		distBuf:    make([]float32, r.numCenters),
		phiBuf:     make([]float32, r.numCenters),
		outputBuf:  make([]float32, r.outSize),
		gradInBuf:  make([]float32, r.inSize),
		dzBuf:      make([]float32, r.outSize),
		training:   r.training,
		device:     r.device,
	}
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
