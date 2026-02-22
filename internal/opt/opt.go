// Package opt provides optimization algorithms.
package opt

import "math"

// Optimizer updates network parameters based on gradients.
type Optimizer interface {

	// Step computes updated parameters: params - lr * gradients
	// Returns a new slice with updated values
	Step(params, gradients []float64) []float64

	// StepInPlace updates params in-place: params = params - lr * gradients
	// This avoids allocations for better performance
	StepInPlace(params, gradients []float64)

	// NewStep signals the beginning of a new optimization step (e.g., for Adam timestep).
	NewStep()
}

// SGD (Stochastic Gradient Descent) optimizer.
type SGD struct {
	LearningRate float64
}

// Step computes updated parameters: params - lr * gradients
func (s SGD) Step(params, gradients []float64) []float64 {
	result := make([]float64, len(params))
	for i := range params {
		result[i] = params[i] - s.LearningRate*gradients[i]
	}
	return result
}

// StepInPlace updates params in-place: params = params - lr * gradients
// This avoids allocations for better performance
func (s SGD) StepInPlace(params, gradients []float64) {
	for i := range params {
		params[i] -= s.LearningRate * gradients[i]
	}
}

// NewStep signals the beginning of a new optimization step (no-op for SGD).
func (s SGD) NewStep() {}

// Adam optimizer for faster convergence.
// Adam maintains per-parameter state (first and second moment estimates).
// Uses flat slices for momentum buffers (one per layer in the network).
type Adam struct {
	LearningRate float64
	Beta1        float64 // Exponential decay rate for first moment
	Beta2        float64 // Exponential decay rate for second moment
	Epsilon      float64 // Small constant for numerical stability

	// Per-layer state - indexed by layer
	// momentum1[i] and momentum2[i] are for layer i
	momentum1 [][]float64 // First moment (m) for each layer
	momentum2 [][]float64 // Second moment (v) for each layer

	// Internal counter to track layer index during training
	// This is set by the network before calling Step
	currentLayer int

	// Global timestep for bias correction
	timestep int
}

// NewAdam creates a new Adam optimizer with default values.
func NewAdam(learningRate float64) *Adam {
	return &Adam{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		currentLayer: 0,
		timestep:     0,
	}
}

// initMoments initializes the moment buffers for given parameter shape.
func (a *Adam) initMoments(params []float64) {
	// Ensure we have enough layer buffers
	for len(a.momentum1) <= a.currentLayer {
		a.momentum1 = append(a.momentum1, nil)
		a.momentum2 = append(a.momentum2, nil)
	}

	// Initialize or resize the buffers for current layer
	if a.momentum1[a.currentLayer] == nil || len(a.momentum1[a.currentLayer]) != len(params) {
		a.momentum1[a.currentLayer] = make([]float64, len(params))
		a.momentum2[a.currentLayer] = make([]float64, len(params))
	}
}

// Step computes updated parameters using Adam
func (a *Adam) Step(params, gradients []float64) []float64 {
	a.initMoments(params)

	currentLayer := a.currentLayer
	a.currentLayer++

	lr := a.LearningRate
	// Use default values if not set
	beta1 := a.Beta1
	if beta1 == 0 {
		beta1 = 0.9
	}
	beta2 := a.Beta2
	if beta2 == 0 {
		beta2 = 0.999
	}
	eps := a.Epsilon
	if eps == 0 {
		eps = 1e-8
	}

	m := a.momentum1[currentLayer]
	v := a.momentum2[currentLayer]

	// Bias correction factors: 1 - beta^t
	// Timestep must be at least 1 for math.Pow to work correctly for bias correction
	ts := float64(a.timestep)
	if ts < 1 {
		ts = 1
	}
	bias1 := 1 - math.Pow(beta1, ts)
	bias2 := 1 - math.Pow(beta2, ts)

	result := make([]float64, len(params))
	for i := range params {
		// Update biased first moment estimate
		m[i] = beta1*m[i] + (1-beta1)*gradients[i]
		// Update biased second moment estimate
		v[i] = beta2*v[i] + (1-beta2)*gradients[i]*gradients[i]
		// Compute bias-corrected first moment estimate
		mHat := m[i] / bias1
		// Compute bias-corrected second moment estimate
		vHat := v[i] / bias2
		// Update parameters
		result[i] = params[i] - lr*mHat/(math.Sqrt(vHat)+eps)
	}
	return result
}

// StepInPlace updates params in-place using Adam
func (a *Adam) StepInPlace(params, gradients []float64) {
	a.initMoments(params)

	currentLayer := a.currentLayer
	a.currentLayer++

	lr := a.LearningRate
	// Use default values if not set
	beta1 := a.Beta1
	if beta1 == 0 {
		beta1 = 0.9
	}
	beta2 := a.Beta2
	if beta2 == 0 {
		beta2 = 0.999
	}
	eps := a.Epsilon
	if eps == 0 {
		eps = 1e-8
	}

	m := a.momentum1[currentLayer]
	v := a.momentum2[currentLayer]

	// Bias correction factors: 1 - beta^t
	ts := float64(a.timestep)
	if ts < 1 {
		ts = 1
	}
	bias1 := 1 - math.Pow(beta1, ts)
	bias2 := 1 - math.Pow(beta2, ts)

	for i := range params {
		// Update biased first moment estimate
		m[i] = beta1*m[i] + (1-beta1)*gradients[i]
		// Update biased second moment estimate
		v[i] = beta2*v[i] + (1-beta2)*gradients[i]*gradients[i]
		// Compute bias-corrected first moment estimate
		mHat := m[i] / bias1
		// Compute bias-corrected second moment estimate
		vHat := v[i] / bias2
		// Update parameters in-place
		params[i] -= lr * mHat / (math.Sqrt(vHat) + eps)
	}
}

// NewStep signals the beginning of a new optimization step for Adam.
func (a *Adam) NewStep() {
	a.timestep++
	a.currentLayer = 0
}
