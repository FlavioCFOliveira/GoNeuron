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

// Adam optimizer for faster convergence.
// Adam maintains per-parameter state (first and second moment estimates).
type Adam struct {
	LearningRate float64
	Beta1        float64 // Exponential decay rate for first moment
	Beta2        float64 // Exponential decay rate for second moment
	Epsilon      float64 // Small constant for numerical stability

	// Per-parameter state - lazily initialized
	momentum1 [][]float64 // First moment (m)
	momentum2 [][]float64 // Second moment (v)
	timestep  int         // Current timestep for bias correction
}

// NewAdam creates a new Adam optimizer with default values.
func NewAdam(learningRate float64) *Adam {
	return &Adam{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		timestep:     0,
	}
}

// initMoments initializes the moment buffers for given parameter shape.
func (a *Adam) initMoments(params []float64) {
	if a.momentum1 == nil {
		a.momentum1 = make([][]float64, 1)
		a.momentum2 = make([][]float64, 1)
	}
	// Ensure we have at least one buffer
	if len(a.momentum1) == 0 {
		a.momentum1 = append(a.momentum1, make([]float64, len(params)))
		a.momentum2 = append(a.momentum2, make([]float64, len(params)))
	}
	// Reuse existing buffers if size matches
	if len(a.momentum1[0]) != len(params) {
		a.momentum1[0] = make([]float64, len(params))
		a.momentum2[0] = make([]float64, len(params))
	}
}

// Step computes updated parameters using Adam
func (a *Adam) Step(params, gradients []float64) []float64 {
	a.initMoments(params)

	a.timestep++
	lr := a.LearningRate
	beta1 := a.Beta1
	beta2 := a.Beta2
	eps := a.Epsilon

	m := a.momentum1[0]
	v := a.momentum2[0]

	// Bias correction factors: 1 - beta^t
	bias1 := 1 - math.Pow(beta1, float64(a.timestep))
	bias2 := 1 - math.Pow(beta2, float64(a.timestep))

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

	a.timestep++
	lr := a.LearningRate
	beta1 := a.Beta1
	beta2 := a.Beta2
	eps := a.Epsilon

	m := a.momentum1[0]
	v := a.momentum2[0]

	// Bias correction factors: 1 - beta^t
	bias1 := 1 - math.Pow(beta1, float64(a.timestep))
	bias2 := 1 - math.Pow(beta2, float64(a.timestep))

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
