// Package opt provides optimization algorithms.
package opt

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
type Adam struct {
	LearningRate float64
	Beta1        float64 // Exponential decay rate for first moment
	Beta2        float64 // Exponential decay rate for second moment
	Epsilon      float64 // Small constant for numerical stability
}

// NewAdam creates a new Adam optimizer with default values.
func NewAdam(learningRate float64) *Adam {
	return &Adam{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
	}
}

// Step computes updated parameters using Adam
func (a *Adam) Step(params, gradients []float64) []float64 {
	// Adam maintains state per parameter (m and v moments)
	// For now, use simple SGD - full Adam would require tracking state
	// This is a placeholder for future optimization
	return SGD{LearningRate: a.LearningRate}.Step(params, gradients)
}

// StepInPlace updates params in-place using Adam
func (a *Adam) StepInPlace(params, gradients []float64) {
	SGD{LearningRate: a.LearningRate}.StepInPlace(params, gradients)
}
