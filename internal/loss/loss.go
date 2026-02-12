// Package loss provides optimized loss functions.
package loss

import "math"

// Loss is a loss function with derivative.
type Loss interface {
	// Forward computes the loss between predicted and true values.
	Forward(yPred, yTrue []float64) float64

	// Backward computes the gradient of the loss w.r.t. prediction.
	Backward(yPred, yTrue []float64) []float64
}

// MSE (Mean Squared Error) loss.
type MSE struct{}

// Forward computes mean squared error: (1/n) * sum((y_pred - y_true)^2)
func (m MSE) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("MSE: prediction and target must have same length")
	}

	var sum float64
	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		sum += diff * diff
	}
	return sum / float64(n)
}

// Backward computes gradient: dL/dy_pred = (2/n) * (y_pred - y_true)
// Note: Returned slice is newly allocated for safety.
func (m MSE) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("MSE: prediction and target must have same length")
	}

	grad := make([]float64, n)
	factor := 2.0 / float64(n)
	for i := 0; i < n; i++ {
		grad[i] = factor * (yPred[i] - yTrue[i])
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// This avoids allocation when grad slice is pre-allocated.
func (m MSE) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("MSE: slices must have same length")
	}

	factor := 2.0 / float64(n)
	for i := 0; i < n; i++ {
		grad[i] = factor * (yPred[i] - yTrue[i])
	}
}

// CrossEntropy loss for classification.
type CrossEntropy struct{}

// Forward computes cross entropy: -sum(y_true * log(y_pred + eps))
func (c CrossEntropy) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("CrossEntropy: prediction and target must have same length")
	}

	const eps = 1e-10
	var sum float64
	for i := 0; i < n; i++ {
		// Clip prediction to avoid log(0)
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		sum -= yTrue[i] * math.Log(pred)
	}
	return sum / float64(n)
}

// Backward computes gradient for cross entropy with softmax.
// For cross entropy + softmax, gradient simplifies to (y_pred - y_true).
func (c CrossEntropy) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("CrossEntropy: prediction and target must have same length")
	}

	grad := make([]float64, n)
	for i := 0; i < n; i++ {
		grad[i] = yPred[i] - yTrue[i]
	}
	return grad
}

// Huber loss for robust regression.
type Huber struct {
	Delta float64 // Threshold for quadratic/linear transition
}

// NewHuber creates a Huber loss with the given delta.
func NewHuber(delta float64) *Huber {
	return &Huber{Delta: delta}
}

// Forward computes Huber loss.
func (h Huber) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("Huber: prediction and target must have same length")
	}

	var sum float64
	for i := 0; i < n; i++ {
		diff := math.Abs(yPred[i] - yTrue[i])
		if diff <= h.Delta {
			sum += 0.5 * diff * diff
		} else {
			sum += h.Delta * (diff - 0.5*h.Delta)
		}
	}
	return sum / float64(n)
}

// Backward computes gradient for Huber loss.
func (h Huber) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("Huber: prediction and target must have same length")
	}

	grad := make([]float64, n)
	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		if math.Abs(diff) <= h.Delta {
			grad[i] = diff
		} else {
			grad[i] = h.Delta * math.Copysign(1, diff)
		}
	}
	return grad
}
