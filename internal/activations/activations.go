// Package activations provides activation functions optimized for performance.
package activations

import "math"

// Activation is an activation function with derivative.
type Activation interface {
	// Activate computes f(x)
	Activate(x float64) float64

	// Derivative computes f'(x)
	Derivative(x float64) float64
}

// ReLU activation function.
type ReLU struct{}

// Activate computes max(0, x)
func (r ReLU) Activate(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// Derivative returns 1 if x > 0, else 0
func (r ReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Sigmoid activation function.
type Sigmoid struct{}

// sigmoid computes the sigmoid function
// Inline for performance
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Activate computes sigmoid(x)
func (s Sigmoid) Activate(x float64) float64 {
	return sigmoid(x)
}

// Derivative computes sigmoid(x) * (1 - sigmoid(x))
func (s Sigmoid) Derivative(x float64) float64 {
	sigma := sigmoid(x)
	return sigma * (1 - sigma)
}

// LeakyReLU activation function to prevent dying neurons.
// PyTorch reference: torch.nn.LeakyReLU(negative_slope=0.01)
type LeakyReLU struct {
	Alpha float64 // Slope for x <= 0
}

// NewLeakyReLU creates a LeakyReLU with the given alpha value.
// PyTorch default: negative_slope=0.01
func NewLeakyReLU(alpha float64) *LeakyReLU {
	return &LeakyReLU{Alpha: alpha}
}

// Activate computes x if x > 0, else alpha*x
// PyTorch: f(x) = max(0, x) + alpha * min(0, x)
func (l *LeakyReLU) Activate(x float64) float64 {
	if x > 0 {
		return x
	}
	return l.Alpha * x
}

// Derivative returns 1 if x > 0, else alpha
// PyTorch: f'(x) = 1 if x > 0, alpha if x <= 0
func (l *LeakyReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return l.Alpha
}

// PReLU (Parametric ReLU) activation function.
// PyTorch reference: torch.nn.PReLU(num_parameters=1)
// Unlike LeakyReLU, PReLU has a learnable alpha parameter.
type PReLU struct {
	Alpha float64 // Learnable slope for x <= 0
}

// NewPReLU creates a PReLU with the given initial alpha value.
// PyTorch default: a=0.25
func NewPReLU(alpha float64) *PReLU {
	return &PReLU{Alpha: alpha}
}

// Activate computes x if x > 0, else alpha*x
// Same formula as LeakyReLU but Alpha is typically a learnable parameter
func (p *PReLU) Activate(x float64) float64 {
	if x > 0 {
		return x
	}
	return p.Alpha * x
}

// Derivative returns 1 if x > 0, else alpha
// Same formula as LeakyReLU
func (p *PReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return p.Alpha
}

// GetAlpha returns the current alpha value (useful for gradient descent).
func (p *PReLU) GetAlpha() float64 {
	return p.Alpha
}

// SetAlpha updates the alpha value (used during training).
func (p *PReLU) SetAlpha(alpha float64) {
	p.Alpha = alpha
}

// Tanh activation function.
type Tanh struct{}

// Activate computes tanh(x)
func (t Tanh) Activate(x float64) float64 {
	return math.Tanh(x)
}

// Derivative computes 1 - tanh(x)^2
func (t Tanh) Derivative(x float64) float64 {
	tanhX := math.Tanh(x)
	return 1 - tanhX*tanhX
}

// Softmax activation function for output layer.
type Softmax struct{}

// Activate computes softmax(x) = exp(x) / sum(exp(x))
func (s Softmax) Activate(x float64) float64 {
	panic("Softmax.Activate: use ActivateBatch for Softmax")
}

// Derivative computes softmax(x) * (1 - softmax(x)) for single element
// Note: Softmax derivative is more complex for full vector
func (s Softmax) Derivative(x float64) float64 {
	panic("Softmax.Derivative: use ActivateBatch for Softmax")
}

// ActivateBatch computes softmax for a slice of values.
func (s Softmax) ActivateBatch(x []float64) []float64 {
	// Find max for numerical stability
	maxVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}

	// Compute exp(x - max)
	sum := 0.0
	for i := range x {
		x[i] = math.Exp(x[i] - maxVal)
		sum += x[i]
	}

	// Normalize
	for i := range x {
		x[i] /= sum
	}

	return x
}
