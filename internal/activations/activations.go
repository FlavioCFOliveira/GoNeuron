// Package activations provides activation functions optimized for performance.
package activations

import "math"

// Activation is an activation function with derivative.
type Activation interface {
	// Activate computes f(x)
	Activate(x float32) float32

	// Derivative computes f'(x)
	Derivative(x float32) float32
}

// BatchActivation is an activation function that can process batches.
type BatchActivation interface {
	// ActivateBatch computes activation for a slice of values.
	ActivateBatch(x []float32) []float32
}

// ReLU activation function.
type ReLU struct{}

// Activate computes max(0, x)
func (r ReLU) Activate(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

// Derivative returns 1 if x > 0, else 0
func (r ReLU) Derivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

// Sigmoid activation function.
type Sigmoid struct{}

// sigmoid computes the sigmoid function
// Inline for performance
func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

// Activate computes sigmoid(x)
func (s Sigmoid) Activate(x float32) float32 {
	return sigmoid(x)
}

// Derivative computes sigmoid(x) * (1 - sigmoid(x))
func (s Sigmoid) Derivative(x float32) float32 {
	sigma := sigmoid(x)
	return sigma * (1 - sigma)
}

// LeakyReLU activation function to prevent dying neurons.
// PyTorch reference: torch.nn.LeakyReLU(negative_slope=0.01)
type LeakyReLU struct {
	Alpha float32 // Slope for x <= 0
}

// NewLeakyReLU creates a LeakyReLU with the given alpha value.
// PyTorch default: negative_slope=0.01
func NewLeakyReLU(alpha float32) *LeakyReLU {
	return &LeakyReLU{Alpha: alpha}
}

// Activate computes x if x > 0, else alpha*x
// PyTorch: f(x) = max(0, x) + alpha * min(0, x)
func (l *LeakyReLU) Activate(x float32) float32 {
	if x > 0 {
		return x
	}
	return l.Alpha * x
}

// Derivative returns 1 if x > 0, else alpha
// PyTorch: f'(x) = 1 if x > 0, alpha if x <= 0
func (l *LeakyReLU) Derivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return l.Alpha
}

// PReLU (Parametric ReLU) activation function.
// PyTorch reference: torch.nn.PReLU(num_parameters=1)
// Unlike LeakyReLU, PReLU has a learnable alpha parameter.
type PReLU struct {
	Alpha float32 // Learnable slope for x <= 0
}

// NewPReLU creates a PReLU with the given initial alpha value.
// PyTorch default: a=0.25
func NewPReLU(alpha float32) *PReLU {
	return &PReLU{Alpha: alpha}
}

// Activate computes x if x > 0, else alpha*x
// Same formula as LeakyReLU but Alpha is typically a learnable parameter
func (p *PReLU) Activate(x float32) float32 {
	if x > 0 {
		return x
	}
	return p.Alpha * x
}

// Derivative returns 1 if x > 0, else alpha
// Same formula as LeakyReLU
func (p *PReLU) Derivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return p.Alpha
}

// GetAlpha returns the current alpha value (useful for gradient descent).
func (p *PReLU) GetAlpha() float32 {
	return p.Alpha
}

// SetAlpha updates the alpha value (used during training).
func (p *PReLU) SetAlpha(alpha float32) {
	p.Alpha = alpha
}

// Tanh activation function.
type Tanh struct{}

// Activate computes tanh(x)
func (t Tanh) Activate(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// Derivative computes 1 - tanh(x)^2
func (t Tanh) Derivative(x float32) float32 {
	tanhX := float32(math.Tanh(float64(x)))
	return 1 - tanhX*tanhX
}

// Softmax activation function for output layer.
type Softmax struct{}

// Activate computes softmax(x) = exp(x) / sum(exp(x))
func (s Softmax) Activate(x float32) float32 {
	panic("Softmax.Activate: use ActivateBatch for Softmax")
}

// Derivative computes softmax(x) * (1 - softmax(x)) for single element
// Note: Softmax derivative is more complex for full vector
func (s Softmax) Derivative(x float32) float32 {
	panic("Softmax.Derivative: use ActivateBatch for Softmax")
}

// ActivateBatch computes softmax for a slice of values.
func (s Softmax) ActivateBatch(x []float32) []float32 {
	// Find max for numerical stability
	maxVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}

	// Compute exp(x - max)
	sum := float32(0.0)
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}

	// Normalize
	for i := range x {
		x[i] /= sum
	}

	return x
}

// LogSoftmax activation function.
// PyTorch reference: torch.nn.LogSoftmax(dim)
// LogSoftmax(x) = log(softmax(x))
type LogSoftmax struct{}

// Activate computes log(softmax(x)) for single value
func (l LogSoftmax) Activate(x float32) float32 {
	panic("LogSoftmax.Activate: use ActivateBatch for LogSoftmax")
}

// Derivative computes 1 - softmax(x) for single element
// Note: LogSoftmax derivative is more complex for full vector
func (l LogSoftmax) Derivative(x float32) float32 {
	panic("LogSoftmax.Derivative: use ActivateBatch for LogSoftmax")
}

// ActivateBatch computes log softmax for a slice of values.
// LogSoftmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
func (l LogSoftmax) ActivateBatch(x []float32) []float32 {
	// Find max for numerical stability
	maxVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}

	// Compute exp(x - max) and sum
	sum := float32(0.0)
	for i := range x {
		sum += float32(math.Exp(float64(x[i] - maxVal)))
	}

	// Compute log sum
	logSum := float32(math.Log(float64(sum)))

	// LogSoftmax(x) = (x - maxVal) - (logSum - maxVal)
	// = x - maxVal - logSum + maxVal
	// = x - logSum
	// But we need to subtract the log of the sum of exp, not add maxVal back
	// Actually: log_softmax(x) = x - log(sum(exp(x)))
	// = x - log(sum(exp(x - maxVal + maxVal)))
	// = x - log(exp(maxVal) * sum(exp(x - maxVal)))
	// = x - (maxVal + log(sum(exp(x - maxVal))))
	// = x - maxVal - logSum
	for i := range x {
		x[i] = x[i] - maxVal - logSum
	}

	return x
}

// GELU (Gaussian Error Linear Unit) activation function.
// PyTorch reference: torch.nn.GELU
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
type GELU struct{}

// Activate computes GELU(x)
func (g GELU) Activate(x float32) float32 {
	// Constants for GELU approximation (PyTorch uses tanh approximation)
	const c = 0.044715
	const sqrt2overpi = 0.7978845608028654 // sqrt(2/pi)

	// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	x3 := x * x * x
	tanhArg := float32(sqrt2overpi) * (x + c*x3)
	tanhVal := float32(math.Tanh(float64(tanhArg)))
	return 0.5 * x * (1 + tanhVal)
}

// Derivative computes GELU'(x) using the analytic derivative of the tanh approximation.
// This matches PyTorch's implementation more closely and is more efficient than numerical differentiation.
func (g GELU) Derivative(x float32) float32 {
	const c = 0.044715
	const sqrt2overpi = 0.7978845608028654 // sqrt(2/pi)

	xf64 := float64(x)
	x2 := xf64 * xf64
	x3 := xf64 * x2
	arg := sqrt2overpi * (xf64 + c*x3)
	tanhVal := math.Tanh(arg)

	// GELU'(x) = 0.5 * (1 + tanh(arg)) + 0.5 * x * (1 - tanhVal^2) * sqrt(2/pi) * (1 + 3 * c * x^2)
	term1 := 0.5 * (1.0 + tanhVal)
	term2 := 0.5 * xf64 * (1.0 - tanhVal*tanhVal) * sqrt2overpi * (1.0 + 3.0*c*x2)
	return float32(term1 + term2)
}

// SiLU (Sigmoid Linear Unit) / Swish activation function.
// PyTorch reference: torch.nn.SiLU
// SiLU(x) = x * sigmoid(x)
type SiLU struct{}

// Activate computes SiLU(x) = x * sigmoid(x)
func (s SiLU) Activate(x float32) float32 {
	// Use math.Sigmoid for better numerical stability if available
	// Otherwise compute directly
	if x >= 0 {
		// For positive x, compute directly
		sig := 1 / (1 + float32(math.Exp(float64(-x))))
		return x * sig
	} else {
		// For negative x, use logit formulation for numerical stability
		// x * sigmoid(x) = x / (1 + exp(-x)) = x * exp(x) / (1 + exp(x))
		expX := float32(math.Exp(float64(x)))
		return x * expX / (1 + expX)
	}
}

// Derivative computes SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
// Simplified: SiLU'(x) = sigmoid(x) + SiLU(x) * (1 - sigmoid(x))
func (s SiLU) Derivative(x float32) float32 {
	var sig float32
	if x >= 0 {
		sig = 1 / (1 + float32(math.Exp(float64(-x))))
	} else {
		expX := float32(math.Exp(float64(x)))
		sig = expX / (1 + expX)
	}
	siluVal := x * sig
	return sig + siluVal*(1-sig)
}

// ELU (Exponential Linear Unit) activation function.
// PyTorch reference: torch.nn.ELU(alpha=1.0)
type ELU struct {
	Alpha float32
}

// NewELU creates an ELU activation with the given alpha value.
// PyTorch default: alpha=1.0
func NewELU(alpha float32) *ELU {
	if alpha == 0 {
		alpha = 1.0 // Default from PyTorch
	}
	return &ELU{Alpha: alpha}
}

// Activate computes ELU(x) = x if x > 0, else alpha*(exp(x) - 1)
func (e ELU) Activate(x float32) float32 {
	if x > 0 {
		return x
	}
	return e.Alpha * (float32(math.Exp(float64(x))) - 1)
}

// Derivative computes ELU'(x) = 1 if x > 0, else alpha*exp(x)
func (e ELU) Derivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return e.Alpha * float32(math.Exp(float64(x)))
}

// Linear (identity) activation function.
// Often used for the output layer in regression or when no activation is desired.
type Linear struct{}

// Activate computes f(x) = x (identity)
func (l Linear) Activate(x float32) float32 {
	return x
}

// Derivative computes f'(x) = 1
func (l Linear) Derivative(x float32) float32 {
	return 1
}

// SELU (Scaled Exponential Linear Unit) activation function.
// PyTorch reference: torch.nn.SELU
// SELU(x) = scale * ELU(x, alpha) where alpha=1.6732632423543772, scale=1.0507009873554805
// These values ensure self-normalizing properties.
type SELU struct{}

// Activate computes SELU(x) = scale * ELU(x, alpha) with fixed parameters
func (s SELU) Activate(x float32) float32 {
	// Fixed parameters for SELU (self-normalizing)
	// Source: PyTorch torch.nn.SELU
	const alpha = 1.6732632423543772
	const scale = 1.0507009873554805

	if x > 0 {
		return float32(scale) * x
	}
	return float32(scale * alpha) * (float32(math.Exp(float64(x))) - 1)
}

// Derivative computes SELU'(x) = scale if x > 0, else scale*alpha*exp(x)
func (s SELU) Derivative(x float32) float32 {
	// Fixed parameters for SELU (self-normalizing)
	const alpha = 1.6732632423543772
	const scale = 1.0507009873554805

	if x > 0 {
		return float32(scale)
	}
	return float32(scale * alpha) * float32(math.Exp(float64(x)))
}
