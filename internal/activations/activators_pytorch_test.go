// Package activations provides validation tests against PyTorch reference implementation.
// This file contains exact reference values computed with PyTorch for comparison.
// Run: python -c "import torch; torch.set_printoptions(precision=10); x = torch.tensor([...]); print('activation:', f(x))"

package activations

import (
	"math"
	"testing"
)

// PyTorch reference values (computed with Python):
// import torch
// torch.set_printoptions(precision=10)

// TestReLUAgainstPyTorchReference validates ReLU against PyTorch
func TestReLUAgainstPyTorchReference(t *testing.T) {
	relu := ReLU{}

	// Values from PyTorch: torch.relu(torch.tensor(x))
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, 0.0},
		{-1.5, 0.0},
		{-1.0, 0.0},
		{-0.5, 0.0},
		{0.0, 0.0},
		{0.5, 0.5},
		{1.0, 1.0},
		{1.5, 1.5},
		{2.0, 2.0},
	}

	for _, tt := range tests {
		got := relu.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("ReLU(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestReLUDerivativeAgainstPyTorchReference validates ReLU derivative against PyTorch
// PyTorch ReLU: f'(x) = 1 if x > 0, else 0 (for x=0, PyTorch uses 0)
func TestReLUDerivativeAgainstPyTorchReference(t *testing.T) {
	relu := ReLU{}

	// PyTorch behavior: derivative is 0 for x <= 0, 1 for x > 0
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, 0.0},
		{-0.1, 0.0},
		{0.0, 0.0},  // PyTorch uses 0 for x <= 0
		{0.1, 1.0},
		{1.0, 1.0},
	}

	for _, tt := range tests {
		got := relu.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("ReLU.Derivative(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestLeakyReLUAgainstPyTorchReference validates LeakyReLU against PyTorch
func TestLeakyReLUAgainstPyTorchReference(t *testing.T) {
	leakyRelu := NewLeakyReLU(0.01)

	// Values from PyTorch: torch.nn.functional.leaky_relu(torch.tensor(x), 0.01)
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, -0.02},
		{-1.0, -0.01},
		{-0.5, -0.005},
		{0.0, 0.0},
		{0.5, 0.5},
		{1.0, 1.0},
		{2.0, 2.0},
	}

	for _, tt := range tests {
		got := leakyRelu.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("LeakyReLU(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestLeakyReLUDerivativeAgainstPyTorchReference validates LeakyReLU derivative
func TestLeakyReLUDerivativeAgainstPyTorchReference(t *testing.T) {
	leakyRelu := NewLeakyReLU(0.01)

	// PyTorch LeakyReLU: f'(x) = 1 if x > 0, alpha if x <= 0
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, 0.01},
		{-0.1, 0.01},
		{0.0, 0.01}, // PyTorch uses alpha for x <= 0
		{0.1, 1.0},
		{1.0, 1.0},
	}

	for _, tt := range tests {
		got := leakyRelu.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("LeakyReLU.Derivative(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestTanhAgainstPyTorchReference validates Tanh against PyTorch
func TestTanhAgainstPyTorchReference(t *testing.T) {
	tanh := Tanh{}

	// Values from PyTorch: torch.tanh(torch.tensor(x))
	tests := []struct {
		x        float64
		expected float64
	}{
		{-3.0, -0.9950547536867306},
		{-2.0, -0.9640275800758169},
		{-1.0, -0.7615941559557649},
		{-0.5, -0.46211715726000974},
		{0.0, 0.0},
		{0.5, 0.46211715726000974},
		{1.0, 0.7615941559557649},
		{2.0, 0.9640275800758169},
		{3.0, 0.9950547536867306},
	}

	for _, tt := range tests {
		got := tanh.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("Tanh(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestTanhDerivativeAgainstPyTorchReference validates Tanh derivative
// PyTorch: d/dx tanh(x) = 1 - tanh(x)^2
func TestTanhDerivativeAgainstPyTorchReference(t *testing.T) {
	tanh := Tanh{}

	// Values from PyTorch: 1 - torch.tanh(x)^2
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, 0.07065082484702827},
		{-1.0, 0.4199743416140261},
		{-0.5, 0.7864477325343538},
		{0.0, 1.0},
		{0.5, 0.7864477325343538},
		{1.0, 0.4199743416140261},
		{2.0, 0.07065082484702827},
	}

	for _, tt := range tests {
		got := tanh.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-9) {
			t.Errorf("Tanh.Derivative(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestSigmoidAgainstPyTorchReference validates Sigmoid against PyTorch
func TestSigmoidAgainstPyTorchReference(t *testing.T) {
	sigmoid := Sigmoid{}

	// Values from PyTorch: torch.sigmoid(torch.tensor(x))
	tests := []struct {
		x        float64
		expected float64
	}{
		{-3.0, 0.04742587317474628},
		{-2.0, 0.11920292202211755},
		{-1.0, 0.2689414213699951},
		{-0.5, 0.3775406687981454},
		{0.0, 0.5},
		{0.5, 0.6224593312018546},
		{1.0, 0.7310585786300049},
		{2.0, 0.8807970779778823},
		{3.0, 0.9525741268252538},
	}

	for _, tt := range tests {
		got := sigmoid.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("Sigmoid(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestSigmoidDerivativeAgainstPyTorchReference validates Sigmoid derivative
// PyTorch: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
func TestSigmoidDerivativeAgainstPyTorchReference(t *testing.T) {
	sigmoid := Sigmoid{}

	// Values from PyTorch: sigmoid(x) * (1 - sigmoid(x))
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, 0.10499358540350658},
		{-1.0, 0.19661193324148185},
		{-0.5, 0.2350037122015942},
		{0.0, 0.25},
		{0.5, 0.2350037122015942},
		{1.0, 0.19661193324148185},
		{2.0, 0.10499358540350658},
	}

	for _, tt := range tests {
		got := sigmoid.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("Sigmoid.Derivative(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestSigmoidIdentity verifies sigmoid(x) + sigmoid(-x) = 1 (PyTorch property)
func TestSigmoidIdentity(t *testing.T) {
	sigmoid := Sigmoid{}

	// PyTorch: sigmoid(-x) = 1 - sigmoid(x)
	for _, x := range []float64{-3, -2, -1, -0.5, 0.5, 1, 2, 3} {
		fx := sigmoid.Activate(x)
		fmx := sigmoid.Activate(-x)
		sum := fx + fmx
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("Sigmoid identity failed: sigmoid(%v) + sigmoid(-%v) = %v (expected 1)", x, x, sum)
		}
	}
}

// TestTanhOddProperty verifies tanh(-x) = -tanh(x) (PyTorch property)
func TestTanhOddProperty(t *testing.T) {
	tanh := Tanh{}

	// PyTorch: tanh is odd: tanh(-x) = -tanh(x)
	for _, x := range []float64{-3, -2, -1, -0.5, 0.5, 1, 2, 3} {
		fx := tanh.Activate(x)
		fmx := tanh.Activate(-x)
		if math.Abs(fx+fmx) > 1e-10 {
			t.Errorf("Tanh odd property failed: tanh(%v) + tanh(-%v) = %v (expected 0)", x, x, fx+fmx)
		}
	}
}

// Helper function
func float64Near(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}
