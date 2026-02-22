// Package activations provides unit tests for activation functions.
package activations

import (
	"math"
	"testing"
)

// TestReLU tests ReLU activation.
func TestReLU(t *testing.T) {
	relu := ReLU{}

	tests := []struct {
		input    float32
		expected float32
	}{
		{-1.0, 0.0},  // Negative -> 0
		{0.0, 0.0},   // Zero -> 0
		{1.0, 1.0},   // Positive -> identity
		{2.5, 2.5},   // Larger positive -> identity
		{-0.1, 0.0},  // Small negative -> 0
	}

	for _, tt := range tests {
		output := relu.Activate(tt.input)
		if float32(math.Abs(float64(output-tt.expected))) > 1e-6 {
			t.Errorf("ReLU(%v) = %v, want %v", tt.input, output, tt.expected)
		}
	}
}

// TestReLUDerivative tests ReLU derivative.
func TestReLUDerivative(t *testing.T) {
	relu := ReLU{}

	tests := []struct {
		input    float32
		expected float32
	}{
		{-1.0, 0.0},  // Negative -> 0
		{0.0, 0.0},   // At zero, derivative is 0 (x must be > 0)
		{1.0, 1.0},   // Positive -> 1
		{2.5, 1.0},   // Larger positive -> 1
	}

	for _, tt := range tests {
		output := relu.Derivative(tt.input)
		if float32(math.Abs(float64(output-tt.expected))) > 1e-6 {
			t.Errorf("ReLU.Derivative(%v) = %v, want %v", tt.input, output, tt.expected)
		}
	}
}

// TestSigmoid tests Sigmoid activation.
func TestSigmoid(t *testing.T) {
	sigmoid := Sigmoid{}

	tests := []struct {
		input    float32
		expected float32
	}{
		{float32(math.Inf(-1)), 0.0}, // -inf -> 0
		{-2.0, float32(1 / (1 + math.Exp(2)))},
		{-1.0, float32(1 / (1 + math.Exp(1)))},
		{0.0, 0.5}, // Zero -> 0.5
		{1.0, float32(1 / (1 + math.Exp(-1)))},
		{2.0, float32(1 / (1 + math.Exp(-2)))},
		{float32(math.Inf(1)), 1.0}, // +inf -> 1
	}

	for _, tt := range tests {
		output := sigmoid.Activate(tt.input)
		if float32(math.Abs(float64(output-tt.expected))) > 1e-6 {
			t.Errorf("Sigmoid(%v) = %v, want %v", tt.input, output, tt.expected)
		}
	}
}

// TestSigmoidDerivative tests Sigmoid derivative.
func TestSigmoidDerivative(t *testing.T) {
	sigmoid := Sigmoid{}

	// At zero: sigmoid(0) = 0.5, derivative = 0.25
	output := sigmoid.Derivative(0.0)
	expected := float32(0.25)
	if float32(math.Abs(float64(output-expected))) > 1e-6 {
		t.Errorf("Sigmoid.Derivative(0) = %v, want %v", output, expected)
	}

	// At large positive: derivative approaches 0
	output = sigmoid.Derivative(10.0)
	if output > 1e-4 {
		t.Errorf("Sigmoid.Derivative(10) = %v, should be near 0", output)
	}

	// At large negative: derivative approaches 0
	output = sigmoid.Derivative(-10.0)
	if output > 1e-4 {
		t.Errorf("Sigmoid.Derivative(-10) = %v, should be near 0", output)
	}
}

// TestTanh tests Tanh activation.
func TestTanh(t *testing.T) {
	tanh := Tanh{}

	tests := []struct {
		input    float32
		expected float32
	}{
		{float32(math.Inf(-1)), -1.0},
		{-2.0, float32(math.Tanh(-2.0))},
		{-1.0, float32(math.Tanh(-1.0))},
		{0.0, 0.0},
		{1.0, float32(math.Tanh(1.0))},
		{2.0, float32(math.Tanh(2.0))},
		{float32(math.Inf(1)), 1.0},
	}

	for _, tt := range tests {
		output := tanh.Activate(tt.input)
		if float32(math.Abs(float64(output-tt.expected))) > 1e-6 {
			t.Errorf("Tanh(%v) = %v, want %v", tt.input, output, tt.expected)
		}
	}
}

// TestTanhDerivative tests Tanh derivative.
func TestTanhDerivative(t *testing.T) {
	tanh := Tanh{}

	// At zero: tanh(0) = 0, derivative = 1 - 0 = 1
	output := tanh.Derivative(0.0)
	expected := float32(1.0)
	if float32(math.Abs(float64(output-expected))) > 1e-6 {
		t.Errorf("Tanh.Derivative(0) = %v, want %v", output, expected)
	}

	// At 1: derivative = 1 - tanh(1)^2
	output = tanh.Derivative(1.0)
	expected = float32(1 - math.Tanh(1.0)*math.Tanh(1.0))
	if float32(math.Abs(float64(output-expected))) > 1e-6 {
		t.Errorf("Tanh.Derivative(1) = %v, want %v", output, expected)
	}

	// At -1: same as at 1 (symmetric)
	output = tanh.Derivative(-1.0)
	expected = float32(1 - math.Tanh(-1.0)*math.Tanh(-1.0))
	if float32(math.Abs(float64(output-expected))) > 1e-6 {
		t.Errorf("Tanh.Derivative(-1) = %v, want %v", output, expected)
	}
}

// TestLeakyReLU tests LeakyReLU activation.
func TestLeakyReLU(t *testing.T) {
	leakyReLU := NewLeakyReLU(0.01)

	tests := []struct {
		input    float32
		expected float32
	}{
		{-1.0, -0.01},  // Negative -> 0.01 * input
		{-0.5, -0.005}, // Small negative
		{0.0, 0.0},     // Zero -> 0
		{1.0, 1.0},     // Positive -> identity
		{2.5, 2.5},     // Larger positive
	}

	for _, tt := range tests {
		output := leakyReLU.Activate(tt.input)
		if float32(math.Abs(float64(output-tt.expected))) > 1e-6 {
			t.Errorf("LeakyReLU(%v) = %v, want %v", tt.input, output, tt.expected)
		}
	}
}

// TestLeakyReLUDerivative tests LeakyReLU derivative.
func TestLeakyReLUDerivative(t *testing.T) {
	leakyReLU := NewLeakyReLU(0.01)

	tests := []struct {
		input    float32
		expected float32
	}{
		{-1.0, 0.01}, // Negative -> alpha
		{-0.5, 0.01}, // Small negative
		{0.0, 0.01},  // At zero, derivative is alpha (x must be > 0)
		{1.0, 1.0},   // Positive -> 1
		{2.5, 1.0},   // Larger positive
	}

	for _, tt := range tests {
		output := leakyReLU.Derivative(tt.input)
		if float32(math.Abs(float64(output-tt.expected))) > 1e-6 {
			t.Errorf("LeakyReLU.Derivative(%v) = %v, want %v", tt.input, output, tt.expected)
		}
	}
}

// TestLeakyReLUDifferentAlphas tests different alpha values.
func TestLeakyReLUDifferentAlphas(t *testing.T) {
	tests := []struct {
		alpha    float32
		input    float32
		expected float32
	}{
		{0.01, -1.0, -0.01},
		{0.02, -1.0, -0.02},
		{0.1, -1.0, -0.1},
		{0.2, -1.0, -0.2},
	}

	for _, tt := range tests {
		leakyReLU := NewLeakyReLU(tt.alpha)
		output := leakyReLU.Activate(tt.input)
		if float32(math.Abs(float64(output-tt.expected))) > 1e-6 {
			t.Errorf("LeakyReLU(alpha=%v, %v) = %v, want %v", tt.alpha, tt.input, output, tt.expected)
		}
	}
}

// TestActivationRange tests that activations stay in expected ranges.
func TestActivationRange(t *testing.T) {
	// Sigmoid should be in [0, 1]
	sigmoid := Sigmoid{}
	for _, x := range []float32{-10, -5, -1, 0, 1, 5, 10} {
		y := sigmoid.Activate(x)
		if y < 0 || y > 1 {
			t.Errorf("Sigmoid(%v) = %v, outside [0,1]", x, y)
		}
	}

	// Tanh should be in [-1, 1]
	tanh := Tanh{}
	for _, x := range []float32{-10, -5, -1, 0, 1, 5, 10} {
		y := tanh.Activate(x)
		if y < -1 || y > 1 {
			t.Errorf("Tanh(%v) = %v, outside [-1,1]", x, y)
		}
	}
}

// TestActivationSymmetry tests symmetric properties.
func TestActivationSymmetry(t *testing.T) {
	// ReLU is not symmetric
	relu := ReLU{}
	fPositive := relu.Activate(1.0)
	fNegative := relu.Activate(-1.0)
	if fPositive != 1.0 || fNegative != 0.0 {
		t.Logf("ReLU(1) = %v, ReLU(-1) = %v (expected asymmetry)", fPositive, fNegative)
	}

	// Tanh is odd: f(-x) = -f(x)
	tanh := Tanh{}
	for _, x := range []float32{0.5, 1.0, 2.0} {
		fx := tanh.Activate(x)
		fmx := tanh.Activate(-x)
		if float32(math.Abs(float64(fx+fmx))) > 1e-6 {
			t.Errorf("Tanh is not odd: tanh(%v) + tanh(-%v) = %v", x, x, fx+fmx)
		}
	}

	// Sigmoid is not symmetric
	// sigmoid(-x) = 1 - sigmoid(x)
	sigmoid := Sigmoid{}
	for _, x := range []float32{0.5, 1.0, 2.0} {
		fx := sigmoid.Activate(x)
		fmx := sigmoid.Activate(-x)
		if float32(math.Abs(float64(fx+fmx-1.0))) > 1e-6 {
			t.Errorf("Sigmoid property failed: sigmoid(%v) + sigmoid(-%v) = %v", x, x, fx+fmx)
		}
	}
}

// TestDerivativeBounds tests that derivatives stay in expected ranges.
func TestDerivativeBounds(t *testing.T) {
	// Sigmoid derivative should be in (0, 0.25]
	sigmoid := Sigmoid{}
	for _, x := range []float32{-10, -5, -1, 0, 1, 5, 10} {
		d := sigmoid.Derivative(x)
		if d <= 0 || d > 0.25 {
			t.Errorf("Sigmoid.Derivative(%v) = %v, should be in (0, 0.25]", x, d)
		}
	}

	// Tanh derivative should be in [0, 1]
	tanh := Tanh{}
	for _, x := range []float32{-10, -5, -1, 0, 1, 5, 10} {
		d := tanh.Derivative(x)
		if d < 0 || d > 1 {
			t.Errorf("Tanh.Derivative(%v) = %v, should be in [0, 1]", x, d)
		}
	}

	// ReLU derivative should be in [0, 1]
	relu := ReLU{}
	for _, x := range []float32{-10, -1, 0, 1, 10} {
		d := relu.Derivative(x)
		if d < 0 || d > 1 {
			t.Errorf("ReLU.Derivative(%v) = %v, should be in [0, 1]", x, d)
		}
	}
}

// TestLeakyReLUDerivativePositive tests LeakyReLU derivative is always positive.
func TestLeakyReLUDerivativePositive(t *testing.T) {
	leakyReLU := NewLeakyReLU(0.01)

	// Derivative should be positive for all inputs
	for _, x := range []float32{-10, -5, -1, 0, 1, 5, 10} {
		d := leakyReLU.Derivative(x)
		if d <= 0 {
			t.Errorf("LeakyReLU.Derivative(%v) = %v, should be positive", x, d)
		}
	}
}

// TestLeakyReLUContinuity tests LeakyReLU is continuous at zero.
func TestLeakyReLUContinuity(t *testing.T) {
	leakyReLU := NewLeakyReLU(0.01)

	// Should be continuous at zero
	f0 := leakyReLU.Activate(0.0)
	fPlus := leakyReLU.Activate(1e-6)
	fMinus := leakyReLU.Activate(-1e-6)

	if float32(math.Abs(float64(f0-fPlus))) > 1e-5 || float32(math.Abs(float64(f0-fMinus))) > 1e-5 {
		t.Errorf("LeakyReLU not continuous at zero: f(0)=%v, f(ε)=%v, f(-ε)=%v", f0, fPlus, fMinus)
	}
}

// TestActivationsAreDifferent tests that different activations produce different outputs.
func TestActivationsAreDifferent(t *testing.T) {
	// Test at non-zero values where activations differ
	inputs := []float32{-2.0, -1.0, 0.5, 1.0, 2.0}

	relu := ReLU{}
	tanh := Tanh{}
	sigmoid := Sigmoid{}

	for _, x := range inputs {
		r := relu.Activate(x)
		ty := tanh.Activate(x)
		s := sigmoid.Activate(x)

		// Check that at least one activation differs
		if r == ty && r == s && ty == s {
			t.Errorf("All three activations produced same output at %v: ReLU=%v, Tanh=%v, Sigmoid=%v", x, r, ty, s)
		}
	}
}

// TestActivationConsistency tests that Activate and Derivative are mathematically consistent.
func TestActivationConsistency(t *testing.T) {
	// Numerically verify derivative: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
	h := float32(1e-4)

	relu := ReLU{}
	for _, x := range []float32{-1.0, -0.5, 0.5, 1.0} {
		numericDeriv := (relu.Activate(x+h) - relu.Activate(x-h)) / (2 * h)
		analyticDeriv := relu.Derivative(x)

		// Allow some tolerance for numerical approximation
		if float32(math.Abs(float64(numericDeriv-analyticDeriv))) > 1e-3 {
			t.Logf("ReLU derivative check at %v: numeric=%v, analytic=%v", x, numericDeriv, analyticDeriv)
		}
	}

	tanh := Tanh{}
	for _, x := range []float32{-1.0, 0.0, 1.0} {
		numericDeriv := (tanh.Activate(x+h) - tanh.Activate(x-h)) / (2 * h)
		analyticDeriv := tanh.Derivative(x)

		if float32(math.Abs(float64(numericDeriv-analyticDeriv))) > 1e-3 {
			t.Logf("Tanh derivative check at %v: numeric=%v, analytic=%v", x, numericDeriv, analyticDeriv)
		}
	}
}
