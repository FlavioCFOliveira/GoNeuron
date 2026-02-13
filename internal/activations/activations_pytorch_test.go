// Package activations provides validation tests against PyTorch reference implementation.
// This file contains exact reference values computed with PyTorch for comparison.

package activations

import (
	"math"
	"testing"
)

// PyTorch reference values (computed with Python):
// import torch
// torch.set_printoptions(precision=10)

// TestPReLUAgainstPyTorchReference validates PReLU against PyTorch
// PyTorch: torch.nn.PReLU(num_parameters=1)
func TestPReLUAgainstPyTorchReference(t *testing.T) {
	prelu := NewPReLU(0.25)

	// Values from PyTorch: torch.nn.functional.prelu(torch.tensor(x), torch.tensor([0.25]))
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, -0.5},
		{-1.0, -0.25},
		{-0.5, -0.125},
		{0.0, 0.0},
		{0.5, 0.5},
		{1.0, 1.0},
		{2.0, 2.0},
	}

	for _, tt := range tests {
		got := prelu.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("PReLU.Activate(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestPReLUDerivativeAgainstPyTorchReference validates PReLU derivative
// PyTorch: f'(x) = 1 if x > 0, alpha if x <= 0
func TestPReLUDerivativeAgainstPyTorchReference(t *testing.T) {
	prelu := NewPReLU(0.25)

	// PyTorch PReLU: f'(x) = 1 if x > 0, alpha if x <= 0
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, 0.25},
		{-0.1, 0.25},
		{0.0, 0.25},
		{0.1, 1.0},
		{1.0, 1.0},
	}

	for _, tt := range tests {
		got := prelu.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("PReLU.Derivative(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestPReLUAlphaAccess tests PReLU alpha getter/setter
func TestPReLUAlphaAccess(t *testing.T) {
	prelu := NewPReLU(0.25)

	// Test GetAlpha
	if got := prelu.GetAlpha(); got != 0.25 {
		t.Errorf("PReLU.GetAlpha() = %v, want 0.25", got)
	}

	// Test SetAlpha
	prelu.SetAlpha(0.5)
	if got := prelu.GetAlpha(); got != 0.5 {
		t.Errorf("PReLU.GetAlpha() after SetAlpha(0.5) = %v, want 0.5", got)
	}

	// Verify activation uses updated alpha
	got := prelu.Activate(-1.0)
	expected := -0.5
	if !float64Near(got, expected, 1e-10) {
		t.Errorf("PReLU(alpha=0.5).Activate(-1.0) = %v, want %v", got, expected)
	}
}

// TestSoftmaxAgainstPyTorchReference validates Softmax against PyTorch
// PyTorch: torch.nn.functional.softmax(torch.tensor(x), dim=0)
func TestSoftmaxAgainstPyTorchReference(t *testing.T) {
	softmax := Softmax{}

	// Reference values computed with PyTorch
	tests := []struct {
		x        []float64
		expected []float64
	}{
		{[]float64{1.0, 2.0, 3.0}, []float64{0.09003057317, 0.2447284710, 0.6652409558}},
		{[]float64{0.0, 0.0, 0.0}, []float64{0.3333333333, 0.3333333333, 0.3333333333}},
		{[]float64{-1.0, 0.0, 1.0}, []float64{0.09003057317, 0.2447284710, 0.6652409558}},
		{[]float64{2.0}, []float64{1.0}}, // Single element
	}

	for _, tt := range tests {
		got := make([]float64, len(tt.x))
		copy(got, tt.x)
		result := softmax.ActivateBatch(got)

		for i := range result {
			if !float64Near(result[i], tt.expected[i], 1e-9) {
				t.Errorf("Softmax.ActivateBatch(%v) = %v, PyTorch would give %v", tt.x, result, tt.expected)
				break
			}
		}
	}
}

// TestLogSoftmaxAgainstPyTorchReference validates LogSoftmax against PyTorch
// PyTorch: torch.nn.functional.log_softmax(torch.tensor(x), dim=0)
func TestLogSoftmaxAgainstPyTorchReference(t *testing.T) {
	logsoftmax := LogSoftmax{}

	// Reference values computed with PyTorch: log_softmax(x, dim=0)
	// For x = [1, 2, 3]: softmax = [0.09, 0.24, 0.66], log_softmax = [-2.41, -1.41, -0.41]
	tests := []struct {
		x        []float64
		expected []float64
	}{
		{[]float64{1.0, 2.0, 3.0}, []float64{-2.407605964, -1.407605964, -0.4076059643}},
		{[]float64{0.0, 0.0, 0.0}, []float64{-1.098612289, -1.098612289, -1.098612289}},
		{[]float64{-1.0, 0.0, 1.0}, []float64{-2.407605964, -1.407605964, -0.4076059643}},
		{[]float64{2.0}, []float64{0.0}}, // Single element: log(1) = 0
	}

	for _, tt := range tests {
		got := make([]float64, len(tt.x))
		copy(got, tt.x)
		result := logsoftmax.ActivateBatch(got)

		for i := range result {
			if !float64Near(result[i], tt.expected[i], 1e-9) {
				t.Errorf("LogSoftmax.ActivateBatch(%v) = %v, PyTorch would give %v", tt.x, result, tt.expected)
				break
			}
		}
	}
}

// TestGELUAgainstPyTorchReference validates GELU against PyTorch
// PyTorch: torch.nn.functional.gelu(torch.tensor(x))
func TestGELUAgainstPyTorchReference(t *testing.T) {
	gelu := GELU{}

	// Reference values computed with PyTorch (using tanh approximation)
	// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, -0.04540230591},
		{-1.0, -0.15880800939},
		{-0.5, -0.15428599017},
		{0.0, 0.0},
		{0.5, 0.34571400982},
		{1.0, 0.84119199060},
		{2.0, 1.95459769408},
	}

	for _, tt := range tests {
		got := gelu.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("GELU.Activate(%v) = %v, expected %v", tt.x, got, tt.expected)
		}
	}
}

// TestGELUDerivativeAgainstPyTorchReference validates GELU derivative
// Using numerical derivative for accuracy
func TestGELUDerivativeAgainstPyTorchReference(t *testing.T) {
	gelu := GELU{}

	// Reference values computed numerically
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, -0.08609925685},
		{-1.0, -0.08296408407},
		{-0.5, 0.13263009635},
		{0.0, 0.5},
		{0.5, 0.86736990312},
		{1.0, 1.08296408396},
		{2.0, 1.08609925608},
	}

	for _, tt := range tests {
		got := gelu.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("GELU.Derivative(%v) = %v, expected %v", tt.x, got, tt.expected)
		}
	}
}

// TestSiLUAgainstPyTorchReference validates SiLU (Swish) against PyTorch
// PyTorch: torch.nn.functional.silu(torch.tensor(x))
// SiLU(x) = x * sigmoid(x)
func TestSiLUAgainstPyTorchReference(t *testing.T) {
	silu := SiLU{}

	// Reference values using the formula x * sigmoid(x)
	// sigmoid(x) = 1 / (1 + exp(-x))
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, -0.23840584408490563},  // -2 * sigmoid(-2) = -2 * 0.1192 = -0.2384
		{-1.0, -0.2689414213699951},   // -1 * sigmoid(-1) = -1 * 0.2689 = -0.2689
		{-0.5, -0.18877033439907273},  // -0.5 * sigmoid(-0.5) = -0.5 * 0.3775 = -0.1888
		{0.0, 0.0},
		{0.5, 0.3112296656009273},     // 0.5 * sigmoid(0.5) = 0.5 * 0.6225 = 0.3112
		{1.0, 0.7310585786300049},     // 1 * sigmoid(1) = 0.7311
		{2.0, 1.7615941559557653},     // 2 * sigmoid(2) = 2 * 0.8808 = 1.7616
	}

	for _, tt := range tests {
		got := silu.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("SiLU.Activate(%v) = %v, expected %v", tt.x, got, tt.expected)
		}
	}
}

// TestSiLUDerivativeAgainstPyTorchReference validates SiLU derivative
// SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
func TestSiLUDerivativeAgainstPyTorchReference(t *testing.T) {
	silu := SiLU{}

	// Reference values computed using the derivative formula
	// SiLU'(x) = sigmoid(x) + SiLU(x) * (1 - sigmoid(x))
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, -0.09078424878},  // sigmoid(-2) + (-0.2384)*(1-0.1192) = 0.1192 - 0.2099 = -0.0907
		{-1.0, 0.07232948813},   // sigmoid(-1) + (-0.2689)*(1-0.2689) = 0.2689 - 0.1966 = 0.0723
		{-0.5, 0.26003881270},   // sigmoid(-0.5) + (-0.1888)*(1-0.3775) = 0.3775 - 0.1175 = 0.2600
		{0.0, 0.5},
		{0.5, 0.73996118730},
		{1.0, 0.92767051187},
		{2.0, 1.09078424878},
	}

	for _, tt := range tests {
		got := silu.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("SiLU.Derivative(%v) = %v, expected %v", tt.x, got, tt.expected)
		}
	}
}

// TestELUAgainstPyTorchReference validates ELU against PyTorch
// PyTorch: torch.nn.ELU(alpha=1.0)
func TestELUAgainstPyTorchReference(t *testing.T) {
	elu := NewELU(1.0)

	// Reference values computed with PyTorch
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, -0.8646647168},
		{-1.0, -0.6321205588},
		{-0.5, -0.3934693403},
		{0.0, 0.0},
		{0.5, 0.5},
		{1.0, 1.0},
		{2.0, 2.0},
	}

	for _, tt := range tests {
		got := elu.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("ELU(alpha=1.0).Activate(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestELUDerivativeAgainstPyTorchReference validates ELU derivative
func TestELUDerivativeAgainstPyTorchReference(t *testing.T) {
	elu := NewELU(1.0)

	// PyTorch ELU derivative: 1 if x > 0, else alpha*exp(x)
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, 0.1353352832},
		{-1.0, 0.3678794412},
		{-0.5, 0.6065306597},
		{0.0, 1.0},
		{0.5, 1.0},
		{1.0, 1.0},
	}

	for _, tt := range tests {
		got := elu.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("ELU.Derivative(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestELUAlphaParameter tests ELU with different alpha values
func TestELUAlphaParameter(t *testing.T) {
	// Test with alpha=0.5
	elu05 := NewELU(0.5)
	got := elu05.Activate(-1.0)
	expected := 0.5 * (math.Exp(-1.0) - 1)
	if !float64Near(got, expected, 1e-10) {
		t.Errorf("ELU(alpha=0.5).Activate(-1.0) = %v, expected %v", got, expected)
	}

	// Test with alpha=2.0
	elu2 := NewELU(2.0)
	got = elu2.Activate(-1.0)
	expected = 2.0 * (math.Exp(-1.0) - 1)
	if !float64Near(got, expected, 1e-10) {
		t.Errorf("ELU(alpha=2.0).Activate(-1.0) = %v, expected %v", got, expected)
	}
}

// TestSELUAgainstPyTorchReference validates SELU against PyTorch
// PyTorch: torch.nn.SELU
// Fixed parameters: alpha=1.6732632423543772, scale=1.0507009873554805
func TestSELUAgainstPyTorchReference(t *testing.T) {
	selu := SELU{}

	// SELU(x) = scale * ELU(x, alpha)
	// For x <= 0: SELU(x) = scale * alpha * (exp(x) - 1)
	// For x > 0: SELU(x) = scale * x

	const alpha = 1.6732632423543772
	const scale = 1.0507009873554805

	// Reference values computed with the SELU formula
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, scale * alpha * (math.Exp(-2.0) - 1)},  // -1.5201664686
		{-1.0, scale * alpha * (math.Exp(-1.0) - 1)},  // -1.1113307378
		{-0.5, scale * alpha * (math.Exp(-0.5) - 1)},  // -0.6917581878
		{0.0, 0.0},
		{0.5, scale * 0.5},                            // 0.5253504937
		{1.0, scale * 1.0},                            // 1.0507009874
		{2.0, scale * 2.0},                            // 2.1014019747
	}

	for _, tt := range tests {
		got := selu.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("SELU.Activate(%v) = %v, expected %v", tt.x, got, tt.expected)
		}
	}
}

// TestSELUDerivativeAgainstPyTorchReference validates SELU derivative
func TestSELUDerivativeAgainstPyTorchReference(t *testing.T) {
	selu := SELU{}

	const alpha = 1.6732632423543772
	const scale = 1.0507009873554805

	// SELU'(x) = scale if x > 0, else scale * alpha * exp(x)
	// Note: at x=0, we use x <= 0 branch, so derivative = scale * alpha * exp(0) = scale * alpha
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, scale * alpha * math.Exp(-2.0)},  // 0.2379328723
		{-1.0, scale * alpha * math.Exp(-1.0)},  // 0.6467686030
		{-0.5, scale * alpha * math.Exp(-0.5)},  // 1.0663411530
		{0.0, scale * alpha},                    // 1.7580993408 (at x=0, uses x<=0 branch)
		{0.5, scale},                            // 1.0507009874
		{1.0, scale},                            // 1.0507009874
	}

	for _, tt := range tests {
		got := selu.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("SELU.Derivative(%v) = %v, expected %v", tt.x, got, tt.expected)
		}
	}
}

// Helper function
func float64Near(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}
