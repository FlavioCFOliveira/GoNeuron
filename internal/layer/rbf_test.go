package layer

import (
	"math"
	"testing"
)

func TestRBFForward(t *testing.T) {
	// 2 inputs, 2 centers, 1 output
	r := NewRBF(2, 2, 1, 1.0)

	// Manually set centers
	// c0 = [0, 0], c1 = [1, 1]
	r.centers[0] = 0; r.centers[1] = 0
	r.centers[2] = 1; r.centers[3] = 1

	// Set weights and biases
	// w = [1, 1], b = [0]
	r.weights[0] = 1; r.weights[1] = 1
	r.biases[0] = 0

	// Test input x = [0, 0]
	// distSq0 = (0-0)^2 + (0-0)^2 = 0
	// distSq1 = (0-1)^2 + (0-1)^2 = 2
	// phi0 = exp(-1.0 * 0) = 1
	// phi1 = exp(-1.0 * 2) = exp(-2) ≈ 0.135335
	// y = 1*1 + 1*exp(-2) + 0 = 1 + exp(-2) ≈ 1.135335
	input := []float64{0, 0}
	output := r.Forward(input)

	expected := 1.0 + math.Exp(-2.0)
	if math.Abs(output[0]-expected) > 1e-6 {
		t.Errorf("output = %v, want %v", output[0], expected)
	}
}

func TestRBFFlow(t *testing.T) {
	r := NewRBF(2, 4, 2, 0.5)
	input := []float64{0.5, -0.5}

	// Forward pass
	out := r.Forward(input)
	if len(out) != 2 {
		t.Errorf("expected output len 2, got %d", len(out))
	}

	// Backward pass
	grad := []float64{0.1, -0.1}
	inGrad := r.Backward(grad)
	if len(inGrad) != 2 {
		t.Errorf("expected input grad len 2, got %d", len(inGrad))
	}

	// Check gradients were accumulated
	grads := r.Gradients()
	expectedLen := 2*4 + 2 + 4*2 // weights + biases + centers
	if len(grads) != expectedLen {
		t.Errorf("expected gradients len %d, got %d", expectedLen, len(grads))
	}

	for _, g := range grads {
		if math.IsNaN(g) {
			t.Errorf("NaN gradient detected")
		}
	}
}

func TestRBFNumericalGradient(t *testing.T) {
	r := NewRBF(2, 3, 1, 1.0)
	input := []float64{0.1, 0.2}

	eps := 1e-7

	// Get analytical gradient w.r.t input
	r.ClearGradients()
	r.Forward(input)
	analyticalInGrad := make([]float64, 2)
	copy(analyticalInGrad, r.Backward([]float64{1.0}))

	// Numerical gradient w.r.t input
	numericalInGrad := make([]float64, 2)
	for i := range input {
		old := input[i]

		input[i] = old + eps
		outPlus := r.Forward(input)[0]

		input[i] = old - eps
		outMinus := r.Forward(input)[0]

		numericalInGrad[i] = (outPlus - outMinus) / (2 * eps)
		input[i] = old
	}

	for i := range analyticalInGrad {
		if math.Abs(analyticalInGrad[i]-numericalInGrad[i]) > 1e-5 {
			t.Errorf("input gradient mismatch at %d: analytical=%v, numerical=%v", i, analyticalInGrad[i], numericalInGrad[i])
		}
	}
}
