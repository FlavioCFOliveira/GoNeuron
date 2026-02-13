package layer

import (
	"math"
	"testing"
)

func TestLayerNormForward(t *testing.T) {
	// Test layer normalization with affine transformation
	ln := NewLayerNorm(4, 1e-5, true)

	// Input: [1, 2, 3, 4]
	// Mean = 2.5, Std = sqrt(1.25) ≈ 1.118
	// Normalized: [-1.342, -0.447, 0.447, 1.342]
	// With gamma=1, beta=0: same as normalized
	input := []float64{1, 2, 3, 4}

	output := ln.Forward(input)

	// Manually compute expected values
	mean := 2.5
	std := math.Sqrt(1.25 + 1e-5)
	expected := []float64{
		(1 - mean) / std,
		(2 - mean) / std,
		(3 - mean) / std,
		(4 - mean) / std,
	}

	for i := 0; i < 4; i++ {
		if math.Abs(output[i]-expected[i]) > 1e-5 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], expected[i])
		}
	}
}

func TestLayerNormWithAffine(t *testing.T) {
	// Test with custom gamma and beta
	ln := NewLayerNorm(4, 1e-5, true)

	// Set gamma = [2, 2, 2, 2], beta = [1, 1, 1, 1]
	// After normalization, output = 2 * normalized + 1
	input := []float64{1, 2, 3, 4}
	ln.Forward(input)

	// Get gamma and verify
	gamma := ln.GetGamma()
	if len(gamma) != 4 {
		t.Errorf("Gamma length = %d, expected 4", len(gamma))
	}

	// Default gamma should be all 1s
	for i := 0; i < 4; i++ {
		if math.Abs(gamma[i]-1.0) > 1e-10 {
			t.Errorf("Gamma[%d] = %f, expected 1.0", i, gamma[i])
		}
	}

	// Default beta should be all 0s
	beta := ln.GetBeta()
	for i := 0; i < 4; i++ {
		if math.Abs(beta[i]) > 1e-10 {
			t.Errorf("Beta[%d] = %f, expected 0.0", i, beta[i])
		}
	}
}

func TestLayerNormWithoutAffine(t *testing.T) {
	// Test without affine transformation
	ln := NewLayerNorm(4, 1e-5, false)

	input := []float64{1, 2, 3, 4}
	output := ln.Forward(input)

	// Should be same as normalized
	mean := 2.5
	std := math.Sqrt(1.25 + 1e-5)
	expected := []float64{
		(1 - mean) / std,
		(2 - mean) / std,
		(3 - mean) / std,
		(4 - mean) / std,
	}

	for i := 0; i < 4; i++ {
		if math.Abs(output[i]-expected[i]) > 1e-5 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], expected[i])
		}
	}

	// Should have no parameters
	params := ln.Params()
	if len(params) != 0 {
		t.Errorf("Expected 0 params, got %d", len(params))
	}
}

func TestLayerNormBatch(t *testing.T) {
	// Test with batch input: 2 samples of 4 features each
	ln := NewLayerNorm(4, 1e-5, false)

	// Input: [1, 2, 3, 4, 5, 6, 7, 8] (2 batches of 4)
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8}

	output := ln.Forward(input)

	// Each batch should be normalized independently
	mean1 := 2.5
	std1 := math.Sqrt(1.25 + 1e-5)
	mean2 := 6.5
	std2 := math.Sqrt(1.25 + 1e-5)

	expected := []float64{
		(1 - mean1) / std1, (2 - mean1) / std1, (3 - mean1) / std1, (4 - mean1) / std1,
		(5 - mean2) / std2, (6 - mean2) / std2, (7 - mean2) / std2, (8 - mean2) / std2,
	}

	for i := 0; i < 8; i++ {
		if math.Abs(output[i]-expected[i]) > 1e-5 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], expected[i])
		}
	}
}

func TestLayerNormBackward(t *testing.T) {
	ln := NewLayerNorm(4, 1e-5, false)

	input := []float64{1, 2, 3, 4}
	ln.Forward(input)

	// Pass gradient of all ones
	grad := []float64{1, 1, 1, 1}
	outputGrad := ln.Backward(grad)

	// For layer norm with all ones gradient and no affine,
	// the gradient should sum to zero (since normalization is shift-invariant)
	sumGrad := 0.0
	for i := 0; i < 4; i++ {
		sumGrad += outputGrad[i]
	}

	if math.Abs(sumGrad) > 1e-10 {
		t.Errorf("Sum of gradients = %f, expected ~0", sumGrad)
	}
}

func TestLayerNormParams(t *testing.T) {
	ln := NewLayerNorm(4, 1e-5, true)

	// Test Params and SetParams
	params := ln.Params()
	if len(params) != 8 { // 4 gamma + 4 beta
		t.Errorf("Expected 8 params, got %d", len(params))
	}

	// Modify params
	newParams := make([]float64, 8)
	for i := 0; i < 8; i++ {
		newParams[i] = float64(i + 10)
	}
	ln.SetParams(newParams)

	// Verify
	params2 := ln.Params()
	for i := 0; i < 8; i++ {
		if math.Abs(params2[i]-float64(i+10)) > 1e-10 {
			t.Errorf("Param[%d] = %f, expected %f", i, params2[i], float64(i+10))
		}
	}
}

func TestLayerNormInOutSize(t *testing.T) {
	ln := NewLayerNorm(10, 1e-5, false)

	if ln.InSize() != 10 {
		t.Errorf("InSize = %d, expected 10", ln.InSize())
	}
	if ln.OutSize() != 10 {
		t.Errorf("OutSize = %d, expected 10", ln.OutSize())
	}
}

func TestLayerNormEps(t *testing.T) {
	ln := NewLayerNorm(4, 1e-6, false)

	if ln.GetEps() != 1e-6 {
		t.Errorf("Eps = %f, expected 1e-6", ln.GetEps())
	}
}

func TestLayerNormNumericalStability(t *testing.T) {
	// Test with constant input (zero variance)
	ln := NewLayerNorm(4, 1e-5, false)

	input := []float64{5, 5, 5, 5}
	output := ln.Forward(input)

	// With zero variance, output should be close to beta (0)
	for i := 0; i < 4; i++ {
		if math.Abs(output[i]) > 1e-5 {
			t.Errorf("Output[%d] = %f, expected ~0 for constant input", i, output[i])
		}
	}
}

func BenchmarkLayerNormForward(b *testing.B) {
	ln := NewLayerNorm(1024, 1e-5, true)

	input := make([]float64, 1024)
	for i := range input {
		input[i] = float64(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ln.Forward(input)
	}
}

func BenchmarkLayerNormBackward(b *testing.B) {
	ln := NewLayerNorm(1024, 1e-5, true)

	input := make([]float64, 1024)
	for i := range input {
		input[i] = float64(i)
	}
	ln.Forward(input)

	grad := make([]float64, 1024)
	for i := range grad {
		grad[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ln.Backward(grad)
	}
}
