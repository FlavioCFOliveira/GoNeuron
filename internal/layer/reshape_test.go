package layer

import (
	"math"
	"testing"
)

func TestFlattenForward(t *testing.T) {
	flatten := NewFlatten()

	// Test flattening a 2D input [2, 3] -> [6]
	input := []float64{1, 2, 3, 4, 5, 6}
	output := flatten.Forward(input)

	if len(output) != 6 {
		t.Errorf("Output length = %d, expected 6", len(output))
	}

	for i := 0; i < 6; i++ {
		if output[i] != float64(i+1) {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], float64(i+1))
		}
	}
}

func TestFlattenBackward(t *testing.T) {
	flatten := NewFlatten()

	input := []float64{1, 2, 3, 4, 5, 6}
	flatten.Forward(input)

	grad := []float64{1, 1, 1, 1, 1, 1}
	outputGrad := flatten.Backward(grad)

	for i := 0; i < 6; i++ {
		if outputGrad[i] != grad[i] {
			t.Errorf("Grad[%d] = %f, expected %f", i, outputGrad[i], grad[i])
		}
	}
}

func TestFlattenInOutSize(t *testing.T) {
	flatten := NewFlatten()

	input := []float64{1, 2, 3, 4, 5, 6}
	flatten.Forward(input)

	// After forward, outSize should be set
	if flatten.OutSize() != 6 {
		t.Errorf("OutSize = %d, expected 6", flatten.OutSize())
	}
}

func TestFlattenParams(t *testing.T) {
	flatten := NewFlatten()

	params := flatten.Params()
	if len(params) != 0 {
		t.Errorf("Expected 0 params, got %d", len(params))
	}

	gradients := flatten.Gradients()
	if len(gradients) != 0 {
		t.Errorf("Expected 0 gradients, got %d", len(gradients))
	}

	flatten.SetParams([]float64{1, 2, 3})
	flatten.SetGradients([]float64{1, 2, 3})

	newParams := flatten.Params()
	if len(newParams) != 0 {
		t.Errorf("Expected 0 params after SetParams, got %d", len(newParams))
	}
}

func TestFlattenLargeInput(t *testing.T) {
	flatten := NewFlatten()

	// Create a larger input: 4x32x32 = 4096
	input := make([]float64, 4096)
	for i := range input {
		input[i] = float64(i)
	}

	output := flatten.Forward(input)

	if len(output) != 4096 {
		t.Errorf("Output length = %d, expected 4096", len(output))
	}

	if flatten.OutSize() != 4096 {
		t.Errorf("OutSize = %d, expected 4096", flatten.OutSize())
	}

	// Verify values preserved
	for i := 0; i < 4096; i++ {
		if math.Abs(output[i]-float64(i)) > 1e-10 {
			t.Errorf("Value mismatch at %d", i)
		}
	}
}

func BenchmarkFlattenForward(b *testing.B) {
	flatten := NewFlatten()

	// 3x224x224 = 150528 (typical image)
	input := make([]float64, 150528)
	for i := range input {
		input[i] = float64(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		flatten.Forward(input)
	}
}

func BenchmarkFlattenBackward(b *testing.B) {
	flatten := NewFlatten()

	input := make([]float64, 150528)
	for i := range input {
		input[i] = float64(i)
	}
	flatten.Forward(input)

	grad := make([]float64, 150528)
	for i := range grad {
		grad[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		flatten.Backward(grad)
	}
}
