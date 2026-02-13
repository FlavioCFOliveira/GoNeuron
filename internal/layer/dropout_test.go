package layer

import (
	"math"
	"testing"
)

func TestDropoutForwardTraining(t *testing.T) {
	// Test that dropout zeros out neurons during training
	dropout := NewDropout(0.5, 100)
	dropout.SetTraining(true)

	// Create input with all ones
	input := make([]float64, 100)
	for i := range input {
		input[i] = 1.0
	}

	output := dropout.Forward(input)

	// Count non-zero outputs
	nonZero := 0
	for _, v := range output {
		if v != 0 {
			nonZero++
		}
	}

	// Approximately 50% should be non-zero (with 0.5 dropout)
	// Allow reasonable variance (40-60 range with seed 42)
	if nonZero < 30 || nonZero > 70 {
		t.Errorf("Expected ~50%% non-zero outputs, got %d/100", nonZero)
	}
}

func TestDropoutForwardInference(t *testing.T) {
	// Test that dropout passes inputs through unchanged during inference
	dropout := NewDropout(0.5, 100)
	dropout.SetTraining(false)

	input := make([]float64, 100)
	for i := range input {
		input[i] = float64(i)
	}

	output := dropout.Forward(input)

	// Should be identical to input
	for i := range input {
		if output[i] != input[i] {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], input[i])
		}
	}
}

func TestDropoutBackward(t *testing.T) {
	// Test gradient computation
	dropout := NewDropout(0.5, 10)
	dropout.SetTraining(true)

	// Generate output first
	input := make([]float64, 10)
	for i := range input {
		input[i] = 1.0
	}
	dropout.Forward(input)

	// Pass gradient of all ones
	grad := make([]float64, 10)
	for i := range grad {
		grad[i] = 1.0
	}

	outputGrad := dropout.Backward(grad)

	// Check that gradient flows only through non-zero positions
	keepProb := 1.0 - 0.5
	scale := 1.0 / keepProb

	for i := 0; i < 10; i++ {
		if dropout.maskBuf[i] > 0 {
			// Gradient should be scaled
			if math.Abs(outputGrad[i]-scale) > 1e-10 {
				t.Errorf("Grad[%d] = %f, expected %f (scaled)", i, outputGrad[i], scale)
			}
		} else {
			// Gradient should be zero
			if outputGrad[i] != 0 {
				t.Errorf("Grad[%d] = %f, expected 0 (dropped)", i, outputGrad[i])
			}
		}
	}
}

func TestDropoutParams(t *testing.T) {
	// Test that dropout has no learnable parameters
	dropout := NewDropout(0.5, 10)

	params := dropout.Params()
	if len(params) != 0 {
		t.Errorf("Expected 0 params, got %d", len(params))
	}

	gradients := dropout.Gradients()
	if len(gradients) != 0 {
		t.Errorf("Expected 0 gradients, got %d", len(gradients))
	}

	// SetParams and SetGradients should be no-ops
	dropout.SetParams([]float64{1, 2, 3})
	dropout.SetGradients([]float64{1, 2, 3})

	// Verify no change
	newParams := dropout.Params()
	if len(newParams) != 0 {
		t.Errorf("Expected 0 params after SetParams, got %d", len(newParams))
	}
}

func TestDropoutInOutSize(t *testing.T) {
	dropout := NewDropout(0.5, 50)

	if dropout.InSize() != 50 {
		t.Errorf("InSize = %d, expected 50", dropout.InSize())
	}
	if dropout.OutSize() != 50 {
		t.Errorf("OutSize = %d, expected 50", dropout.OutSize())
	}
}

func TestDropoutReset(t *testing.T) {
	// Test that reset produces same dropout mask
	dropout := NewDropout(0.5, 10)
	dropout.SetTraining(true)

	input := make([]float64, 10)
	for i := range input {
		input[i] = 1.0
	}

	// First forward pass
	dropout.Forward(input)
	mask1 := make([]float64, 10)
	copy(mask1, dropout.maskBuf)

	// Reset and forward again
	dropout.Reset()
	dropout.Forward(input)
	mask2 := make([]float64, 10)
	copy(mask2, dropout.maskBuf)

	// Masks should be identical after reset (deterministic)
	for i := 0; i < 10; i++ {
		if mask1[i] != mask2[i] {
			t.Errorf("Mask mismatch at %d: %f vs %f", i, mask1[i], mask2[i])
		}
	}
}

func TestDropoutPValue(t *testing.T) {
	// Test different dropout probabilities
	for _, p := range []float64{0.0, 0.25, 0.5, 0.75, 1.0} {
		dropout := NewDropout(p, 100)
		dropout.SetTraining(true)

		input := make([]float64, 100)
		for i := range input {
			input[i] = 1.0
		}

		output := dropout.Forward(input)

		nonZero := 0
		for _, v := range output {
			if v != 0 {
				nonZero++
			}
		}

		// With p=0, all should be non-zero
		// With p=1, all should be zero
		// With p=0.5, approximately half
		switch {
		case p == 0:
			if nonZero != 100 {
				t.Errorf("With p=0, expected 100%% non-zero, got %d%%", nonZero)
			}
		case p == 1:
			if nonZero != 0 {
				t.Errorf("With p=1, expected 0%% non-zero, got %d%%", nonZero)
			}
		}
	}
}

func BenchmarkDropoutForwardTraining(b *testing.B) {
	dropout := NewDropout(0.5, 1024)
	dropout.SetTraining(true)

	input := make([]float64, 1024)
	for i := range input {
		input[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dropout.Forward(input)
	}
}

func BenchmarkDropoutForwardInference(b *testing.B) {
	dropout := NewDropout(0.5, 1024)
	dropout.SetTraining(false)

	input := make([]float64, 1024)
	for i := range input {
		input[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dropout.Forward(input)
	}
}

func BenchmarkDropoutBackward(b *testing.B) {
	dropout := NewDropout(0.5, 1024)
	dropout.SetTraining(true)

	input := make([]float64, 1024)
	for i := range input {
		input[i] = 1.0
	}
	dropout.Forward(input)

	grad := make([]float64, 1024)
	for i := range grad {
		grad[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dropout.Backward(grad)
	}
}
