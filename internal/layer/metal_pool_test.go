package layer

import (
	"math"
	"testing"
)

func TestMetalAvgPool2D(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	pool := NewAvgPool2D(1, 2, 2, 0)
	pool.SetDevice(metal)

	// 4x4 input
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	// Forward Batch
	output := pool.ForwardBatch(input, 1)

	expected := []float32{3.5, 5.5, 11.5, 13.5}
	for i := range expected {
		if math.Abs(float64(output[i]-expected[i])) > 1e-5 {
			t.Errorf("Forward mismatch at %d: got %f, want %f", i, output[i], expected[i])
		}
	}

	// Backward Batch
	grad := []float32{1, 1, 1, 1}
	gradIn := pool.BackwardBatch(grad, 1)

	expectedGrad := float32(0.25)
	for i := range gradIn {
		if math.Abs(float64(gradIn[i]-expectedGrad)) > 1e-5 {
			t.Errorf("Backward mismatch at %d: got %f, want %f", i, gradIn[i], expectedGrad)
		}
	}
}

func TestMetalDropout(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	dropout := NewDropout(0.5, 100)
	dropout.SetDevice(metal)
	dropout.SetTraining(true)

	input := make([]float32, 100)
	for i := range input {
		input[i] = 1.0
	}

	output := dropout.ForwardBatch(input, 1)

	// Check if some are zero and others are scaled (1/0.5 = 2.0)
	zeros := 0
	twos := 0
	for _, v := range output {
		if v == 0 {
			zeros++
		} else if math.Abs(float64(v-2.0)) < 1e-5 {
			twos++
		} else {
			t.Errorf("Unexpected value in dropout output: %f", v)
		}
	}

	if zeros == 0 || twos == 0 {
		t.Errorf("Dropout didn't seem to work: zeros=%d, twos=%d", zeros, twos)
	}

	// Backward
	grad := make([]float32, 100)
	for i := range grad {
		grad[i] = 1.0
	}
	gradIn := dropout.BackwardBatch(grad, 1)

	for i, v := range gradIn {
		if output[i] == 0 {
			if v != 0 {
				t.Errorf("Backward mismatch: output was 0 but gradIn is %f", v)
			}
		} else {
			if math.Abs(float64(v-2.0)) < 1e-5 {
				// OK
			} else {
				t.Errorf("Backward mismatch: output was %f but gradIn is %f", output[i], v)
			}
		}
	}
}
