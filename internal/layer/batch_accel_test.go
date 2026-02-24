package layer

import (
	"math"
	"testing"
)

func TestAvgPool2DBatch(t *testing.T) {
	// Test both CPU and Metal (if available)
	devices := []Device{&CPUDevice{}}
	metal := NewMetalDevice()
	if metal.IsAvailable() {
		devices = append(devices, metal)
	}

	for _, dev := range devices {
		t.Run(dev.Type().String(), func(t *testing.T) {
			pool := NewAvgPool2D(1, 2, 2, 0)
			pool.SetDevice(dev)

			batchSize := 2
			// Two 4x4 samples
			input := []float32{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
			}

			// Forward Batch
			output := pool.ForwardBatch(input, batchSize)

			expectedSample := []float32{3.5, 5.5, 11.5, 13.5}
			expected := append(expectedSample, expectedSample...)

			if len(output) != len(expected) {
				t.Fatalf("Output length mismatch: got %d, want %d", len(output), len(expected))
			}

			for i := range expected {
				if math.Abs(float64(output[i]-expected[i])) > 1e-5 {
					t.Errorf("Forward mismatch at %d: got %f, want %f", i, output[i], expected[i])
				}
			}

			// Backward Batch
			grad := make([]float32, 4*batchSize)
			for i := range grad {
				grad[i] = 1.0
			}
			gradIn := pool.BackwardBatch(grad, batchSize)

			expectedGradVal := float32(0.25)
			if len(gradIn) != len(input) {
				t.Fatalf("GradIn length mismatch: got %d, want %d", len(gradIn), len(input))
			}

			for i := range gradIn {
				if math.Abs(float64(gradIn[i]-expectedGradVal)) > 1e-5 {
					t.Errorf("Backward mismatch at %d: got %f, want %f", i, gradIn[i], expectedGradVal)
				}
			}
		})
	}
}

func TestDropoutBatch(t *testing.T) {
	devices := []Device{&CPUDevice{}}
	metal := NewMetalDevice()
	if metal.IsAvailable() {
		devices = append(devices, metal)
	}

	for _, dev := range devices {
		t.Run(dev.Type().String(), func(t *testing.T) {
			inSize := 100
			batchSize := 2
			dropout := NewDropout(0.5, inSize)
			dropout.SetDevice(dev)
			dropout.SetTraining(true)

			input := make([]float32, inSize*batchSize)
			for i := range input {
				input[i] = 1.0
			}

			// Use arena to test that path
			arena := make([]float32, 1000)
			offset := 0
			output := dropout.ForwardBatchWithArena(input, batchSize, &arena, &offset)

			if len(output) != inSize*batchSize {
				t.Fatalf("Output length mismatch: got %d, want %d", len(output), inSize*batchSize)
			}

			// Check scaling (1/0.5 = 2.0)
			for _, v := range output {
				if v != 0 && math.Abs(float64(v-2.0)) > 1e-5 {
					t.Errorf("Unexpected value in dropout output: %f", v)
				}
			}

			// Backward
			grad := make([]float32, inSize*batchSize)
			for i := range grad {
				grad[i] = 1.0
			}
			gradIn := dropout.BackwardBatch(grad, batchSize)

			if len(gradIn) != inSize*batchSize {
				t.Fatalf("GradIn length mismatch: got %d, want %d", len(gradIn), inSize*batchSize)
			}

			for i, v := range gradIn {
				if output[i] == 0 {
					if v != 0 {
						t.Errorf("Backward mismatch at %d: output was 0 but gradIn is %f", i, v)
					}
				} else {
					if math.Abs(float64(v-2.0)) > 1e-5 {
						t.Errorf("Backward mismatch at %d: output was %f but gradIn is %f", i, output[i], v)
					}
				}
			}
		})
	}
}
