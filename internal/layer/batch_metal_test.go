package layer

import (
	"math"
	"testing"
)

func TestMetalAvgPool2DBatch(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	batchSize := 4
	kernelSize := 2
	stride := 2
	padding := 0
	inputHeight := 4
	inputWidth := 4

	pool := NewAvgPool2D(1, kernelSize, stride, padding)
	pool.SetDevice(metal)

	// Create batch input: batchSize x inputHeight x inputWidth
	inSize := inputHeight * inputWidth
	input := make([]float32, batchSize*inSize)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < inSize; i++ {
			input[b*inSize+i] = float32(b*inSize + i + 1)
		}
	}

	// Forward Batch
	output := pool.ForwardBatch(input, batchSize)

	outH, outW := pool.computeOutputSize(inputHeight, inputWidth)
	outSize := outH * outW
	if len(output) != batchSize*outSize {
		t.Fatalf("Expected output size %d, got %d", batchSize*outSize, len(output))
	}

	// Verify forward
	for b := 0; b < batchSize; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				sum := float32(0)
				count := 0
				for kh := 0; kh < kernelSize; kh++ {
					for kw := 0; kw < kernelSize; kw++ {
						ih := oh*stride + kh - padding
						iw := ow*stride + kw - padding
						if ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth {
							sum += input[b*inSize+ih*inputWidth+iw]
							count++
						}
					}
				}
				expected := sum / float32(count)
				got := output[b*outSize+oh*outW+ow]
				if math.Abs(float64(got-expected)) > 1e-5 {
					t.Errorf("Forward mismatch at batch %d, pos (%d,%d): got %f, want %f", b, oh, ow, got, expected)
				}
			}
		}
	}

	// Backward Batch
	grad := make([]float32, batchSize*outSize)
	for i := range grad {
		grad[i] = 1.0
	}
	gradIn := pool.BackwardBatch(grad, batchSize)

	if len(gradIn) != batchSize*inSize {
		t.Fatalf("Expected gradIn size %d, got %d", batchSize*inSize, len(gradIn))
	}

	// Verify backward
	// For AvgPool2D with 2x2 kernel and 2x2 stride, each input contributes to exactly one output
	// Grad should be 1.0 / count = 1.0 / 4 = 0.25
	expectedGrad := float32(1.0) / float32(kernelSize*kernelSize)
	for i, v := range gradIn {
		if math.Abs(float64(v-expectedGrad)) > 1e-5 {
			t.Errorf("Backward mismatch at %d: got %f, want %f", i, v, expectedGrad)
		}
	}
}

func TestMetalDropoutBatch(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	batchSize := 4
	inSize := 100
	p := float32(0.5)

	dropout := NewDropout(p, inSize)
	dropout.SetDevice(metal)
	dropout.SetTraining(true)

	input := make([]float32, batchSize*inSize)
	for i := range input {
		input[i] = 1.0
	}

	output := dropout.ForwardBatch(input, batchSize)

	if len(output) != batchSize*inSize {
		t.Fatalf("Expected output size %d, got %d", batchSize*inSize, len(output))
	}

	// Check scaling and zeros
	scale := 1.0 / (1.0 - p)
	zeros := 0
	scaled := 0
	for _, v := range output {
		if v == 0 {
			zeros++
		} else if math.Abs(float64(v-scale)) < 1e-5 {
			scaled++
		} else {
			t.Errorf("Unexpected value in dropout output: %f", v)
		}
	}

	if zeros == 0 || scaled == 0 {
		t.Errorf("Dropout didn't seem to work: zeros=%d, scaled=%d", zeros, scaled)
	}

	// Backward
	grad := make([]float32, batchSize*inSize)
	for i := range grad {
		grad[i] = 1.0
	}
	gradIn := dropout.BackwardBatch(grad, batchSize)

	for i, v := range gradIn {
		if output[i] == 0 {
			if v != 0 {
				t.Errorf("Backward mismatch: output was 0 but gradIn is %f at %d", v, i)
			}
		} else {
			if math.Abs(float64(v-scale)) < 1e-5 {
				// OK
			} else {
				t.Errorf("Backward mismatch: output was %f but gradIn is %f at %d", output[i], v, i)
			}
		}
	}
}
