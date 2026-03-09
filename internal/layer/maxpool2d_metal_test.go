package layer

import (
	"math"
	"testing"
)

func TestMetalMaxPool2D(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	inChannels := 2
	kernelSize := 2
	stride := 2
	padding := 0
	inH, inW := 4, 4

	// Create layers
	poolCPU , _ := NewMaxPool2D(inChannels, kernelSize, stride, padding)
	poolMetal , _ := NewMaxPool2D(inChannels, kernelSize, stride, padding)

	poolMetal.SetDevice(metal)

	// Create input
	input := make([]float32, inChannels*inH*inW)
	rng := NewRNG(42)
	for i := range input {
		input[i] = rng.RandFloat()*2 - 1
	}

	// Forward
	outCPU, _ := poolCPU.Forward(input)
	outMetal, _ := poolMetal.Forward(input)

	if len(outCPU) != len(outMetal) {
		t.Fatalf("Output length mismatch: cpu=%d, metal=%d", len(outCPU), len(outMetal))
	}

	// Compare forward results
	for i := range outCPU {
		if math.Abs(float64(outCPU[i]-outMetal[i])) > 1e-5 {
			t.Errorf("Forward mismatch at index %d: cpu=%f, metal=%f", i, outCPU[i], outMetal[i])
			break
		}
	}

	// Backward
	gradOut := make([]float32, len(outCPU))
	for i := range gradOut {
		gradOut[i] = rng.RandFloat()*2 - 1
	}

	gradInCPU, _ := poolCPU.Backward(gradOut)
	gradInMetal, _ := poolMetal.Backward(gradOut)

	if len(gradInCPU) != len(gradInMetal) {
		t.Fatalf("GradIn length mismatch: cpu=%d, metal=%d", len(gradInCPU), len(gradInMetal))
	}

	// Compare gradIn
	for i := range gradInCPU {
		if math.Abs(float64(gradInCPU[i]-gradInMetal[i])) > 1e-5 {
			t.Errorf("GradIn mismatch at index %d: cpu=%f, metal=%f", i, gradInCPU[i], gradInMetal[i])
			break
		}
	}
}

func TestMetalMaxPool2DBatch(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	batchSize := 1
	inChannels := 2
	kernelSize := 2
	stride := 2
	padding := 0
	inH, inW := 4, 4

	poolCPU , _ := NewMaxPool2D(inChannels, kernelSize, stride, padding)
	poolMetal , _ := NewMaxPool2D(inChannels, kernelSize, stride, padding)

	poolMetal.SetDevice(metal)

	input := make([]float32, batchSize*inChannels*inH*inW)
	rng := NewRNG(123)
	for i := range input {
		input[i] = rng.RandFloat()
	}

	// Forward Batch
	outCPU, _ := poolCPU.ForwardBatch(input, batchSize)
	outMetal, _ := poolMetal.ForwardBatch(input, batchSize)

	if len(outCPU) != len(outMetal) {
		t.Fatalf("Forward batch output length mismatch: cpu=%d, metal=%d", len(outCPU), len(outMetal))
	}

	for i := range outCPU {
		if math.Abs(float64(outCPU[i]-outMetal[i])) > 1e-5 {
			t.Errorf("Forward batch mismatch at index %d: cpu=%f, metal=%f", i, outCPU[i], outMetal[i])
			// Print first 8 elements of both to compare
			t.Logf("CPU:   %v", outCPU[:8])
			t.Logf("Metal: %v", outMetal[:8])
			t.Logf("CPU Argmax:   %v", poolCPU.GetArgmax()[:8])
			t.Logf("Metal Argmax: %v", poolMetal.GetArgmax()[:8])
			break
		}
	}

	// Backward Batch
	gradOut := make([]float32, len(outCPU))
	for i := range gradOut {
		gradOut[i] = rng.RandFloat()
	}

	gradInCPU, _ := poolCPU.BackwardBatch(gradOut, batchSize)
	gradInMetal, _ := poolMetal.BackwardBatch(gradOut, batchSize)

	for i := range gradInCPU {
		if math.Abs(float64(gradInCPU[i]-gradInMetal[i])) > 1e-5 {
			t.Errorf("GradIn batch mismatch at index %d: cpu=%f, metal=%f", i, gradInCPU[i], gradInMetal[i])
			break
		}
	}
}
