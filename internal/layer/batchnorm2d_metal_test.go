package layer

import (
	"math"
	"testing"
)

func TestMetalBatchNorm2D(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	numFeatures := 2
	eps := float32(1e-5)
	momentum := float32(0.1)
	inH, inW := 4, 4

	// Create layers
	bnCPU := NewBatchNorm2D(numFeatures, eps, momentum, true)
	bnMetal := NewBatchNorm2D(numFeatures, eps, momentum, true)

	bnMetal.SetDevice(metal)
	bnMetal.Build(numFeatures)

	// Create input
	input := make([]float32, numFeatures*inH*inW)
	rng := NewRNG(42)
	for i := range input {
		input[i] = rng.RandFloat()*2 - 1
	}

	// Forward
	outCPU := bnCPU.Forward(input)
	outMetal := bnMetal.Forward(input)

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

	gradInCPU := bnCPU.Backward(gradOut)
	gradInMetal := bnMetal.Backward(gradOut)

	if len(gradInCPU) != len(gradInMetal) {
		t.Fatalf("GradIn length mismatch: cpu=%d, metal=%d", len(gradInCPU), len(gradInMetal))
	}

	// Compare gradIn
	for i := range gradInCPU {
		if math.Abs(float64(gradInCPU[i]-gradInMetal[i])) > 1e-4 {
			t.Errorf("GradIn mismatch at index %d: cpu=%f, metal=%f", i, gradInCPU[i], gradInMetal[i])
			break
		}
	}

    // Compare gradGamma and gradBeta
    paramsCPU := bnCPU.Gradients()
    paramsMetal := bnMetal.Gradients()
    for i := range paramsCPU {
        if math.Abs(float64(paramsCPU[i]-paramsMetal[i])) > 1e-4 {
            t.Errorf("Gradient mismatch at index %d: cpu=%f, metal=%f", i, paramsCPU[i], paramsMetal[i])
            break
        }
    }
}

func TestMetalBatchNorm2DBatch(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	batchSize := 2
	numFeatures := 2
	eps := float32(1e-5)
	momentum := float32(0.1)
	inH, inW := 4, 4

	bnCPU := NewBatchNorm2D(numFeatures, eps, momentum, true)
	bnMetal := NewBatchNorm2D(numFeatures, eps, momentum, true)
	bnMetal.SetDevice(metal)

	input := make([]float32, batchSize*numFeatures*inH*inW)
	rng := NewRNG(123)
	for i := range input {
		input[i] = rng.RandFloat()
	}

	outCPU := bnCPU.ForwardBatch(input, batchSize)
	outMetal := bnMetal.ForwardBatch(input, batchSize)

	if len(outCPU) != len(outMetal) {
		t.Fatalf("Forward batch output length mismatch: cpu=%d, metal=%d", len(outCPU), len(outMetal))
	}

	for i := range outCPU {
		if math.Abs(float64(outCPU[i]-outMetal[i])) > 1e-5 {
			t.Errorf("Forward batch mismatch at index %d: cpu=%f, metal=%f", i, outCPU[i], outMetal[i])
			break
		}
	}

	gradOut := make([]float32, len(outCPU))
	for i := range gradOut {
		gradOut[i] = rng.RandFloat()
	}

	gradInCPU := bnCPU.BackwardBatch(gradOut, batchSize)
	gradInMetal := bnMetal.BackwardBatch(gradOut, batchSize)

	for i := range gradInCPU {
		if math.Abs(float64(gradInCPU[i]-gradInMetal[i])) > 1e-4 {
			t.Errorf("GradIn batch mismatch at index %d: cpu=%f, metal=%f", i, gradInCPU[i], gradInMetal[i])
			break
		}
	}
}
