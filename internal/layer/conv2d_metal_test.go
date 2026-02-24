package layer

import (
	"math"
	"testing"
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

func TestMetalConv2D(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	inChannels := 3
	outChannels := 2
	kernelSize := 3
	stride := 1
	padding := 1
	inH, inW := 8, 8

	// Create layers: one for CPU (reference) and one for Metal
	convCPU := NewConv2D(inChannels, outChannels, kernelSize, stride, padding, activations.Linear{})
	convMetal := NewConv2D(inChannels, outChannels, kernelSize, stride, padding, activations.Linear{})

	// Sync parameters
	convMetal.SetParams(convCPU.Params())
	convMetal.SetDevice(metal)

	// Create input
	input := make([]float32, inChannels*inH*inW)
	rng := NewRNG(42)
	for i := range input {
		input[i] = rng.RandFloat()*2 - 1
	}

	// Forward
	outCPU := convCPU.Forward(input)
	outMetal := convMetal.Forward(input)

	if len(outCPU) != len(outMetal) {
		t.Fatalf("Output length mismatch: cpu=%d, metal=%d", len(outCPU), len(outMetal))
	}

	// Compare forward results
	for i := range outCPU {
		if math.Abs(float64(outCPU[i]-outMetal[i])) > 1e-3 {
			t.Errorf("Forward mismatch at index %d: cpu=%f, metal=%f", i, outCPU[i], outMetal[i])
			break
		}
	}

	// Backward
	gradOut := make([]float32, len(outCPU))
	for i := range gradOut {
		gradOut[i] = rng.RandFloat()*2 - 1
	}

	convCPU.ClearGradients()
	convMetal.ClearGradients()

	gradInCPU := convCPU.Backward(gradOut)
	gradInMetal := convMetal.Backward(gradOut)

	// Compare gradIn
	for i := range gradInCPU {
		if math.Abs(float64(gradInCPU[i]-gradInMetal[i])) > 1e-3 {
			t.Errorf("GradIn mismatch at index %d: cpu=%f, metal=%f", i, gradInCPU[i], gradInMetal[i])
			break
		}
	}

	// Compare parameter gradients
	gradsCPU := convCPU.Gradients()
	gradsMetal := convMetal.Gradients()

	for i := range gradsCPU {
		if math.Abs(float64(gradsCPU[i]-gradsMetal[i])) > 1e-2 { // Slightly higher tolerance for parameter grads
			t.Errorf("Param grad mismatch at index %d: cpu=%f, metal=%f", i, gradsCPU[i], gradsMetal[i])
			break
		}
	}
}

func TestMetalConv2DBatch(t *testing.T) {
	metal := NewMetalDevice()
	if !metal.IsAvailable() {
		t.Skip("Metal not available")
	}

	batchSize := 2
	inChannels := 3
	outChannels := 4
	kernelSize := 3
	stride := 1
	padding := 1
	inH, inW := 10, 10

	convCPU := NewConv2D(inChannels, outChannels, kernelSize, stride, padding, activations.Linear{})
	convMetal := NewConv2D(inChannels, outChannels, kernelSize, stride, padding, activations.Linear{})

	convMetal.SetParams(convCPU.Params())
	convMetal.SetDevice(metal)

	input := make([]float32, batchSize*inChannels*inH*inW)
	rng := NewRNG(123)
	for i := range input {
		input[i] = rng.RandFloat()
	}

	// Forward Batch
	outCPU := convCPU.ForwardBatch(input, batchSize)
	outMetal := convMetal.ForwardBatch(input, batchSize)

	for i := range outCPU {
		if math.Abs(float64(outCPU[i]-outMetal[i])) > 1e-3 {
			t.Errorf("Forward batch mismatch at index %d: cpu=%f, metal=%f", i, outCPU[i], outMetal[i])
			break
		}
	}

	// Backward Batch
	gradOut := make([]float32, len(outCPU))
	for i := range gradOut {
		gradOut[i] = rng.RandFloat()
	}

	convCPU.ClearGradients()
	convMetal.ClearGradients()

	gradInCPU := convCPU.BackwardBatch(gradOut, batchSize)
	gradInMetal := convMetal.BackwardBatch(gradOut, batchSize)

	for i := range gradInCPU {
		if math.Abs(float64(gradInCPU[i]-gradInMetal[i])) > 1e-3 {
			t.Errorf("GradIn batch mismatch at index %d: cpu=%f, metal=%f", i, gradInCPU[i], gradInMetal[i])
			break
		}
	}

	gradsCPU := convCPU.Gradients()
	gradsMetal := convMetal.Gradients()

	for i := range gradsCPU {
		if math.Abs(float64(gradsCPU[i]-gradsMetal[i])) > 1e-2 {
			t.Errorf("Param grad batch mismatch at index %d: cpu=%f, metal=%f", i, gradsCPU[i], gradsMetal[i])
			break
		}
	}
}
