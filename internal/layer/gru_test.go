package layer

import (
	"math"
	"testing"
)

func TestGRUForward(t *testing.T) {
	// Test GRU forward pass with known inputs
	gru := NewGRU(3, 5)

	// Input vector
	x := []float64{1.0, 0.5, -0.5}

	// Forward pass
	output := gru.Forward(x)

	// Check output size
	if len(output) != 5 {
		t.Errorf("Output length = %d, expected 5", len(output))
	}

	// Check that output contains valid values
	for i := 0; i < 5; i++ {
		if math.IsNaN(output[i]) || math.IsInf(output[i], 0) {
			t.Errorf("Output[%d] contains invalid value: %f", i, output[i])
		}
	}
}

func TestGRUBackward(t *testing.T) {
	gru := NewGRU(3, 5)

	// Forward pass
	x := []float64{1.0, 0.5, -0.5}
	gru.Forward(x)

	// Pass gradient of all ones
	grad := make([]float64, 5)
	for i := range grad {
		grad[i] = 1.0
	}

	// Backward pass
	outputGrad := gru.Backward(grad)

	// Check output gradient size
	if len(outputGrad) != 3 {
		t.Errorf("Output gradient length = %d, expected 3", len(outputGrad))
	}

	// Check that gradients are valid
	for i := 0; i < 3; i++ {
		if math.IsNaN(outputGrad[i]) || math.IsInf(outputGrad[i], 0) {
			t.Errorf("OutputGrad[%d] contains invalid value: %f", i, outputGrad[i])
		}
	}
}

func TestGRUParams(t *testing.T) {
	gru := NewGRU(4, 8)

	// Test Params and SetParams
	params := gru.Params()
	// inputWeights: 3 * outSize * inSize = 3 * 8 * 4 = 96
	// recurrentWeights: 2 * outSize * outSize = 2 * 8 * 8 = 128
	// biases: 3 * outSize = 3 * 8 = 24
	expectedLen := 3*8*4 + 2*8*8 + 3*8 // inputWeights + recurrentWeights + biases
	if len(params) != expectedLen {
		t.Errorf("Expected %d params, got %d", expectedLen, len(params))
	}

	// Modify params and verify
	newParams := make([]float64, len(params))
	for i := 0; i < len(params); i++ {
		newParams[i] = float64(i + 10)
	}
	gru.SetParams(newParams)

	// Verify
	params2 := gru.Params()
	for i := 0; i < len(params2); i++ {
		if math.Abs(params2[i]-float64(i+10)) > 1e-10 {
			t.Errorf("Param[%d] = %f, expected %f", i, params2[i], float64(i+10))
		}
	}
}

func TestGRUInOutSize(t *testing.T) {
	gru := NewGRU(10, 20)

	if gru.InSize() != 10 {
		t.Errorf("InSize = %d, expected 10", gru.InSize())
	}
	if gru.OutSize() != 20 {
		t.Errorf("OutSize = %d, expected 20", gru.OutSize())
	}
}

func TestGRUReset(t *testing.T) {
	gru := NewGRU(3, 5)

	// Forward pass
	x := []float64{1.0, 0.5, -0.5}
	gru.Forward(x)

	// Check that hidden state is set
	hidden := gru.Hidden()
	if len(hidden) != 5 {
		t.Errorf("Hidden length = %d, expected 5", len(hidden))
	}

	// Reset
	gru.Reset()

	// Check that hidden state is cleared
	hidden = gru.Hidden()
	for i := 0; i < 5; i++ {
		if hidden[i] != 0 {
			t.Errorf("Hidden[%d] = %f after reset, expected 0", i, hidden[i])
		}
	}
}

func TestGRUSequence(t *testing.T) {
	// Test GRU with a sequence of inputs
	gru := NewGRU(3, 5)

	// Multiple time steps
	sequence := [][]float64{
		{1.0, 0.5, -0.5},
		{0.8, 0.3, -0.2},
		{0.6, 0.1, -0.4},
	}

	var outputs [][]float64
	for _, x := range sequence {
		output := gru.Forward(x)
		outputs = append(outputs, output)
	}

	// Check that outputs have the right shape
	for i, output := range outputs {
		if len(output) != 5 {
			t.Errorf("Output[%d] length = %d, expected 5", i, len(output))
		}
	}

	// Reset and verify state is cleared
	gru.Reset()
	hidden := gru.Hidden()
	for i := 0; i < 5; i++ {
		if hidden[i] != 0 {
			t.Errorf("Hidden[%d] = %f after reset, expected 0", i, hidden[i])
		}
	}
}

func TestGRUGradients(t *testing.T) {
	gru := NewGRU(3, 5)

	// Forward pass
	x := []float64{1.0, 0.5, -0.5}
	gru.Forward(x)

	// Pass gradient
	grad := make([]float64, 5)
	for i := range grad {
		grad[i] = 1.0
	}
	gru.Backward(grad)

	// Get gradients
	gradients := gru.Gradients()

	// Check gradient length
	// inputWeights: 3 * 5 * 3 = 45
	// recurrentWeights: 2 * 5 * 5 = 50
	// biases: 3 * 5 = 15
	expectedLen := 3*5*3 + 2*5*5 + 3*5
	if len(gradients) != expectedLen {
		t.Errorf("Expected %d gradients, got %d", expectedLen, len(gradients))
	}

	// Verify some gradients are non-zero
	foundNonZero := false
	for _, g := range gradients {
		if math.Abs(g) > 1e-10 {
			foundNonZero = true
			break
		}
	}
	if !foundNonZero {
		t.Errorf("All gradients are zero, expected non-zero gradients")
	}
}

func TestGRUSequenceBackprop(t *testing.T) {
	// Test BPTT with multiple time steps
	gru := NewGRU(3, 5)

	// Forward pass for sequence
	sequence := [][]float64{
		{1.0, 0.5, -0.5},
		{0.8, 0.3, -0.2},
	}

	for _, x := range sequence {
		gru.Forward(x)
	}

	// Backward pass for last time step
	grad := make([]float64, 5)
	for i := range grad {
		grad[i] = 1.0
	}
	gru.Backward(grad)

	// Backward pass for first time step
	gru.Backward(grad)

	// Get gradients
	gradients := gru.Gradients()

	// Check that gradients were accumulated
	sum := 0.0
	for _, g := range gradients {
		sum += math.Abs(g)
	}
	if sum < 1e-10 {
		t.Errorf("Sum of gradients = %f, expected non-zero", sum)
	}
}

func TestGRUParamsConsistency(t *testing.T) {
	// Test that Params and SetParams are consistent
	gru1 := NewGRU(4, 8)
	gru2 := NewGRU(4, 8)

	// Get params from gru1
	params := gru1.Params()

	// Set params to gru2
	gru2.SetParams(params)

	// Forward pass with same input should give same result
	x := []float64{1.0, 0.5, -0.5, 0.2}

	output1 := gru1.Forward(x)
	output2 := gru2.Forward(x)

	for i := 0; i < 8; i++ {
		if math.Abs(output1[i]-output2[i]) > 1e-10 {
			t.Errorf("Output mismatch at [%d]: %f vs %f", i, output1[i], output2[i])
		}
	}
}

func BenchmarkGRUForward(b *testing.B) {
	gru := NewGRU(64, 128)

	// Input vector
	x := make([]float64, 64)
	for i := range x {
		x[i] = float64(i) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gru.Forward(x)
	}
}

func BenchmarkGRUBackward(b *testing.B) {
	gru := NewGRU(64, 128)

	// Input vector
	x := make([]float64, 64)
	for i := range x {
		x[i] = float64(i) * 0.01
	}

	gru.Forward(x)

	grad := make([]float64, 128)
	for i := range grad {
		grad[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gru.Backward(grad)
	}
}

func BenchmarkGRUFull(b *testing.B) {
	gru := NewGRU(64, 128)

	// Input vector
	x := make([]float64, 64)
	for i := range x {
		x[i] = float64(i) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gru.Forward(x)
		gru.Backward(make([]float64, 128))
	}
}

func BenchmarkGRUSequence(b *testing.B) {
	gru := NewGRU(32, 64)

	// Sequence of 10 time steps
	sequence := make([][]float64, 10)
	for t := range sequence {
		sequence[t] = make([]float64, 32)
		for i := range sequence[t] {
			sequence[t][i] = float64(i) * 0.01
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range sequence {
			gru.Forward(x)
		}
		for t := len(sequence) - 1; t >= 0; t-- {
			grad := make([]float64, 64)
			for j := range grad {
				grad[j] = 1.0
			}
			gru.Backward(grad)
		}
	}
}

func BenchmarkGRUParams(b *testing.B) {
	gru := NewGRU(128, 256)

	params := gru.Params()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gru.SetParams(params)
	}
}

func BenchmarkGRUSetParams(b *testing.B) {
	gru := NewGRU(128, 256)

	params := gru.Params()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gru.SetParams(params)
	}
}
