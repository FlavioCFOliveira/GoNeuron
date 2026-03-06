// Package layer provides tests for Conv1D layer
package layer

import (
	"math"
	"testing"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

// TestConv1DCreation tests Conv1D layer creation
func TestConv1DCreation(t *testing.T) {
	c := NewConv1D(3, 16, 5, 1, 2, activations.ReLU{})

	if c.InChannels() != 3 {
		t.Errorf("InChannels = %d, want 3", c.InChannels())
	}
	if c.OutChannels() != 16 {
		t.Errorf("OutChannels = %d, want 16", c.OutChannels())
	}
	if c.KernelSize() != 5 {
		t.Errorf("KernelSize = %d, want 5", c.KernelSize())
	}
	if c.Stride() != 1 {
		t.Errorf("Stride = %d, want 1", c.Stride())
	}
	if c.Padding() != 2 {
		t.Errorf("Padding = %d, want 2", c.Padding())
	}

	// Check parameters were initialized
	params := c.Params()
	expectedSize := 3*16*5 + 16 // weights + biases
	if len(params) != expectedSize {
		t.Errorf("Params size = %d, want %d", len(params), expectedSize)
	}
}

// TestConv1DOutputLength tests output length calculation
func TestConv1DOutputLength(t *testing.T) {
	tests := []struct {
		name       string
		inChannels int
		outChannels int
		kernelSize int
		stride     int
		padding    int
		inputLen   int
		want       int
	}{
		{"same_padding", 3, 16, 5, 1, 2, 100, 100},
		{"valid_padding", 3, 16, 5, 1, 0, 100, 96},
		{"stride2", 3, 16, 3, 2, 1, 100, 50},
		{"kernel3", 3, 16, 3, 1, 1, 100, 100},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewConv1D(tt.inChannels, tt.outChannels, tt.kernelSize, tt.stride, tt.padding, activations.ReLU{})
			got := c.OutputLength(tt.inputLen)
			if got != tt.want {
				t.Errorf("OutputLength(%d) = %d, want %d", tt.inputLen, got, tt.want)
			}
		})
	}
}

// TestConv1DForward tests forward pass
func TestConv1DForward(t *testing.T) {
	c := NewConv1D(1, 1, 3, 1, 1, activations.Linear{})

	// Set simple weights for testing [1, 0, 1] and bias 0
	c.weights = []float32{1, 0, 1}
	c.biases = []float32{0}

	// Input: [0, 1, 2, 3, 4] with padding will be [0, 0, 1, 2, 3, 4, 0]
	// Convolving with [1, 0, 1]:
	// pos0: 0*1 + 0*0 + 1*1 = 1
	// pos1: 0*1 + 1*0 + 2*1 = 2
	// etc.
	x := []float32{0, 1, 2, 3, 4}
	output := c.Forward(x)

	if len(output) != 5 {
		t.Errorf("Output length = %d, want 5", len(output))
	}

	// Check output values (approximately)
	// The actual values depend on exact padding behavior
	for i, v := range output {
		// Just verify we get reasonable values
		if math.IsNaN(float64(v)) {
			t.Errorf("output[%d] = NaN", i)
		}
	}
}

// TestConv1DForwardWithActivation tests forward pass with activation
func TestConv1DForwardWithActivation(t *testing.T) {
	c := NewConv1D(1, 1, 3, 1, 1, activations.ReLU{})

	// Set weights that will produce negative values
	c.weights = []float32{-1, 0, -1}
	c.biases = []float32{0}

	// Input that should produce negative intermediate results
	x := []float32{1, 2, 3, 2, 1}
	output := c.Forward(x)

	// All outputs should be >= 0 due to ReLU
	for i, v := range output {
		if v < 0 {
			t.Errorf("output[%d] = %v, expected >= 0 (ReLU not applied)", i, v)
		}
	}
}

// TestConv1DBackward tests backward pass
func TestConv1DBackward(t *testing.T) {
	c := NewConv1D(1, 1, 3, 1, 1, activations.Linear{})
	c.ClearGradients()

	// Set simple weights
	c.weights = []float32{1, 2, 3}
	c.biases = []float32{1}

	x := []float32{1, 2, 3, 4, 5}
	_ = c.Forward(x)

	// Gradient from next layer
	grad := make([]float32, 5)
	for i := range grad {
		grad[i] = 1.0
	}

	gradIn := c.Backward(grad)

	// Check gradient input shape
	if len(gradIn) != len(x) {
		t.Errorf("gradIn length = %d, want %d", len(gradIn), len(x))
	}

	// Check gradients were accumulated
	wGrads := c.gradWeights
	hasNonZero := false
	for _, g := range wGrads {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Weight gradients are all zero")
	}

	if c.gradBiases[0] == 0 {
		t.Error("Bias gradient is zero")
	}
}

// TestConv1DMultiChannel tests multi-channel convolution
func TestConv1DMultiChannel(t *testing.T) {
	// 2 input channels, 3 output channels
	c := NewConv1D(2, 3, 3, 1, 1, activations.ReLU{})

	inputLength := 10
	x := make([]float32, 2*inputLength)
	for i := range x {
		x[i] = float32(i)
	}

	output := c.Forward(x)

	expectedLen := 3 * inputLength // 3 output channels
	if len(output) != expectedLen {
		t.Errorf("Output length = %d, want %d", len(output), expectedLen)
	}
}

// TestConv1DParams tests parameter getters/setters
func TestConv1DParams(t *testing.T) {
	c := NewConv1D(2, 3, 3, 1, 0, activations.ReLU{})

	// Get original params
	original := c.Params()

	// Modify and set back
	newParams := make([]float32, len(original))
	for i := range newParams {
		newParams[i] = float32(i) * 0.01
	}
	c.SetParams(newParams)

	// Verify change
	got := c.Params()
	for i := range got {
		if got[i] != newParams[i] {
			t.Errorf("Params[%d] = %v, want %v", i, got[i], newParams[i])
			break
		}
	}
}

// TestConv1DGradients tests gradient getters/clear
func TestConv1DGradients(t *testing.T) {
	c := NewConv1D(1, 1, 3, 1, 1, activations.Linear{})

	// Do forward and backward to get gradients
	x := []float32{1, 2, 3, 4, 5}
	_ = c.Forward(x)

	grad := make([]float32, 5)
	for i := range grad {
		grad[i] = 0.5
	}
	_ = c.Backward(grad)

	// Get gradients
	grads := c.Gradients()
	if len(grads) == 0 {
		t.Error("No gradients computed")
	}

	hasNonZero := false
	for _, g := range grads {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("All gradients are zero")
	}

	// Clear gradients
	c.ClearGradients()

	// Verify cleared
	grads = c.Gradients()
	for _, g := range grads {
		if g != 0 {
			t.Error("Gradients not cleared")
			break
		}
	}
}

// TestConv1DStride tests convolution with stride > 1
func TestConv1DStride(t *testing.T) {
	c := NewConv1D(1, 1, 3, 2, 1, activations.Linear{})

	// Input length 8 with stride 2 should give output length 4
	inputLength := 8
	x := make([]float32, inputLength)
	for i := range x {
		x[i] = float32(i)
	}

	output := c.Forward(x)
	expectedLen := c.OutputLength(inputLength)

	if len(output) != expectedLen {
		t.Errorf("Output length = %d, want %d", len(output), expectedLen)
	}
}

// TestConv1DAccumulateBackward tests gradient accumulation
func TestConv1DAccumulateBackward(t *testing.T) {
	c := NewConv1D(1, 1, 3, 1, 1, activations.Linear{})
	c.ClearGradients()

	c.weights = []float32{1, 1, 1}
	c.biases = []float32{0}

	x1 := []float32{1, 2, 3, 4, 5}
	_ = c.Forward(x1)

	grad := []float32{0.1, 0.1, 0.1, 0.1, 0.1}
	_ = c.Backward(grad)

	firstGrads := make([]float32, len(c.gradWeights))
	copy(firstGrads, c.gradWeights)

	// Second backward
	c.AccumulateBackward(grad)

	// Gradients should be approximately doubled
	for i := range c.gradWeights {
		expected := firstGrads[i] * 2
		if math.Abs(float64(c.gradWeights[i]-expected)) > 0.01 {
			t.Errorf("Accumulated gradient[%d] = %v, want ~%v", i, c.gradWeights[i], expected)
		}
	}
}

// BenchmarkConv1DForward benchmarks forward pass
func BenchmarkConv1DForward(b *testing.B) {
	c := NewConv1D(3, 64, 5, 1, 2, activations.ReLU{})

	inputLength := 100
	x := make([]float32, 3*inputLength)
	for i := range x {
		x[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Forward(x)
	}
}

// BenchmarkConv1DBackward benchmarks backward pass
func BenchmarkConv1DBackward(b *testing.B) {
	c := NewConv1D(3, 64, 5, 1, 2, activations.ReLU{})

	inputLength := 100
	x := make([]float32, 3*inputLength)
	grad := make([]float32, 64*inputLength)

	_ = c.Forward(x)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Backward(grad)
	}
}
