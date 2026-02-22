package layer

import (
	"math"
	"testing"
)

func TestMaxPool2DForward(t *testing.T) {
	// Test 2x2 max pooling with stride 2, no padding (single channel)
	pool := NewMaxPool2D(1, 2, 2, 0)

	// Input: 4x4 = 16 values
	// 1  2  3  4
	// 5  6  7  8
	// 9  10 11 12
	// 13 14 15 16
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	output := pool.Forward(input)

	// Expected output: 2x2 = 4 values
	// max(1,2,5,6) = 6, max(3,4,7,8) = 8
	// max(9,10,13,14) = 14, max(11,12,15,16) = 16
	expected := []float64{6, 8, 14, 16}

	if len(output) != 4 {
		t.Errorf("Output length = %d, expected 4", len(output))
	}

	for i := 0; i < 4; i++ {
		if math.Abs(output[i]-expected[i]) > 1e-10 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], expected[i])
		}
	}
}

func TestMaxPool2DForwardStride(t *testing.T) {
	// Test with stride different from kernel size (single channel)
	pool := NewMaxPool2D(1, 2, 1, 0)

	// Input: 3x3 = 9 values
	// 1 2 3
	// 4 5 6
	// 7 8 9
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}

	output := pool.Forward(input)

	// With stride 1 and kernel 2: output = 2x2 = 4
	// Position (0,0): max(1,2,4,5) = 5
	// Position (0,1): max(2,3,5,6) = 6
	// Position (1,0): max(4,5,7,8) = 8
	// Position (1,1): max(5,6,8,9) = 9
	expected := []float64{5, 6, 8, 9}

	if len(output) != 4 {
		t.Errorf("Output length = %d, expected 4", len(output))
	}

	for i := 0; i < 4; i++ {
		if math.Abs(output[i]-expected[i]) > 1e-10 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], expected[i])
		}
	}
}

func TestMaxPool2DForwardPadding(t *testing.T) {
	// Test with padding (single channel)
	pool := NewMaxPool2D(1, 2, 2, 1)

	// Input: 3x3 = 9 values
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}

	output := pool.Forward(input)

	// With padding 1: effective input is 5x5 (padded with -inf for max)
	// Output size: (3 + 2*1 - 2) / 2 + 1 = 2
	// So output should be 2x2 = 4
	if len(output) != 4 {
		t.Errorf("Output length = %d, expected 4", len(output))
	}
}

func TestMaxPool2DBackward(t *testing.T) {
	// Test backward pass (single channel)
	pool := NewMaxPool2D(1, 2, 2, 0)

	// Input: 4x4 = 16 values
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	pool.Forward(input)

	// Pass gradient of all ones
	grad := []float64{1, 1, 1, 1}
	outputGrad := pool.Backward(grad)

	// Only the max positions should get gradient
	// max(1,2,5,6) = 6 at index 5
	// max(3,4,7,8) = 8 at index 7
	// max(9,10,13,14) = 14 at index 13
	// max(11,12,15,16) = 16 at index 15
	expected := []float64{0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1}

	for i := 0; i < 16; i++ {
		if outputGrad[i] != expected[i] {
			t.Errorf("Grad[%d] = %f, expected %f", i, outputGrad[i], expected[i])
		}
	}
}

func TestMaxPool2DParams(t *testing.T) {
	// Test params (single channel)
	pool := NewMaxPool2D(1, 2, 2, 0)

	params := pool.Params()
	if len(params) != 0 {
		t.Errorf("Expected 0 params, got %d", len(params))
	}

	gradients := pool.Gradients()
	if len(gradients) != 0 {
		t.Errorf("Expected 0 gradients, got %d", len(gradients))
	}

	pool.SetParams([]float64{1, 2, 3})
	pool.SetGradients([]float64{1, 2, 3})

	newParams := pool.Params()
	if len(newParams) != 0 {
		t.Errorf("Expected 0 params after SetParams, got %d", len(newParams))
	}
}

func TestMaxPool2DInOutSize(t *testing.T) {
	// Test in/out sizes (single channel)
	pool := NewMaxPool2D(1, 2, 2, 0)

	// 4x4 input -> 2x2 output
	input := make([]float64, 16)
	for i := range input {
		input[i] = float64(i)
	}
	pool.Forward(input)

	if pool.InSize() != 16 {
		t.Errorf("InSize = %d, expected 16", pool.InSize())
	}
	if pool.OutSize() != 4 {
		t.Errorf("OutSize = %d, expected 4", pool.OutSize())
	}
}

func TestMaxPool2DArgmax(t *testing.T) {
	// Test argmax indices (single channel)
	pool := NewMaxPool2D(1, 2, 2, 0)

	input := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	pool.Forward(input)

	argmax := pool.GetArgmax()

	// Verify argmax indices are correct
	// Position 0 (input indices 0,1,4,5): max at index 5
	// Position 1 (input indices 2,3,6,7): max at index 7
	// Position 2 (input indices 8,9,12,13): max at index 13
	// Position 3 (input indices 10,11,14,15): max at index 15
	expectedArgmax := []int{5, 7, 13, 15}

	for i := 0; i < 4; i++ {
		if argmax[i] != expectedArgmax[i] {
			t.Errorf("Argmax[%d] = %d, expected %d", i, argmax[i], expectedArgmax[i])
		}
	}
}

func TestMaxPool2DLargeInput(t *testing.T) {
	// Test with large input (single channel)
	pool := NewMaxPool2D(1, 2, 2, 0)

	// 64x64 input
	input := make([]float64, 4096)
	for i := range input {
		input[i] = float64(i)
	}

	for i := 0; i < 100; i++ {
		pool.Forward(input)
	}
}

func TestMaxPool2DBackwardLarge(t *testing.T) {
	// Test backward with large input (single channel)
	pool := NewMaxPool2D(1, 2, 2, 0)

	input := make([]float64, 4096)
	for i := range input {
		input[i] = float64(i)
	}
	pool.Forward(input)

	grad := make([]float64, 1024)
	for i := range grad {
		grad[i] = 1.0
	}

	for i := 0; i < 100; i++ {
		pool.Backward(grad)
	}
}
