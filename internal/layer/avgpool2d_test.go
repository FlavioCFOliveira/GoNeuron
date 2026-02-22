package layer

import (
	"math"
	"testing"
)

func TestAvgPool2DForward(t *testing.T) {
	// Test 2x2 average pooling with stride 2, no padding
	pool := NewAvgPool2D(2, 2, 0)

	// Input: 4x4 = 16 values
	// 1  2  3  4
	// 5  6  7  8
	// 9  10 11 12
	// 13 14 15 16
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	output := pool.Forward(input)

	// Expected output: 2x2 = 4 values
	// (1+2+5+6)/4 = 3.5, (3+4+7+8)/4 = 5.5
	// (9+10+13+14)/4 = 11.5, (11+12+15+16)/4 = 13.5
	expected := []float32{3.5, 5.5, 11.5, 13.5}

	if len(output) != 4 {
		t.Errorf("Output length = %d, expected 4", len(output))
	}

	for i := 0; i < 4; i++ {
		if float32(math.Abs(float64(output[i]-expected[i]))) > 1e-6 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], expected[i])
		}
	}
}

func TestAvgPool2DForwardStride(t *testing.T) {
	// Test with stride different from kernel size
	pool := NewAvgPool2D(2, 1, 0)

	// Input: 3x3 = 9 values
	// 1 2 3
	// 4 5 6
	// 7 8 9
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}

	output := pool.Forward(input)

	// With stride 1 and kernel 2: output = 2x2 = 4
	// Position (0,0): (1+2+4+5)/4 = 3
	// Position (0,1): (2+3+5+6)/4 = 4
	// Position (1,0): (4+5+7+8)/4 = 6
	// Position (1,1): (5+6+8+9)/4 = 7
	expected := []float32{3, 4, 6, 7}

	if len(output) != 4 {
		t.Errorf("Output length = %d, expected 4", len(output))
	}

	for i := 0; i < 4; i++ {
		if float32(math.Abs(float64(output[i]-expected[i]))) > 1e-6 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], expected[i])
		}
	}
}

func TestAvgPool2DForwardPadding(t *testing.T) {
	// Test with padding
	pool := NewAvgPool2D(2, 2, 1)

	// Input: 3x3 = 9 values
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}

	output := pool.Forward(input)

	// With padding 1: effective input is 5x5 (padded with zeros)
	// Output size: (3 + 2*1 - 2) / 2 + 1 = 2
	// So output should be 2x2 = 4
	if len(output) != 4 {
		t.Errorf("Output length = %d, expected 4", len(output))
	}
}

func TestAvgPool2DBackward(t *testing.T) {
	pool := NewAvgPool2D(2, 2, 0)

	// Input: 4x4 = 16 values
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	pool.Forward(input)

	// Pass gradient of all ones
	grad := []float32{1, 1, 1, 1}
	outputGrad := pool.Backward(grad)

	// Each input in a 2x2 pool should get 1/4 of the gradient
	// Position (0,0) contributes to output (0,0): grad = 0.25
	// Position (0,1) contributes to output (0,0): grad = 0.25
	// Position (1,0) contributes to output (0,0): grad = 0.25
	// Position (1,1) contributes to output (0,0): grad = 0.25
	// And so on...

	expected := []float32{
		0.25, 0.25, 0.25, 0.25,
		0.25, 0.25, 0.25, 0.25,
		0.25, 0.25, 0.25, 0.25,
		0.25, 0.25, 0.25, 0.25,
	}

	for i := 0; i < 16; i++ {
		if float32(math.Abs(float64(outputGrad[i]-expected[i]))) > 1e-6 {
			t.Errorf("Grad[%d] = %f, expected %f", i, outputGrad[i], expected[i])
		}
	}
}

func TestAvgPool2DParams(t *testing.T) {
	pool := NewAvgPool2D(2, 2, 0)

	params := pool.Params()
	if len(params) != 0 {
		t.Errorf("Expected 0 params, got %d", len(params))
	}

	gradients := pool.Gradients()
	if len(gradients) != 0 {
		t.Errorf("Expected 0 gradients, got %d", len(gradients))
	}

	pool.SetParams([]float32{1, 2, 3})
	pool.SetGradients([]float32{1, 2, 3})

	newParams := pool.Params()
	if len(newParams) != 0 {
		t.Errorf("Expected 0 params after SetParams, got %d", len(newParams))
	}
}

func TestAvgPool2DInOutSize(t *testing.T) {
	pool := NewAvgPool2D(2, 2, 0)

	// 4x4 input -> 2x2 output
	input := make([]float32, 16)
	for i := range input {
		input[i] = float32(i)
	}
	pool.Forward(input)

	if pool.InSize() != 16 {
		t.Errorf("InSize = %d, expected 16", pool.InSize())
	}
	if pool.OutSize() != 4 {
		t.Errorf("OutSize = %d, expected 4", pool.OutSize())
	}
}

func TestAvgPool2DLargeInput(t *testing.T) {
	pool := NewAvgPool2D(2, 2, 0)

	// 64x64 input
	input := make([]float32, 4096)
	for i := range input {
		input[i] = float32(i)
	}

	for i := 0; i < 100; i++ {
		pool.Forward(input)
	}
}

func TestAvgPool2DBackwardLarge(t *testing.T) {
	pool := NewAvgPool2D(2, 2, 0)

	input := make([]float32, 4096)
	for i := range input {
		input[i] = float32(i)
	}
	pool.Forward(input)

	grad := make([]float32, 1024)
	for i := range grad {
		grad[i] = 1.0
	}

	for i := 0; i < 100; i++ {
		pool.Backward(grad)
	}
}
