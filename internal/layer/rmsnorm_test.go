package layer

import (
	"math"
	"testing"
)

func TestRMSNorm(t *testing.T) {
	dim := 4
	eps := float32(1e-5)
	ln := NewRMSNorm(dim, eps)

	x := []float32{1, 2, 3, 4}

	// Manual calculation
	// RMS = sqrt((1^2 + 2^2 + 3^2 + 4^2)/4 + eps)
	// RMS = sqrt(30/4 + 1e-5) = sqrt(7.50001) ≈ 2.738616
	sumSq := float32(1*1 + 2*2 + 3*3 + 4*4)
	expectedRMS := float32(math.Sqrt(float64(sumSq/4.0 + eps)))

	y := ln.Forward(x)

	for i := range x {
		expected := x[i] / expectedRMS
		if math.Abs(float64(y[i]-expected)) > 1e-5 {
			t.Errorf("At index %d: expected %f, got %f", i, expected, y[i])
		}
	}

	// Test backward
	grad := []float32{0.1, 0.2, 0.3, 0.4}
	dx := ln.Backward(grad)

	if len(dx) != dim {
		t.Errorf("Expected gradient length %d, got %d", dim, len(dx))
	}
}

func TestRMSNormBackward(t *testing.T) {
	dim := 8
	eps := float32(1e-5)
	l := NewRMSNorm(dim, eps)

	x := make([]float32, dim)
	for i := range x {
		x[i] = float32(i + 1)
	}

	_ = l.Forward(x)

	grad := make([]float32, dim)
	for i := range grad {
		grad[i] = 1.0
	}

	dx := l.Backward(grad)

	// Numerical gradient check would be better but for now just check it's not zero
	hasNonZero := false
	for _, val := range dx {
		if val != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Backward produced all zeros")
	}
}
