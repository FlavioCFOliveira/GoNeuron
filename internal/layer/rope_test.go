package layer

import (
	"math"
	"testing"
)

func TestRoPE(t *testing.T) {
	seqLen := 2
	numHeads := 1
	headDim := 4
	x := []float32{
		1, 2, 3, 4, // t=0
		1, 2, 3, 4, // t=1
	}

	// For t=0, rotation is 0, so should be identity
	xCopy := make([]float32, len(x))
	copy(xCopy, x)

	applyRoPE(xCopy, seqLen, numHeads, headDim, false)

	// t=0
	for i := 0; i < 4; i++ {
		if xCopy[i] != x[i] {
			t.Errorf("t=0: expected %f, got %f at index %d", x[i], xCopy[i], i)
		}
	}

	// t=1
	// theta_0 = 10000^(0) = 1
	// theta_1 = 10000^(-2/4) = 10000^(-0.5) = 1/100 = 0.01
	// idx1=4, idx2=6 (i=0) -> pairs (x[4], x[6]) = (1, 3)
	// idx1=5, idx2=7 (i=1) -> pairs (x[5], x[7]) = (2, 4)

	theta0 := float32(1.0)
	c0 := float32(math.Cos(float64(theta0)))
	s0 := float32(math.Sin(float64(theta0)))

	expected4 := 1*c0 - 3*s0
	expected6 := 1*s0 + 3*c0

	if math.Abs(float64(xCopy[4]-expected4)) > 1e-5 {
		t.Errorf("t=1, i=0: expected %f, got %f", expected4, xCopy[4])
	}
	if math.Abs(float64(xCopy[6]-expected6)) > 1e-5 {
		t.Errorf("t=1, i=0: expected %f, got %f", expected6, xCopy[6])
	}
}

func TestRoPEInversion(t *testing.T) {
	seqLen := 10
	numHeads := 4
	headDim := 32
	x := make([]float32, seqLen*numHeads*headDim)
	for i := range x {
		x[i] = float32(i)
	}

	xCopy := make([]float32, len(x))
	copy(xCopy, x)

	// Apply forward
	applyRoPE(xCopy, seqLen, numHeads, headDim, false)
	// Apply backward (conjugate)
	applyRoPE(xCopy, seqLen, numHeads, headDim, true)

	for i := range x {
		if math.Abs(float64(xCopy[i]-x[i])) > 1e-3 {
			t.Errorf("Inversion failed at index %d: expected %f, got %f", i, x[i], xCopy[i])
		}
	}
}
