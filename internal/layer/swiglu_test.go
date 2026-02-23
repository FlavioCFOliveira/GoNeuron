package layer

import (
	"math"
	"testing"
)

func TestSwiGLU(t *testing.T) {
	in := 4
	out := 2
	s := NewSwiGLU(in, out)

	x := []float32{1, 2, 3, 4}

	// Just check if it runs without panic and produces reasonable output
	y := s.Forward(x)

	if len(y) != out {
		t.Errorf("Expected output length %d, got %d", out, len(y))
	}

	// Test backward
	grad := []float32{0.1, 0.2}
	dx := s.Backward(grad)

	if len(dx) != in {
		t.Errorf("Expected gradient length %d, got %d", in, len(dx))
	}
}

func TestSwiGLUMathematics(t *testing.T) {
	in := 1
	out := 1
	s := NewSwiGLU(in, out)

	// Set parameters manually
	// w=2, v=3
	s.Params()[0] = 2.0
	s.Params()[1] = 3.0

	x := []float32{1.0}
	// gate = x*w = 1*2 = 2
	// up = x*v = 1*3 = 3
	// Swish(2) = 2 * sigmoid(2) = 2 * (1/(1+e^-2)) = 2 * (1/(1+0.1353)) = 2 * 0.880797 = 1.761594
	// SwiGLU = 1.761594 * 3 = 5.284782

	y := s.Forward(x)
	expected := float32(5.284782)

	if math.Abs(float64(y[0]-expected)) > 1e-5 {
		t.Errorf("Expected %f, got %f", expected, y[0])
	}
}
