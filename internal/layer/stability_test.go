// Package layer provides numerical stability tests for layer implementations.
package layer

import (
	"math"
	"testing"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

// TestDenseNumericalStability tests Dense layer with extreme values.
func TestDenseNumericalStability(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{"Very large values", []float32{1e10, 1e10, 1e10, 1e10}},
		{"Very small values", []float32{1e-10, 1e-10, 1e-10, 1e-10}},
		{"Mixed signs large", []float32{1e10, -1e10, 1e10, -1e10}},
		{"Near zero", []float32{1e-15, 1e-15, 1e-15, 1e-15}},
		{"Gradual increase", []float32{1, 10, 100, 1000}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l , _ := NewDense(4, 3, activations.Tanh{})

			// Initialize weights with small values for stability
			params := l.Params()
			rng := NewRNG(42)
			for i := range params {
				params[i] = (rng.RandFloat()*2 - 1) * 0.1
			}
			l.ClearGradients()

			// Forward pass should not panic or return NaN/Inf
			output, err := l.Forward(tt.input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Check for NaN or Inf
			for i, v := range output {
				if math.IsNaN(float64(v)) {
					t.Errorf("Output[%d] is NaN", i)
				}
				if math.IsInf(float64(v), 0) {
					t.Errorf("Output[%d] is Inf", i)
				}
			}

			// Backward pass
			lossGrad := make([]float32, len(output))
			for i := range lossGrad {
				lossGrad[i] = 0.1
			}

			gradIn, err := l.Backward(lossGrad)
			if err != nil {
				t.Fatalf("Backward failed: %v", err)
			}

			// Check gradients for NaN or Inf
			for i, v := range gradIn {
				if math.IsNaN(float64(v)) {
					t.Errorf("GradIn[%d] is NaN", i)
				}
				if math.IsInf(float64(v), 0) {
					t.Errorf("GradIn[%d] is Inf", i)
				}
			}
		})
	}
}

// TestConv2DNumericalStability tests Conv2D with extreme values.
func TestConv2DNumericalStability(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{"Large values", make([]float32, 16)},
		{"Small values", make([]float32, 16)},
	}

	// Fill test inputs
	for i := range tests[0].input {
		tests[0].input[i] = 1e6
	}
	for i := range tests[1].input {
		tests[1].input[i] = 1e-8
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l , _ := NewConv2D(1, 2, 3, 1, 1, activations.ReLU{})
			l.Build(1)

			// Initialize with small weights
			params := l.Params()
			rng := NewRNG(42)
			for i := range params {
				params[i] = (rng.RandFloat()*2 - 1) * 0.1
			}
			l.ClearGradients()

			// Forward
			output, err := l.Forward(tt.input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Check for NaN/Inf
			for i, v := range output {
				if math.IsNaN(float64(v)) {
					t.Errorf("Output[%d] is NaN", i)
				}
				if math.IsInf(float64(v), 0) {
					t.Errorf("Output[%d] is Inf", i)
				}
			}
		})
	}
}

// TestLSTMNumericalStability tests LSTM with extreme values.
func TestLSTMNumericalStability(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{"Large values", []float32{1e5, 1e5, 1e5}},
		{"Small values", []float32{1e-8, 1e-8, 1e-8}},
		{"Mixed", []float32{1e5, 1e-8, 1e5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l , _ := NewLSTM(3, 2)

			// Initialize with small weights
			params := l.Params()
			rng := NewRNG(42)
			for i := range params {
				params[i] = (rng.RandFloat()*2 - 1) * 0.05
			}
			l.ClearGradients()
			l.Reset()

			// Process multiple timesteps to test state accumulation
			for step := 0; step < 3; step++ {
				output, err := l.Forward(tt.input)
				if err != nil {
					t.Fatalf("Forward failed at step %d: %v", step, err)
				}

				// Check for NaN/Inf
				for i, v := range output {
					if math.IsNaN(float64(v)) {
						t.Errorf("Step %d: Output[%d] is NaN", step, i)
					}
					if math.IsInf(float64(v), 0) {
						t.Errorf("Step %d: Output[%d] is Inf", step, i)
					}
				}
			}
		})
	}
}

// TestBatchNormNumericalStability tests BatchNorm with extreme values.
func TestBatchNormNumericalStability(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{"Large values", []float32{1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8}},
		{"Small values", []float32{1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10}},
		{"Near constant", []float32{1.0, 1.00001, 0.99999, 1.0, 1.0, 1.00001, 0.99999, 1.0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l , _ := NewBatchNorm2D(2, 1e-5, 0.1, true)
			l.ClearGradients()
			l.SetTraining(true)

			// Forward
			output, err := l.Forward(tt.input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Check for NaN/Inf
			for i, v := range output {
				if math.IsNaN(float64(v)) {
					t.Errorf("Output[%d] is NaN", i)
				}
				if math.IsInf(float64(v), 0) {
					t.Errorf("Output[%d] is Inf", i)
				}
			}

			// Backward
			lossGrad := make([]float32, len(output))
			for i := range lossGrad {
				lossGrad[i] = 0.1
			}

			gradIn, err := l.Backward(lossGrad)
			if err != nil {
				t.Fatalf("Backward failed: %v", err)
			}

			// Check gradients
			for i, v := range gradIn {
				if math.IsNaN(float64(v)) {
					t.Errorf("GradIn[%d] is NaN", i)
				}
				if math.IsInf(float64(v), 0) {
					t.Errorf("GradIn[%d] is Inf", i)
				}
			}
		})
	}
}

// TestActivationsNumericalStability tests activation functions with extreme values.
func TestActivationsNumericalStability(t *testing.T) {
	activations := []struct {
		name string
		act  activations.Activation
	}{
		{"Sigmoid", activations.Sigmoid{}},
		{"Tanh", activations.Tanh{}},
		{"ReLU", activations.ReLU{}},
		{"GELU", activations.GELU{}},
		{"LeakyReLU", activations.NewLeakyReLU(0.01)},
	}

	extremeValues := []float32{1e10, -1e10, 1e-10, -1e-10, 0}

	for _, act := range activations {
		t.Run(act.name, func(t *testing.T) {
			for _, v := range extremeValues {
				output := act.act.Activate(v)

				if math.IsNaN(float64(output)) {
					t.Errorf("%s(%v) is NaN", act.name, v)
				}

				// Check derivative
				deriv := act.act.Derivative(v)
				if math.IsNaN(float64(deriv)) {
					t.Errorf("%s.Derivative(%v) is NaN", act.name, v)
				}
			}
		})
	}
}

// TestDivisionByZeroProtection tests protection against division by zero.
func TestDivisionByZeroProtection(t *testing.T) {
	t.Run("BatchNorm with zero variance", func(t *testing.T) {
		l , _ := NewBatchNorm2D(2, 1e-5, 0.1, true)

		// Input with zero variance (all same values)
		x := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}

		l.SetTraining(true)
		output, err := l.Forward(x)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		// Should not produce NaN due to epsilon
		for i, v := range output {
			if math.IsNaN(float64(v)) {
				t.Errorf("Output[%d] is NaN - epsilon protection failed", i)
			}
		}
	})

	t.Run("LayerNorm with single element", func(t *testing.T) {
		l , _ := NewLayerNorm(1, 1e-5, true)

		// Single element input
		x := []float32{5.0}

		output, err := l.Forward(x)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		// Should not produce NaN
		if math.IsNaN(float64(output[0])) {
			t.Error("Output is NaN - single element case failed")
		}
	})

	t.Run("RMSNorm with zero input", func(t *testing.T) {
		l := NewRMSNorm(4, 1e-5)

		// All zeros input
		x := []float32{0, 0, 0, 0}

		output, err := l.Forward(x)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		// Should not produce NaN
		for i, v := range output {
			if math.IsNaN(float64(v)) {
				t.Errorf("Output[%d] is NaN - zero input protection failed", i)
			}
		}
	})
}

// TestEdgeCases tests various edge cases.
func TestEdgeCases(t *testing.T) {
	t.Run("Empty input Dense", func(t *testing.T) {
		l , _ := NewDense(0, 2, activations.ReLU{})
		if l != nil {
			t.Error("Dense should return nil for zero input size")
		}
	})

	t.Run("Single element Dense", func(t *testing.T) {
		l , _ := NewDense(1, 1, activations.ReLU{})
		x := []float32{0.5}

		output, err := l.Forward(x)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		if len(output) != 1 {
			t.Errorf("Expected 1 output, got %d", len(output))
		}
	})

	t.Run("Large batch simulation", func(t *testing.T) {
		l , _ := NewDense(100, 50, activations.ReLU{})

		// Large input
		x := make([]float32, 100)
		rng := NewRNG(42)
		for i := range x {
			x[i] = rng.RandFloat()
		}

		output, err := l.Forward(x)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		if len(output) != 50 {
			t.Errorf("Expected 50 outputs, got %d", len(output))
		}

		// Check all outputs are valid
		for i, v := range output {
			if math.IsNaN(float64(v)) {
				t.Errorf("Output[%d] is NaN", i)
			}
		}
	})
}
