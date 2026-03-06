// Package layer provides gradient checking tests for layer implementations.
// These tests validate that backward passes compute gradients correctly.
package layer

import (
	"math"
	"testing"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

// TestDenseGradientFlow validates that Dense layer computes non-zero gradients.
func TestDenseGradientFlow(t *testing.T) {
	layer := NewDense(4, 3, activations.Tanh{})

	// Initialize weights
	params := layer.Params()
	rng := NewRNG(42)
	for i := range params {
		params[i] = rng.RandFloat()*2 - 1
	}
	layer.ClearGradients()

	// Input and target
	x := []float32{0.5, -0.3, 0.2, 0.1}

	// Forward pass
	output := layer.Forward(x)

	// Compute loss gradient
	lossGrad := make([]float32, len(output))
	for i := range output {
		lossGrad[i] = 0.1
	}

	// Backward pass
	gradIn := layer.Backward(lossGrad)

	// Check output gradient exists and has correct shape
	if len(gradIn) != len(x) {
		t.Errorf("Dense backward returned wrong shape: %d vs %d", len(gradIn), len(x))
	}

	// Check parameter gradients are non-zero
	paramGrads := layer.Gradients()
	allZero := true
	for _, g := range paramGrads {
		if g != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("Dense layer computed all-zero gradients")
	}
}

// TestConv2DGradientFlow validates Conv2D backward pass.
func TestConv2DGradientFlow(t *testing.T) {
	layer := NewConv2D(1, 2, 3, 1, 1, activations.ReLU{})
	layer.Build(1)

	// Initialize
	params := layer.Params()
	rng := NewRNG(42)
	for i := range params {
		params[i] = rng.RandFloat()*0.5 - 0.25
	}
	layer.ClearGradients()

	// Input: 1 channel, 4x4
	x := make([]float32, 16)
	for i := range x {
		x[i] = rng.RandFloat()*2 - 1
	}

	// Forward and backward
	output := layer.Forward(x)
	lossGrad := make([]float32, len(output))
	for i := range lossGrad {
		lossGrad[i] = 0.1
	}

	gradIn := layer.Backward(lossGrad)

	// Check shapes
	if len(gradIn) != len(x) {
		t.Errorf("Conv2D backward returned wrong shape: %d vs %d", len(gradIn), len(x))
	}

	// Check gradients exist
	paramGrads := layer.Gradients()
	hasNonZero := false
	for _, g := range paramGrads {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Conv2D layer computed all-zero gradients")
	}
}

// TestLSTMGradientFlow validates LSTM backward pass.
// Note: LSTM processes one timestep at a time (input size = inSize, output size = outSize).
func TestLSTMGradientFlow(t *testing.T) {
	inSize, outSize := 3, 2
	layer := NewLSTM(inSize, outSize)

	// Initialize with small values
	params := layer.Params()
	rng := NewRNG(42)
	for i := range params {
		params[i] = (rng.RandFloat()*2 - 1) * 0.1
	}
	layer.ClearGradients()
	layer.Reset()

	// Process 2 timesteps
	for step := 0; step < 2; step++ {
		x := []float32{0.5, -0.3, 0.2}
		output := layer.Forward(x)

		// Check output shape
		if len(output) != outSize {
			t.Errorf("LSTM forward returned wrong shape at step %d: %d vs %d", step, len(output), outSize)
		}

		lossGrad := make([]float32, outSize)
		for i := range lossGrad {
			lossGrad[i] = 0.1
		}

		gradIn := layer.Backward(lossGrad)

		// Backward returns gradient w.r.t input (inSize), not sequence length
		if len(gradIn) != inSize {
			t.Errorf("LSTM backward returned wrong shape at step %d: %d vs %d", step, len(gradIn), inSize)
		}
	}

	// Check gradients exist
	paramGrads := layer.Gradients()
	hasNonZero := false
	for _, g := range paramGrads {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("LSTM layer computed all-zero gradients")
	}
}

// TestGRUGradientFlow validates GRU backward pass.
// Note: GRU processes one timestep at a time (input size = inSize, output size = outSize).
func TestGRUGradientFlow(t *testing.T) {
	inSize, outSize := 3, 2
	layer := NewGRU(inSize, outSize)

	// Initialize
	params := layer.Params()
	rng := NewRNG(42)
	for i := range params {
		params[i] = (rng.RandFloat()*2 - 1) * 0.1
	}
	layer.ClearGradients()
	layer.Reset()

	// Process 2 timesteps
	for step := 0; step < 2; step++ {
		x := []float32{0.5, -0.3, 0.2}
		output := layer.Forward(x)

		// Check output shape
		if len(output) != outSize {
			t.Errorf("GRU forward returned wrong shape at step %d: %d vs %d", step, len(output), outSize)
		}

		lossGrad := make([]float32, outSize)
		for i := range lossGrad {
			lossGrad[i] = 0.1
		}

		gradIn := layer.Backward(lossGrad)

		// Backward returns gradient w.r.t input (inSize), not sequence length
		if len(gradIn) != inSize {
			t.Errorf("GRU backward returned wrong shape at step %d: %d vs %d", step, len(gradIn), inSize)
		}
	}

	// Check gradients exist
	paramGrads := layer.Gradients()
	hasNonZero := false
	for _, g := range paramGrads {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("GRU layer computed all-zero gradients")
	}
}

// TestEmbeddingGradientFlow validates Embedding backward pass.
func TestEmbeddingGradientFlow(t *testing.T) {
	layer := NewEmbedding(10, 4)

	// Initialize
	params := layer.Params()
	rng := NewRNG(42)
	for i := range params {
		params[i] = (rng.RandFloat()*2 - 1) * 0.5
	}
	layer.ClearGradients()

	// Input: token indices
	x := []float32{1, 3, 5}

	// Forward and backward
	output := layer.Forward(x)
	lossGrad := make([]float32, len(output))
	for i := range lossGrad {
		lossGrad[i] = 0.1
	}

	gradIn := layer.Backward(lossGrad)

	// Check shapes
	if len(gradIn) != len(x) {
		t.Errorf("Embedding backward returned wrong shape: %d vs %d", len(gradIn), len(x))
	}

	// Check gradients exist
	paramGrads := layer.Gradients()
	hasNonZero := false
	for _, g := range paramGrads {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Embedding layer computed all-zero gradients")
	}
}

// TestBatchNorm2DGradientFlow validates BatchNorm2D backward pass.
func TestBatchNorm2DGradientFlow(t *testing.T) {
	layer := NewBatchNorm2D(2, 1e-5, 0.1, true)

	// Input: 2 channels, 2x2
	x := []float32{
		1.0, 2.0, 3.0, 4.0,
		-1.0, -2.0, -3.0, -4.0,
	}

	layer.ClearGradients()
	layer.SetTraining(true)

	// Forward and backward
	output := layer.Forward(x)
	lossGrad := make([]float32, len(output))
	for i := range lossGrad {
		lossGrad[i] = 0.1
	}

	gradIn := layer.Backward(lossGrad)

	// Check shapes
	if len(gradIn) != len(x) {
		t.Errorf("BatchNorm2D backward returned wrong shape: %d vs %d", len(gradIn), len(x))
	}

	// BatchNorm has parameters (gamma, beta)
	paramGrads := layer.Gradients()
	if len(paramGrads) == 0 {
		t.Error("BatchNorm2D has no gradients")
	}
}

// TestMaxPool2DGradientFlow validates MaxPool2D backward pass.
func TestMaxPool2DGradientFlow(t *testing.T) {
	layer := NewMaxPool2D(1, 2, 2, 0)
	layer.Build(1)

	// Input: 1 channel, 4x4
	x := []float32{
		1.0, 2.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0,
	}

	// Forward
	output := layer.Forward(x)

	// Target gradients
	lossGrad := make([]float32, len(output))
	for i := range lossGrad {
		lossGrad[i] = 1.0
	}

	// Backward
	gradIn := layer.Backward(lossGrad)

	// Check shape
	if len(gradIn) != len(x) {
		t.Errorf("MaxPool2D backward returned wrong shape: %d vs %d", len(gradIn), len(x))
	}

	// Max pooling should route gradients to max positions
	hasNonZero := false
	for _, g := range gradIn {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("MaxPool2D returned all-zero gradients")
	}
}

// TestAvgPool2DGradientFlow validates AvgPool2D backward pass.
func TestAvgPool2DGradientFlow(t *testing.T) {
	layer := NewAvgPool2D(1, 2, 2, 0)
	layer.Build(1)

	// Input
	x := []float32{
		1.0, 2.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0,
	}

	// Forward and backward
	output := layer.Forward(x)
	lossGrad := make([]float32, len(output))
	for i := range lossGrad {
		lossGrad[i] = 1.0
	}

	gradIn := layer.Backward(lossGrad)

	// Check shape
	if len(gradIn) != len(x) {
		t.Errorf("AvgPool2D backward returned wrong shape: %d vs %d", len(gradIn), len(x))
	}

	// Avg pooling distributes gradients evenly
	hasNonZero := false
	for _, g := range gradIn {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("AvgPool2D returned all-zero gradients")
	}
}

// TestFlattenGradientFlow validates Flatten backward pass.
func TestFlattenGradientFlow(t *testing.T) {
	layer := NewFlatten()
	layer.Build(12)

	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

	// Forward and backward
	output := layer.Forward(x)
	lossGrad := make([]float32, len(output))
	for i := range lossGrad {
		lossGrad[i] = float32(i + 1)
	}

	gradIn := layer.Backward(lossGrad)

	// Check shape
	if len(gradIn) != len(x) {
		t.Errorf("Flatten backward returned wrong shape: %d vs %d", len(gradIn), len(x))
	}

	// Flatten should pass gradients through unchanged
	for i := range gradIn {
		if math.Abs(float64(gradIn[i]-lossGrad[i])) > 1e-6 {
			t.Errorf("Flatten gradient mismatch at %d: %f vs %f", i, gradIn[i], lossGrad[i])
		}
	}
}

// TestDropoutGradientFlow validates Dropout backward pass.
func TestDropoutGradientFlow(t *testing.T) {
	layer := NewDropout(0.5, 10)
	layer.SetTraining(true)

	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	// Forward and backward
	output := layer.Forward(x)
	lossGrad := make([]float32, len(output))
	for i := range lossGrad {
		lossGrad[i] = 1.0
	}

	gradIn := layer.Backward(lossGrad)

	// Check shape
	if len(gradIn) != len(x) {
		t.Errorf("Dropout backward returned wrong shape: %d vs %d", len(gradIn), len(x))
	}
}

// TestTransformerBlockGradientFlow validates Transformer block gradients.
func TestTransformerBlockGradientFlow(t *testing.T) {
	layer := NewTransformerBlock(32, 4, 8, 64, true)

	// Initialize
	params := layer.Params()
	rng := NewRNG(42)
	for i := range params {
		params[i] = (rng.RandFloat()*2 - 1) * 0.1
	}
	layer.ClearGradients()

	// Input: seqLen * dim = 8 * 32
	x := make([]float32, 256)
	for i := range x {
		x[i] = rng.RandFloat()*2 - 1
	}

	// Forward and backward
	output := layer.Forward(x)
	lossGrad := make([]float32, len(output))
	for i := range lossGrad {
		lossGrad[i] = 0.01
	}

	gradIn := layer.Backward(lossGrad)

	// Check shape
	if len(gradIn) != len(x) {
		t.Errorf("TransformerBlock backward returned wrong shape: %d vs %d", len(gradIn), len(x))
	}

	// Check gradients exist
	paramGrads := layer.Gradients()
	hasNonZero := false
	for _, g := range paramGrads {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("TransformerBlock computed all-zero gradients")
	}
}

// TestGradientAccumulation validates that gradients accumulate correctly.
func TestGradientAccumulation(t *testing.T) {
	layer := NewDense(3, 2, activations.Linear{})

	// Initialize
	params := layer.Params()
	rng := NewRNG(42)
	for i := range params {
		params[i] = rng.RandFloat()*2 - 1
	}
	layer.ClearGradients()

	// First forward-backward
	x1 := []float32{0.5, 0.3, 0.2}
	_ = layer.Forward(x1)
	lossGrad1 := []float32{0.1, -0.1}
	layer.Backward(lossGrad1)

	gradsAfterFirst := make([]float32, len(layer.Gradients()))
	copy(gradsAfterFirst, layer.Gradients())

	// Second forward-backward (should accumulate)
	layer.Reset()
	x2 := []float32{-0.2, 0.4, 0.1}
	_ = layer.Forward(x2)
	lossGrad2 := []float32{-0.05, 0.05}
	layer.AccumulateBackward(lossGrad2)

	gradsAfterSecond := layer.Gradients()

	// Gradients should have changed
	same := true
	for i := range gradsAfterFirst {
		if gradsAfterFirst[i] != gradsAfterSecond[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("Gradients did not accumulate")
	}
}

// TestClearGradients validates that ClearGradients works correctly.
func TestClearGradients(t *testing.T) {
	layer := NewDense(3, 2, activations.Linear{})

	// Initialize
	params := layer.Params()
	rng := NewRNG(42)
	for i := range params {
		params[i] = rng.RandFloat()*2 - 1
	}

	// Forward-backward to get gradients
	x := []float32{0.5, 0.3, 0.2}
	_ = layer.Forward(x)
	lossGrad := []float32{0.1, -0.1}
	layer.Backward(lossGrad)

	// Verify gradients exist
	grads := layer.Gradients()
	hasNonZero := false
	for _, g := range grads {
		if g != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Fatal("No gradients computed")
	}

	// Clear gradients
	layer.ClearGradients()

	// Verify all gradients are zero
	grads = layer.Gradients()
	for i, g := range grads {
		if g != 0 {
			t.Errorf("Gradient not cleared at index %d: %f", i, g)
		}
	}
}

// TestGradientShapes validates gradient shapes for all layer types.
func TestGradientShapes(t *testing.T) {
	tests := []struct {
		name             string
		layer            Layer
		inputSize        int
		expectedGradSize int // For recurrent layers, grad size may differ from input size
	}{
		{"Dense", NewDense(4, 3, activations.ReLU{}), 4, 4},
		{"Conv2D", func() Layer { l := NewConv2D(1, 2, 3, 1, 0, activations.ReLU{}); l.Build(1); return l }(), 9, 9},
		// LSTM/GRU process one timestep at a time: input size = inSize, output size = outSize
		// Backward returns gradient w.r.t input (inSize), not the full sequence
		{"LSTM", NewLSTM(3, 2), 3, 3},
		{"GRU", NewGRU(3, 2), 3, 3},
		{"Embedding", NewEmbedding(10, 4), 3, 3},
		{"Flatten", func() Layer { l := NewFlatten(); l.Build(12); return l }(), 12, 12},
		{"Dropout", NewDropout(0.5, 10), 10, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset state for recurrent layers
			if r, ok := tt.layer.(*LSTM); ok {
				r.Reset()
			}
			if r, ok := tt.layer.(*GRU); ok {
				r.Reset()
			}
			tt.layer.ClearGradients()

			// Create input
			x := make([]float32, tt.inputSize)
			rng := NewRNG(42)
			for i := range x {
				x[i] = rng.RandFloat()
			}

			// Forward
			output := tt.layer.Forward(x)

			// Backward
			lossGrad := make([]float32, len(output))
			for i := range lossGrad {
				lossGrad[i] = 0.1
			}

			gradIn := tt.layer.Backward(lossGrad)

			// Check input gradient shape
			expectedSize := tt.expectedGradSize
			if expectedSize == 0 {
				expectedSize = tt.inputSize
			}
			if len(gradIn) != expectedSize {
				t.Errorf("Input gradient shape mismatch: %d vs %d", len(gradIn), expectedSize)
			}

			// Check parameter gradients exist
			paramGrads := tt.layer.Gradients()
			params := tt.layer.Params()

			if len(paramGrads) != len(params) {
				t.Errorf("Parameter gradient count mismatch: %d vs %d", len(paramGrads), len(params))
			}
		})
	}
}
