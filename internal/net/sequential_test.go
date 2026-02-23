package net

import (
	"testing"
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

func TestSequentialAPI(t *testing.T) {
	// 1. Creation
	model := NewSequential(
		layer.NewDense(2, 4, activations.ReLU{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	)

	if len(model.layers) != 2 {
		t.Errorf("Expected 2 layers, got %d", len(model.layers))
	}

	// 2. Compile
	model.Compile(opt.NewAdam(0.01), loss.MSE{})
	if model.opt == nil {
		t.Error("Optimizer should be set after Compile")
	}
	if model.loss == nil {
		t.Error("Loss should be set after Compile")
	}

	// 3. Predict
	input := []float32{0.5, 0.5}
	pred := model.Predict(input)
	if len(pred) != 1 {
		t.Errorf("Expected output length 1, got %d", len(pred))
	}

	// 4. Fit
	x := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := [][]float32{{0}, {1}, {1}, {0}}

	// Just test it runs without error
	model.Fit(x, y, 2, 1)

	// 5. Evaluate
	l := model.Evaluate(x, y)
	if l < 0 {
		t.Errorf("Loss should be non-negative, got %f", l)
	}

	// 6. Summary
	// This just checks for panics
	model.Summary()
}
