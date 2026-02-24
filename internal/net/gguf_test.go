package net

import (
	"os"
	"testing"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

func TestSaveGGUF(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(10, 20, activations.ReLU{}),
		layer.NewDense(20, 5, activations.Softmax{}),
	}
	n := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	filename := "test_model.gguf"
	defer os.Remove(filename)

	if err := n.SaveGGUF(filename); err != nil {
		t.Fatalf("Failed to save GGUF: %v", err)
	}

	// Verify file existence and minimal size
	info, err := os.Stat(filename)
	if err != nil {
		t.Fatalf("Failed to stat GGUF file: %v", err)
	}
	if info.Size() < 100 {
		t.Errorf("GGUF file too small: %d bytes", info.Size())
	}

	// Try saving as F16
	filenameF16 := "test_model_f16.gguf"
	defer os.Remove(filenameF16)
	if err := n.SaveGGUFExt(filenameF16, GGMLTypeF16); err != nil {
		t.Fatalf("Failed to save GGUF F16: %v", err)
	}
}
