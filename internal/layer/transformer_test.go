package layer

import (
	"testing"
)

func TestPositionalEncoding(t *testing.T) {
	seqLen := 5
	dim := 8
	pe := NewPositionalEncoding(seqLen, dim)

	input := make([]float32, seqLen*dim)
	for i := range input {
		input[i] = 1.0
	}

	output := pe.Forward(input)
	if len(output) != seqLen*dim {
		t.Errorf("Expected output length %d, got %d", seqLen*dim, len(output))
	}

	// Output should be different from input because of added weights
	allSame := true
	for i := range output {
		if output[i] == input[i] {
			continue
		}
		allSame = false
		break
	}
	if allSame {
		t.Error("Output is same as input, weights were not added")
	}
}

func TestMultiHeadAttention(t *testing.T) {
	dim := 16
	numHeads := 4
	seqLen := 5
	mha := NewMultiHeadAttention(dim, numHeads, seqLen)

	input := make([]float32, seqLen*dim)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	output := mha.Forward(input)
	if len(output) != seqLen*dim {
		t.Errorf("Expected output length %d, got %d", seqLen*dim, len(output))
	}
}

func TestTransformerBlock(t *testing.T) {
	dim := 16
	numHeads := 4
	seqLen := 5
	ffDim := 32
	tb := NewTransformerBlock(dim, numHeads, seqLen, ffDim)

	input := make([]float32, seqLen*dim)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	output := tb.Forward(input)
	if len(output) != seqLen*dim {
		t.Errorf("Expected output length %d, got %d", seqLen*dim, len(output))
	}
}

func TestGlobalAveragePooling1D(t *testing.T) {
	seqLen := 4
	dim := 2
	gap := NewGlobalAveragePooling1D(seqLen, dim)

	// [[1, 2], [3, 4], [5, 6], [7, 8]]
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	// Expected: [(1+3+5+7)/4, (2+4+6+8)/4] = [4, 5]
	expected := []float32{4, 5}

	output := gap.Forward(input)
	if len(output) != dim {
		t.Errorf("Expected output length %d, got %d", dim, len(output))
	}

	for i := range output {
		if output[i] != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], output[i])
		}
	}
}
