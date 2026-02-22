package layer

import (
	"testing"
)

func TestBidirectionalForward(t *testing.T) {
	inSize := 3
	outSize := 5
	f := NewLSTM(inSize, outSize)
	bi := NewBidirectional(f)

	input := []float64{1.0, 0.5, -0.5}
	output := bi.Forward(input)

	if len(output) != outSize*2 {
		t.Errorf("Expected output length %d, got %d", outSize*2, len(output))
	}
}

func TestBidirectionalProcessSequence(t *testing.T) {
	inSize := 3
	outSize := 5
	f := NewLSTM(inSize, outSize)
	bi := NewBidirectional(f)

	seq := [][]float64{
		{1.0, 0.5, -0.5},
		{0.8, 0.3, -0.2},
		{0.6, 0.1, -0.4},
	}

	results := bi.ProcessSequence(seq)

	if len(results) != len(seq) {
		t.Errorf("Expected %d results, got %d", len(seq), len(results))
	}

	for i, res := range results {
		if len(res) != outSize*2 {
			t.Errorf("Result[%d] length = %d, expected %d", i, len(res), outSize*2)
		}
	}
}

func TestAttentionForward(t *testing.T) {
	inSize := 8
	att := NewGlobalAttention(inSize)

	input := make([]float64, inSize)
	for i := range input {
		input[i] = float64(i) * 0.1
	}

	output := att.Forward(input)
	if len(output) != inSize {
		t.Errorf("Expected output length %d, got %d", inSize, len(output))
	}
}
