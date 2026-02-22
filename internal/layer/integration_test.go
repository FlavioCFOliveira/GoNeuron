package layer

import (
	"fmt"
	"math"
	"testing"
)

func TestBatchNormIntegration(t *testing.T) {
	// Test BatchNorm2D
	bn := NewBatchNorm2D(4, 1e-5, 0.1, true)

	input := make([]float64, 8) // 2 batches x 4 features
	for i := range input {
		input[i] = float64(i)
	}

	output := bn.Forward(input)
	params := bn.Params()

	fmt.Printf("BatchNorm2D test: OutputLen=%d, Params=%d\n", len(output), len(params))
}

func TestLayerNormIntegration(t *testing.T) {
	// Test LayerNorm
	ln := NewLayerNorm(8, 1e-5, true)

	input := make([]float64, 8)
	for i := range input {
		input[i] = float64(i)
	}

	output := ln.Forward(input)

	fmt.Printf("LayerNorm test: OutputLen=%d\n", len(output))
}

func TestEmbeddingIntegration(t *testing.T) {
	// Test Embedding
	emb := NewEmbedding(100, 16)

	input := []float64{0, 5, 10, 15}
	output := emb.Forward(input)

	fmt.Printf("Embedding test: OutputLen=%d\n", len(output))
}

func TestGRUIntegration(t *testing.T) {
	// Test GRU
	gru := NewGRU(5, 10)

	input := make([]float64, 5)
	for i := range input {
		input[i] = float64(i) * 0.1
	}

	output := gru.Forward(input)

	fmt.Printf("GRU test: OutputLen=%d\n", len(output))
}

func TestIntegrationValidateOutputs(t *testing.T) {
	// Test Dropout
	dropout := NewDropout(0.5, 10)
	dropout.SetTraining(false)
	input1 := make([]float64, 10)
	for i := range input1 {
		input1[i] = float64(i)
	}
	out1 := dropout.Forward(input1)
	for _, v := range out1 {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Dropout output contains invalid value: %f", v)
		}
	}

	// Test MaxPool2D (single channel input)
	maxPool := NewMaxPool2D(1, 2, 2, 0)
	input2 := make([]float64, 64)
	for i := range input2 {
		input2[i] = float64(i % 16)
	}
	out2 := maxPool.Forward(input2)
	for _, v := range out2 {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("MaxPool2D output contains invalid value: %f", v)
		}
	}

	// Test BatchNorm2D
	bn := NewBatchNorm2D(4, 1e-5, 0.1, true)
	input3 := make([]float64, 8)
	for i := range input3 {
		input3[i] = float64(i)
	}
	out3 := bn.Forward(input3)
	for _, v := range out3 {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("BatchNorm2D output contains invalid value: %f", v)
		}
	}

	// Test LayerNorm
	ln := NewLayerNorm(8, 1e-5, true)
	input4 := make([]float64, 8)
	for i := range input4 {
		input4[i] = float64(i)
	}
	out4 := ln.Forward(input4)
	for _, v := range out4 {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("LayerNorm output contains invalid value: %f", v)
		}
	}

	// Test Embedding
	emb := NewEmbedding(100, 16)
	input5 := []float64{0, 5, 10, 15}
	out5 := emb.Forward(input5)
	for _, v := range out5 {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Embedding output contains invalid value: %f", v)
		}
	}

	// Test GRU
	gru := NewGRU(5, 10)
	input6 := make([]float64, 5)
	for i := range input6 {
		input6[i] = float64(i) * 0.1
	}
	out6 := gru.Forward(input6)
	for _, v := range out6 {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("GRU output contains invalid value: %f", v)
		}
	}
}
