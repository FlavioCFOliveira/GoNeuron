package layer

import (
	"fmt"
	"math"
	"testing"
)

func TestBatchNormIntegration(t *testing.T) {
	// Test BatchNorm2D
	bn , _ := NewBatchNorm2D(4, 1e-5, 0.1, true)

	input := make([]float32, 8) // 2 batches x 4 features
	for i := range input {
		input[i] = float32(i)
	}

	output, _ := bn.Forward(input)
	params := bn.Params()

	fmt.Printf("BatchNorm2D test: OutputLen=%d, Params=%d\n", len(output), len(params))
}

func TestLayerNormIntegration(t *testing.T) {
	// Test LayerNorm
	ln , _ := NewLayerNorm(8, 1e-5, true)

	input := make([]float32, 8)
	for i := range input {
		input[i] = float32(i)
	}

	output, _ := ln.Forward(input)

	fmt.Printf("LayerNorm test: OutputLen=%d\n", len(output))
}

func TestEmbeddingIntegration(t *testing.T) {
	// Test Embedding
	emb , _ := NewEmbedding(100, 16)

	input := []float32{0, 5, 10, 15}
	output, _ := emb.Forward(input)

	fmt.Printf("Embedding test: OutputLen=%d\n", len(output))
}

func TestGRUIntegration(t *testing.T) {
	// Test GRU
	gru , _ := NewGRU(5, 10)

	input := make([]float32, 5)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	output, _ := gru.Forward(input)

	fmt.Printf("GRU test: OutputLen=%d\n", len(output))
}

func TestIntegrationValidateOutputs(t *testing.T) {
	// Test Dropout
	dropout , _ := NewDropout(0.5, 10)
	dropout.SetTraining(false)
	input1 := make([]float32, 10)
	for i := range input1 {
		input1[i] = float32(i)
	}
	out1, err := dropout.Forward(input1)
	if err != nil {
		t.Fatalf("Dropout Forward failed: %v", err)
	}
	for _, v := range out1 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("Dropout output contains invalid value: %f", v)
		}
	}

	// Test MaxPool2D (single channel input)
	maxPool , _ := NewMaxPool2D(1, 2, 2, 0)
	input2 := make([]float32, 64)
	for i := range input2 {
		input2[i] = float32(i % 16)
	}
	out2, err := maxPool.Forward(input2)
	if err != nil {
		t.Fatalf("MaxPool2D Forward failed: %v", err)
	}
	for _, v := range out2 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("MaxPool2D output contains invalid value: %f", v)
		}
	}

	// Test BatchNorm2D
	bn , _ := NewBatchNorm2D(4, 1e-5, 0.1, true)
	input3 := make([]float32, 8)
	for i := range input3 {
		input3[i] = float32(i)
	}
	out3, err := bn.Forward(input3)
	if err != nil {
		t.Fatalf("BatchNorm2D Forward failed: %v", err)
	}
	for _, v := range out3 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("BatchNorm2D output contains invalid value: %f", v)
		}
	}

	// Test LayerNorm
	ln , _ := NewLayerNorm(8, 1e-5, true)
	input4 := make([]float32, 8)
	for i := range input4 {
		input4[i] = float32(i)
	}
	out4, err := ln.Forward(input4)
	if err != nil {
		t.Fatalf("LayerNorm Forward failed: %v", err)
	}
	for _, v := range out4 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("LayerNorm output contains invalid value: %f", v)
		}
	}

	// Test Embedding
	emb , _ := NewEmbedding(100, 16)
	input5 := []float32{0, 5, 10, 15}
	out5, _ := emb.Forward(input5)
	for _, v := range out5 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("Embedding output contains invalid value: %f", v)
		}
	}

	// Test GRU
	gru , _ := NewGRU(5, 10)
	input6 := make([]float32, 5)
	for i := range input6 {
		input6[i] = float32(i) * 0.1
	}
	out6, _ := gru.Forward(input6)
	for _, v := range out6 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("GRU output contains invalid value: %f", v)
		}
	}
}
