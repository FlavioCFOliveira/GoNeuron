package layer

import (
	"math"
	"testing"
)

func TestEmbeddingForward(t *testing.T) {
	// Test basic embedding lookup
	emb := NewEmbedding(5, 3)

	// Input indices
	input := []float32{0, 2, 4}

	output := emb.Forward(input)

	// Check output shape
	if len(output) != 3*3 {
		t.Errorf("Output length = %d, expected 9", len(output))
	}

	// Check that each embedding vector is correct
	weights := emb.GetWeights()
	for i := 0; i < 3; i++ {
		idx := int(input[i])
		start := i * 3
		weightStart := idx * 3
		for d := 0; d < 3; d++ {
			if float32(math.Abs(float64(output[start+d]-weights[weightStart+d]))) > 1e-6 {
				t.Errorf("Output[%d] = %f, expected %f", start+d, output[start+d], weights[weightStart+d])
			}
		}
	}
}

func TestEmbeddingBackward(t *testing.T) {
	emb := NewEmbedding(5, 3)

	input := []float32{0, 2, 4}
	emb.Forward(input)

	// Pass gradient of all ones
	grad := make([]float32, 9)
	for i := range grad {
		grad[i] = 1.0
	}

	outputGrad := emb.Backward(grad)

	// Check that gradient w.r.t. input is zero
	for i := 0; i < 3; i++ {
		if outputGrad[i] != 0 {
			t.Errorf("GradInput[%d] = %f, expected 0", i, outputGrad[i])
		}
	}

	// Check that gradients are accumulated for the used embeddings
	gradWeights := emb.Gradients()

	// Embedding 0 (indices[0]=0) should have gradient 1 for each dimension
	for d := 0; d < 3; d++ {
		if float32(math.Abs(float64(gradWeights[d]-1.0))) > 1e-6 {
			t.Errorf("GradWeights[%d] = %f, expected 1.0", d, gradWeights[d])
		}
	}

	// Embedding 2 (indices[1]=2) should have gradient 1 for each dimension
	for d := 0; d < 3; d++ {
		if float32(math.Abs(float64(gradWeights[6+d]-1.0))) > 1e-6 {
			t.Errorf("GradWeights[%d] = %f, expected 1.0", 6+d, gradWeights[6+d])
		}
	}

	// Embedding 4 (indices[2]=4) should have gradient 1 for each dimension
	for d := 0; d < 3; d++ {
		if float32(math.Abs(float64(gradWeights[12+d]-1.0))) > 1e-6 {
			t.Errorf("GradWeights[%d] = %f, expected 1.0", 12+d, gradWeights[12+d])
		}
	}

	// Embedding 1 and 3 should have zero gradient
	for d := 0; d < 3; d++ {
		if gradWeights[3+d] != 0 {
			t.Errorf("GradWeights[%d] = %f, expected 0", 3+d, gradWeights[3+d])
		}
		if gradWeights[9+d] != 0 {
			t.Errorf("GradWeights[%d] = %f, expected 0", 9+d, gradWeights[9+d])
		}
	}
}

func TestEmbeddingParams(t *testing.T) {
	emb := NewEmbedding(4, 2)

	// Test Params and SetParams
	params := emb.Params()
	if len(params) != 8 { // 4 gamma + 4 beta
		t.Errorf("Expected 8 params, got %d", len(params))
	}

	// Modify params
	newParams := make([]float32, 8)
	for i := 0; i < 8; i++ {
		newParams[i] = float32(i + 10)
	}
	emb.SetParams(newParams)

	// Verify
	params2 := emb.Params()
	for i := 0; i < 8; i++ {
		if float32(math.Abs(float64(params2[i]-float32(i+10)))) > 1e-6 {
			t.Errorf("Param[%d] = %f, expected %f", i, params2[i], float32(i+10))
		}
	}
}

func TestEmbeddingInOutSize(t *testing.T) {
	emb := NewEmbedding(10, 5)

	if emb.InSize() != 10 {
		t.Errorf("InSize = %d, expected 10", emb.InSize())
	}
	if emb.OutSize() != 5 {
		t.Errorf("OutSize = %d, expected 5", emb.OutSize())
	}
}

func TestEmbeddingGetWeight(t *testing.T) {
	emb := NewEmbedding(3, 4)

	// Get specific weight
	weight := emb.GetWeight(1)

	if len(weight) != 4 {
		t.Errorf("Weight length = %d, expected 4", len(weight))
	}

	// Verify it matches the internal weights
	weights := emb.GetWeights()
	for i := 0; i < 4; i++ {
		if float32(math.Abs(float64(weight[i]-weights[4+i]))) > 1e-6 {
			t.Errorf("Weight[%d] = %f, expected %f", i, weight[i], weights[4+i])
		}
	}
}

func TestEmbeddingSetWeight(t *testing.T) {
	emb := NewEmbedding(3, 4)

	// Set specific weight
	newWeight := []float32{1, 2, 3, 4}
	emb.SetWeight(1, newWeight)

	// Verify
	weight := emb.GetWeight(1)
	for i := 0; i < 4; i++ {
		if float32(math.Abs(float64(weight[i]-newWeight[i]))) > 1e-6 {
			t.Errorf("Weight[%d] = %f, expected %f", i, weight[i], newWeight[i])
		}
	}
}

func TestEmbeddingOutOfBounds(t *testing.T) {
	emb := NewEmbedding(3, 2)

	// Use out-of-bounds index
	input := []float32{100}
	output := emb.Forward(input)

	// Should default to first embedding (index 0)
	weights := emb.GetWeights()
	for i := 0; i < 2; i++ {
		if float32(math.Abs(float64(output[i]-weights[i]))) > 1e-6 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], weights[i])
		}
	}
}

func TestEmbeddingBatch(t *testing.T) {
	emb := NewEmbedding(5, 3)

	// Batch of 4 indices
	input := []float32{0, 1, 2, 3}
	output := emb.Forward(input)

	// Check output shape
	if len(output) != 4*3 {
		t.Errorf("Output length = %d, expected 12", len(output))
	}

	// Verify each embedding is correct
	weights := emb.GetWeights()
	for b := 0; b < 4; b++ {
		idx := int(input[b])
		start := b * 3
		weightStart := idx * 3
		for d := 0; d < 3; d++ {
			if float32(math.Abs(float64(output[start+d]-weights[weightStart+d]))) > 1e-6 {
				t.Errorf("Output[%d] = %f, expected %f", start+d, output[start+d], weights[weightStart+d])
			}
		}
	}
}

func TestEmbeddingReset(t *testing.T) {
	emb := NewEmbedding(3, 2)

	input := []float32{0, 1}
	emb.Forward(input)

	// Reset
	emb.Reset()

	// Saved indices should be cleared
	if len(emb.savedIndices) != 0 {
		t.Errorf("Saved indices not cleared after reset: %d", len(emb.savedIndices))
	}
}

func BenchmarkEmbeddingForward(b *testing.B) {
	emb := NewEmbedding(10000, 128)

	// Batch of 32 indices
	input := make([]float32, 32)
	for i := range input {
		input[i] = float32(i % 1000)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		emb.Forward(input)
	}
}

func BenchmarkEmbeddingBackward(b *testing.B) {
	emb := NewEmbedding(10000, 128)

	input := make([]float32, 32)
	for i := range input {
		input[i] = float32(i % 1000)
	}
	emb.Forward(input)

	grad := make([]float32, 32*128)
	for i := range grad {
		grad[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		emb.Backward(grad)
	}
}
