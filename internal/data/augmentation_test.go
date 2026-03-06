// Package data provides tests for data augmentation
package data

import (
	"math"
	"testing"
)

// TestFlipAugmenter tests flip augmentation
func TestFlipAugmenter(t *testing.T) {
	// Create a simple 2x2 image with 1 channel
	// [1, 2]
	// [3, 4]
	input := []float32{1, 2, 3, 4}
	flip := NewFlipAugmenter(1.0, true, false, 2, 2, 1) // Always apply, horizontal only

	output := flip.Apply(input)

	// After horizontal flip:
	// [2, 1]
	// [4, 3]
	expected := []float32{2, 1, 4, 3}

	if len(output) != len(expected) {
		t.Errorf("Output length = %d, want %d", len(output), len(expected))
	}

	for i, exp := range expected {
		if output[i] != exp {
			t.Errorf("output[%d] = %v, want %v", i, output[i], exp)
		}
	}
}

// TestFlipAugmenterVertical tests vertical flip
func TestFlipAugmenterVertical(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	flip := NewFlipAugmenter(1.0, false, true, 2, 2, 1) // Always apply, vertical only

	output := flip.Apply(input)

	// After vertical flip:
	// [3, 4]
	// [1, 2]
	expected := []float32{3, 4, 1, 2}

	for i, exp := range expected {
		if output[i] != exp {
			t.Errorf("output[%d] = %v, want %v", i, output[i], exp)
		}
	}
}

// TestFlipAugmenterProbability tests probability
func TestFlipAugmenterProbability(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	flip := NewFlipAugmenter(0.0, true, false, 2, 2, 1) // Never apply

	// Apply multiple times
	for i := 0; i < 10; i++ {
		output := flip.Apply(input)
		// Should always return unchanged input
		for j, v := range input {
			if output[j] != v {
				t.Errorf("With prob=0, output should be unchanged")
				return
			}
		}
	}
}

// TestScaleAugmenter tests scale augmentation
func TestScaleAugmenter(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0}
	scale := NewScaleAugmenter(1.0, 2.0, 2.0) // Always scale by 2.0

	output := scale.Apply(input)

	expected := []float32{2.0, 4.0, 6.0}
	for i, exp := range expected {
		if math.Abs(float64(output[i]-exp)) > 1e-6 {
			t.Errorf("output[%d] = %v, want %v", i, output[i], exp)
		}
	}
}

// TestJitterAugmenter tests jitter augmentation
func TestJitterAugmenter(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0}
	jitter := NewJitterAugmenter(1.0, 0.1) // Always apply with sigma 0.1

	output := jitter.Apply(input)

	// Output should be different from input
	same := true
	for i := range input {
		if output[i] != input[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("Jitter should modify input")
	}

	// But values should be close (within reasonable range)
	for i := range input {
		diff := math.Abs(float64(output[i] - input[i]))
		if diff > 0.5 { // Should be within ~5 sigma
			t.Errorf("Jitter produced too large change: %v", diff)
		}
	}
}

// TestWindowSliceAugmenter tests window slicing
func TestWindowSliceAugmenter(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	window := NewWindowSliceAugmenter(1.0, 5, 1) // Always apply, window size 5

	output := window.Apply(input)

	if len(output) != 5 {
		t.Errorf("Output length = %d, want 5", len(output))
	}

	// Output should be a contiguous slice of input
	found := false
	for start := 0; start <= len(input)-len(output); start++ {
		match := true
		for i := 0; i < len(output); i++ {
			if output[i] != input[start+i] {
				match = false
				break
			}
		}
		if match {
			found = true
			break
		}
	}
	if !found {
		t.Error("Output should be a contiguous slice of input")
	}
}

// TestAugmentationPipeline tests the pipeline
func TestAugmentationPipeline(t *testing.T) {
	p := NewAugmentationPipeline(false)
	p.Add(NewScaleAugmenter(1.0, 2.0, 2.0))
	p.Add(NewJitterAugmenter(1.0, 0.01))

	input := []float32{1.0, 2.0, 3.0}
	output := p.Apply(input)

	if len(output) != len(input) {
		t.Errorf("Output length = %d, want %d", len(output), len(input))
	}

	// Output should be scaled (approximately 2x)
	for i := range output {
		// Allow some tolerance for jitter
		if output[i] < 1.8*float32(i+1) || output[i] > 2.2*float32(i+1) {
			t.Errorf("output[%d] = %v, expected approximately %v", i, output[i], 2.0*float32(i+1))
		}
	}
}

// TestAugmentationPipelineBatch tests batch processing
func TestAugmentationPipelineBatch(t *testing.T) {
	p := NewAugmentationPipeline(false)
	p.Add(NewScaleAugmenter(1.0, 2.0, 2.0))

	batch := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}

	result := p.ApplyBatch(batch)

	if len(result) != len(batch) {
		t.Errorf("Result batch size = %d, want %d", len(result), len(batch))
	}

	// Each sample should be scaled by 2
	for i, sample := range result {
		for j, v := range sample {
			expected := batch[i][j] * 2.0
			if math.Abs(float64(v-expected)) > 1e-6 {
				t.Errorf("result[%d][%d] = %v, want %v", i, j, v, expected)
			}
		}
	}
}

// TestImageAugmentation tests predefined image augmentation
func TestImageAugmentation(t *testing.T) {
	aug := ImageAugmentation(28, 28, 1)

	if aug.Length() != 2 {
		t.Errorf("Pipeline length = %d, want 2", aug.Length())
	}

	// Create a sample image
	input := make([]float32, 28*28*1)
	for i := range input {
		input[i] = float32(i) / float32(len(input))
	}

	output := aug.Apply(input)

	if len(output) != len(input) {
		t.Errorf("Output length = %d, want %d", len(output), len(input))
	}
}

// TestTimeSeriesAugmentation tests predefined time series augmentation
func TestTimeSeriesAugmentation(t *testing.T) {
	aug := TimeSeriesAugmentation()

	if aug.Length() != 2 {
		t.Errorf("Pipeline length = %d, want 2", aug.Length())
	}

	input := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	output := aug.Apply(input)

	if len(output) != len(input) {
		t.Errorf("Output length = %d, want %d", len(output), len(input))
	}
}

// TestSetSeed tests reproducibility
func TestSetSeed(t *testing.T) {
	jitter := NewJitterAugmenter(1.0, 0.1)
	jitter.SetSeed(42)

	input := []float32{1.0, 2.0, 3.0}

	// First run
	output1 := jitter.Apply(input)

	// Reset seed and run again
	jitter.SetSeed(42)
	output2 := jitter.Apply(input)

	// Should be identical
	for i := range output1 {
		if output1[i] != output2[i] {
			t.Error("Same seed should produce same results")
			return
		}
	}
}

// TestProbabilityGetterSetter tests probability methods
func TestProbabilityGetterSetter(t *testing.T) {
	flip := NewFlipAugmenter(0.5, true, false, 2, 2, 1)

	if flip.Probability() != 0.5 {
		t.Errorf("Initial probability = %v, want 0.5", flip.Probability())
	}

	flip.SetProbability(0.8)
	if flip.Probability() != 0.8 {
		t.Errorf("Updated probability = %v, want 0.8", flip.Probability())
	}
}

// TestRotateAugmenter tests rotation augmentation
func TestRotateAugmenter(t *testing.T) {
	// Create a simple 3x3 image with 1 channel
	// [1, 2, 3]
	// [4, 5, 6]
	// [7, 8, 9]
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	rotate := NewRotateAugmenter(1.0, 0.0, 3, 3, 1) // Always apply, 0 rotation (should return same)

	output := rotate.Apply(input)

	if len(output) != len(input) {
		t.Errorf("Output length = %d, want %d", len(output), len(input))
	}
}

// BenchmarkAugmentation benchmarks augmentation pipeline
func BenchmarkAugmentationPipeline(b *testing.B) {
	p := NewAugmentationPipeline(false)
	p.Add(NewScaleAugmenter(1.0, 0.9, 1.1))
	p.Add(NewJitterAugmenter(1.0, 0.01))

	input := make([]float32, 1000)
	for i := range input {
		input[i] = float32(i) * 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = p.Apply(input)
	}
}

// BenchmarkFlipAugmenter benchmarks flip augmentation
func BenchmarkFlipAugmenter(b *testing.B) {
	flip := NewFlipAugmenter(1.0, true, false, 28, 28, 1)
	input := make([]float32, 28*28)
	for i := range input {
		input[i] = float32(i) / float32(len(input))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = flip.Apply(input)
	}
}
