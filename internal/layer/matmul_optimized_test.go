// Package layer provides tests for optimized CPU operations.
package layer

import (
	"math"
	"testing"
)

// TestMatMulOptimizedCorrectness verifies optimized MatMul produces correct results
func TestMatMulOptimizedCorrectness(t *testing.T) {
	tests := []struct {
		name      string
		batchSize int
		inSize    int
		outSize   int
	}{
		{"Small_1x4x3", 1, 4, 3},
		{"Small_2x8x5", 2, 8, 5},
		{"Medium_16x64x32", 16, 64, 32},
		{"Medium_32x128x64", 32, 128, 64},
		{"Large_64x256x128", 64, 256, 128},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Allocate matrices
			input := make([]float32, tt.batchSize*tt.inSize)
			weights := make([]float32, tt.outSize*tt.inSize)
			bias := make([]float32, tt.outSize)
			output := make([]float32, tt.batchSize*tt.outSize)

			// Initialize with deterministic values
			for i := range input {
				input[i] = float32(i) / 100.0
			}
			for i := range weights {
				weights[i] = float32(i) / 1000.0
			}
			for i := range bias {
				bias[i] = float32(i) / 10.0
			}

			// Compute with optimized version
			matMulOptimized(input, weights, bias, output, tt.batchSize, tt.inSize, tt.outSize)

			// Compute with simple version for comparison
			expected := make([]float32, tt.batchSize*tt.outSize)
			matMulSimple(input, weights, bias, expected, tt.batchSize, tt.inSize, tt.outSize)

			// Compare results
			maxDiff := float64(0)
			for i := range output {
				diff := math.Abs(float64(output[i] - expected[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > 1e-5 {
					t.Errorf("Output mismatch at index %d: got %v, want %v, diff=%v",
						i, output[i], expected[i], diff)
					break
				}
			}

			t.Logf("Max difference: %v", maxDiff)
		})
	}
}

// TestMatMulUnrolled verifies unrolled implementation
func TestMatMulUnrolled(t *testing.T) {
	batchSize, inSize, outSize := 8, 128, 64

	input := make([]float32, batchSize*inSize)
	weights := make([]float32, outSize*inSize)
	bias := make([]float32, outSize)
	outputSimple := make([]float32, batchSize*outSize)
	outputUnrolled := make([]float32, batchSize*outSize)

	// Initialize
	for i := range input {
		input[i] = float32(i%100) / 100.0
	}
	for i := range weights {
		weights[i] = float32(i%100) / 1000.0
	}
	for i := range bias {
		bias[i] = float32(i) / 10.0
	}

	// Compute with both methods
	matMulSimple(input, weights, bias, outputSimple, batchSize, inSize, outSize)
	matMulUnrolled(input, weights, bias, outputUnrolled, batchSize, inSize, outSize)

	// Compare
	for i := range outputSimple {
		diff := math.Abs(float64(outputSimple[i] - outputUnrolled[i]))
		if diff > 1e-5 {
			t.Errorf("Unrolled mismatch at %d: simple=%v, unrolled=%v, diff=%v",
				i, outputSimple[i], outputUnrolled[i], diff)
		}
	}
}

// TestMatMulParallelBlocked verifies parallel blocked implementation
func TestMatMulParallelBlocked(t *testing.T) {
	// Use large enough size to trigger parallel path
	batchSize, inSize, outSize := 64, 256, 128

	input := make([]float32, batchSize*inSize)
	weights := make([]float32, outSize*inSize)
	bias := make([]float32, outSize)
	outputSimple := make([]float32, batchSize*outSize)
	outputBlocked := make([]float32, batchSize*outSize)

	// Initialize
	for i := range input {
		input[i] = float32(i%100) / 100.0
	}
	for i := range weights {
		weights[i] = float32(i%100) / 1000.0
	}
	for i := range bias {
		bias[i] = float32(i) / 10.0
	}

	// Compute with both methods
	matMulSimple(input, weights, bias, outputSimple, batchSize, inSize, outSize)
	matMulParallelBlocked(input, weights, bias, outputBlocked, batchSize, inSize, outSize)

	// Compare
	maxDiff := float64(0)
	for i := range outputSimple {
		diff := math.Abs(float64(outputSimple[i] - outputBlocked[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-4 { // Parallel may have slightly higher numerical error
			t.Errorf("Parallel blocked mismatch at %d: simple=%v, blocked=%v, diff=%v",
				i, outputSimple[i], outputBlocked[i], diff)
			if i > 10 {
				t.Logf("(showing first 10 errors only)")
				break
			}
		}
	}

	t.Logf("Max difference in parallel blocked: %v", maxDiff)
}

// TestMatMulBackwardOptimized verifies backward pass gradients
func TestMatMulBackwardOptimized(t *testing.T) {
	batchSize, inSize, outSize := 8, 32, 16

	// Allocate
	input := make([]float32, batchSize*inSize)
	weights := make([]float32, outSize*inSize)
	gradOutput := make([]float32, batchSize*outSize)
	gradW := make([]float32, outSize*inSize)
	gradIn := make([]float32, batchSize*inSize)

	// Initialize
	for i := range input {
		input[i] = float32(i) / 100.0
	}
	for i := range weights {
		weights[i] = float32(i) / 1000.0
	}
	for i := range gradOutput {
		gradOutput[i] = float32(i) / 10.0
	}

	// Compute gradients
	matMulBackwardOptimized(input, weights, gradOutput, gradW, gradIn, batchSize, inSize, outSize)

	// Verify gradW shape and non-zero
	hasNonZero := false
	for _, v := range gradW {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("gradW contains all zeros")
	}

	// Verify gradIn shape and non-zero
	hasNonZero = false
	for _, v := range gradIn {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("gradIn contains all zeros")
	}
}

// TestSelectMatMulImplementation verifies implementation selection
func TestSelectMatMulImplementation(t *testing.T) {
	tests := []struct {
		batchSize int
		inSize    int
		outSize   int
		expected  string
	}{
		{1, 8, 8, "simple"},         // Small: 512 ops
		{16, 64, 32, "unrolled"},    // Medium: 65536 ops
		{64, 128, 64, "parallel_blocked"},    // Large: 524288 ops
		{128, 256, 256, "parallel_blocked"}, // Very large
	}

	for _, tt := range tests {
		impl := SelectMatMulImplementation(tt.batchSize, tt.inSize, tt.outSize)
		if impl != tt.expected {
			t.Errorf("SelectMatMulImplementation(%d, %d, %d) = %s, want %s",
				tt.batchSize, tt.inSize, tt.outSize, impl, tt.expected)
		}
	}
}

// TestGetPerformanceMetrics verifies metrics calculation
func TestGetPerformanceMetrics(t *testing.T) {
	metrics := GetPerformanceMetrics(32, 128, 64)

	requiredKeys := []string{
		"total_ops",
		"memory_bytes",
		"arithmetic_intensity",
		"recommended_implementation",
		"optimal_block_size",
		"estimated_peak_gflops",
	}

	for _, key := range requiredKeys {
		if _, ok := metrics[key]; !ok {
			t.Errorf("Missing metric key: %s", key)
		}
	}

	// Verify arithmetic intensity is reasonable
	ai, ok := metrics["arithmetic_intensity"].(float64)
	if !ok || ai <= 0 {
		t.Error("Invalid arithmetic_intensity")
	}

	// Verify block size
	blockSize, ok := metrics["optimal_block_size"].(int)
	if !ok || blockSize <= 0 {
		t.Error("Invalid optimal_block_size")
	}
}

// TestEstimateOptimalBlockSize returns positive value
func TestEstimateOptimalBlockSize(t *testing.T) {
	blockSize := EstimateOptimalBlockSize()
	if blockSize <= 0 {
		t.Errorf("EstimateOptimalBlockSize() = %d, want positive value", blockSize)
	}
	if blockSize > 1024 {
		t.Logf("Warning: block size %d seems large", blockSize)
	}
}

// TestMatMulBackwardCorrectness verifies backward pass computes correct gradients
func TestMatMulBackwardCorrectness(t *testing.T) {
	// Simple case: 1 sample, 2 inputs, 3 outputs
	batchSize, inSize, outSize := 1, 2, 3

	input := []float32{1.0, 2.0}                    // shape (1, 2)
	weights := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0} // shape (3, 2)
	gradOutput := []float32{0.1, 0.2, 0.3}          // shape (1, 3)

	gradW := make([]float32, outSize*inSize)
	gradIn := make([]float32, batchSize*inSize)

	matMulBackwardOptimized(input, weights, gradOutput, gradW, gradIn, batchSize, inSize, outSize)

	// Expected gradW:
	// gradW[0,0] = gradOutput[0,0] * input[0,0] = 0.1 * 1.0 = 0.1
	// gradW[0,1] = gradOutput[0,0] * input[0,1] = 0.1 * 2.0 = 0.2
	// gradW[1,0] = gradOutput[0,1] * input[0,0] = 0.2 * 1.0 = 0.2
	// etc.

	expectedGradW := []float32{0.1, 0.2, 0.2, 0.4, 0.3, 0.6}
	for i := range gradW {
		diff := math.Abs(float64(gradW[i] - expectedGradW[i]))
		if diff > 1e-6 {
			t.Errorf("gradW[%d] = %v, want %v", i, gradW[i], expectedGradW[i])
		}
	}

	// Expected gradIn:
	// gradIn[0,0] = sum_o gradOutput[0,o] * weights[o,0]
	//             = 0.1*1.0 + 0.2*3.0 + 0.3*5.0 = 0.1 + 0.6 + 1.5 = 2.2
	// gradIn[0,1] = 0.1*2.0 + 0.2*4.0 + 0.3*6.0 = 0.2 + 0.8 + 1.8 = 2.8
	expectedGradIn := []float32{2.2, 2.8}
	for i := range gradIn {
		diff := math.Abs(float64(gradIn[i] - expectedGradIn[i]))
		if diff > 1e-6 {
			t.Errorf("gradIn[%d] = %v, want %v", i, gradIn[i], expectedGradIn[i])
		}
	}
}

// BenchmarkMatMul implementations
func BenchmarkMatMulSimple(b *testing.B) {
	batchSize, inSize, outSize := 32, 128, 64

	input := make([]float32, batchSize*inSize)
	weights := make([]float32, outSize*inSize)
	bias := make([]float32, outSize)
	output := make([]float32, batchSize*outSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matMulSimple(input, weights, bias, output, batchSize, inSize, outSize)
	}
}

func BenchmarkMatMulUnrolled(b *testing.B) {
	batchSize, inSize, outSize := 32, 128, 64

	input := make([]float32, batchSize*inSize)
	weights := make([]float32, outSize*inSize)
	bias := make([]float32, outSize)
	output := make([]float32, batchSize*outSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matMulUnrolled(input, weights, bias, output, batchSize, inSize, outSize)
	}
}

func BenchmarkMatMulOptimized(b *testing.B) {
	batchSize, inSize, outSize := 32, 128, 64

	input := make([]float32, batchSize*inSize)
	weights := make([]float32, outSize*inSize)
	bias := make([]float32, outSize)
	output := make([]float32, batchSize*outSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matMulOptimized(input, weights, bias, output, batchSize, inSize, outSize)
	}
}
