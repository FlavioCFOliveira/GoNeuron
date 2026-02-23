package layer

import (
	"math"
	"testing"
)

func TestBatchNorm2DForward(t *testing.T) {
	// Test batch normalization with 2 features
	bn := NewBatchNorm2D(2, 1e-5, 0.1, false)

	// Input: 2 spatial positions x 2 features = 4 values
	// Feature 0: [1, 5], Feature 1: [2, 6]
	// Layout: [1, 5, 2, 6] (channel-major NCHW)
	input := []float32{1, 5, 2, 6}

	output := bn.Forward(input)

	// Feature 0: mean=3, std=sqrt(4)=2
	// Normalized: [(1-3)/2, (5-3)/2] = [-1, 1]
	// Feature 1: mean=4, std=sqrt(4)=2
	// Normalized: [(2-4)/2, (6-4)/2] = [-1, 1]
	expected := []float32{-1, 1, -1, 1}

	for i := 0; i < 4; i++ {
		if float32(math.Abs(float64(output[i]-expected[i]))) > 1e-5 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], expected[i])
		}
	}
}

func TestBatchNorm2DWithAffine(t *testing.T) {
	// Test with affine transformation
	bn := NewBatchNorm2D(2, 1e-5, 0.1, true)

	// Set gamma = [2, 2], beta = [1, 1]
	// After normalization, output = 2 * normalized + 1
	input := []float32{1, 5, 2, 6}
	bn.Forward(input)

	// Get gamma and verify
	gamma := bn.GetGamma()
	if len(gamma) != 2 {
		t.Errorf("Gamma length = %d, expected 2", len(gamma))
	}

	// Default gamma should be all 1s
	for i := 0; i < 2; i++ {
		if float32(math.Abs(float64(gamma[i]-1.0))) > 1e-6 {
			t.Errorf("Gamma[%d] = %f, expected 1.0", i, gamma[i])
		}
	}

	// Default beta should be all 0s
	beta := bn.GetBeta()
	for i := 0; i < 2; i++ {
		if float32(math.Abs(float64(beta[i]))) > 1e-6 {
			t.Errorf("Beta[%d] = %f, expected 0.0", i, beta[i])
		}
	}
}

func TestBatchNorm2DWithoutAffine(t *testing.T) {
	// Test without affine transformation
	bn := NewBatchNorm2D(2, 1e-5, 0.1, false)

	input := []float32{1, 5, 2, 6}
	output := bn.Forward(input)

	// Should be same as normalized
	expected := []float32{-1, 1, -1, 1}

	for i := 0; i < 4; i++ {
		if float32(math.Abs(float64(output[i]-expected[i]))) > 1e-5 {
			t.Errorf("Output[%d] = %f, expected %f", i, output[i], expected[i])
		}
	}

	// Should have no parameters
	params := bn.Params()
	if len(params) != 0 {
		t.Errorf("Expected 0 params, got %d", len(params))
	}
}

func TestBatchNorm2DBackward(t *testing.T) {
	bn := NewBatchNorm2D(2, 1e-5, 0.1, false)

	input := []float32{1, 5, 2, 6}
	bn.Forward(input)

	// Pass gradient of all ones
	grad := []float32{1, 1, 1, 1}
	outputGrad := bn.Backward(grad)

	// For batch norm with all ones gradient and no affine,
	// the gradient should sum to zero per channel
	// Channel 0: grad[0] + grad[1] = 1 + 1 = 2
	// Channel 1: grad[2] + grad[3] = 1 + 1 = 2
	// Sum of all gradients should be ~0 (due to normalization)
	sumGrad := float32(0.0)
	for i := 0; i < 4; i++ {
		sumGrad += outputGrad[i]
	}

	if float32(math.Abs(float64(sumGrad))) > 1e-6 {
		t.Errorf("Sum of gradients = %f, expected ~0", sumGrad)
	}
}

func TestBatchNorm2DParams(t *testing.T) {
	bn := NewBatchNorm2D(2, 1e-5, 0.1, true)

	// Test Params and SetParams
	params := bn.Params()
	if len(params) != 4 { // 2 gamma + 2 beta
		t.Errorf("Expected 4 params, got %d", len(params))
	}

	// Modify params
	newParams := make([]float32, 4)
	for i := 0; i < 4; i++ {
		newParams[i] = float32(i + 10)
	}
	bn.SetParams(newParams)

	// Verify
	params2 := bn.Params()
	for i := 0; i < 4; i++ {
		if float32(math.Abs(float64(params2[i]-float32(i+10)))) > 1e-6 {
			t.Errorf("Param[%d] = %f, expected %f", i, params2[i], float32(i+10))
		}
	}
}

func TestBatchNorm2DInOutSize(t *testing.T) {
	bn := NewBatchNorm2D(10, 1e-5, 0.1, false)

	if bn.InSize() != 10 {
		t.Errorf("InSize = %d, expected 10", bn.InSize())
	}
	if bn.OutSize() != 10 {
		t.Errorf("OutSize = %d, expected 10", bn.OutSize())
	}
}

func TestBatchNorm2DEps(t *testing.T) {
	bn := NewBatchNorm2D(2, float32(1e-6), 0.1, false)

	if bn.GetEps() != 1e-6 {
		t.Errorf("Eps = %f, expected 1e-6", bn.GetEps())
	}
}

func TestBatchNorm2DMomentum(t *testing.T) {
	bn := NewBatchNorm2D(2, 1e-5, 0.5, false)

	if bn.GetMomentum() != 0.5 {
		t.Errorf("Momentum = %f, expected 0.5", bn.GetMomentum())
	}
}

func TestBatchNorm2DRunningStats(t *testing.T) {
	// Test that running statistics are updated during training
	bn := NewBatchNorm2D(2, 1e-5, 0.5, false) // Use higher momentum for more noticeable changes

	// First forward pass with input centered around 3
	input1 := []float32{1, 5, 2, 6} // mean = 3.5
	bn.Forward(input1)

	runningMean1 := make([]float32, 2)
	copy(runningMean1, bn.GetRunningMean())
	runningVar1 := make([]float32, 2)
	copy(runningVar1, bn.GetRunningVar())

	// Second forward pass with input centered around 6
	input2 := []float32{4, 8, 5, 9} // mean = 6.5
	bn.Forward(input2)

	runningMean2 := bn.GetRunningMean()
	runningVar2 := bn.GetRunningVar()

	// Running stats should be updated (with momentum=0.5, should be roughly halfway)
	for i := 0; i < 2; i++ {
		if runningMean1[i] == runningMean2[i] {
			t.Errorf("Running mean[%d] did not change: %f", i, runningMean1[i])
		}
		if runningVar1[i] == runningVar2[i] {
			t.Errorf("Running var[%d] did not change: %f", i, runningVar1[i])
		}
	}

	// Verify running mean moved towards 6.5
	for i := 0; i < 2; i++ {
		if runningMean2[i] <= runningMean1[i] {
			t.Errorf("Running mean[%d] should have increased, was %f now %f", i, runningMean1[i], runningMean2[i])
		}
	}
}

func TestBatchNorm2DNumFeatures(t *testing.T) {
	bn := NewBatchNorm2D(3, 1e-5, 0.1, false)

	// Input: 2 batches x 3 features = 6 values
	input := []float32{1, 2, 3, 4, 5, 6}

	output := bn.Forward(input)

	if len(output) != 6 {
		t.Errorf("Output length = %d, expected 6", len(output))
	}
}

func BenchmarkBatchNorm2DForward(b *testing.B) {
	bn := NewBatchNorm2D(64, 1e-5, 0.1, true)

	// 32x32 feature maps, 10 batches
	input := make([]float32, 32*32*64*10)
	for i := range input {
		input[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bn.Forward(input)
	}
}

func BenchmarkBatchNorm2DBackward(b *testing.B) {
	bn := NewBatchNorm2D(64, 1e-5, 0.1, true)

	input := make([]float32, 32*32*64*10)
	for i := range input {
		input[i] = float32(i)
	}
	bn.Forward(input)

	grad := make([]float32, len(input))
	for i := range grad {
		grad[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bn.Backward(grad)
	}
}
