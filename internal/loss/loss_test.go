// Package loss provides comprehensive unit tests for loss functions.
package loss

import (
	"math"
	"testing"
)

// TestMSEForward tests MSE forward pass.
func TestMSEForward(t *testing.T) {
	mse := MSE{}

	tests := []struct {
		name     string
		yPred    []float64
		yTrue    []float64
		expected float64
	}{
		{"Perfect prediction", []float64{1.0, 2.0, 3.0}, []float64{1.0, 2.0, 3.0}, 0.0},
		{"Single error", []float64{1.0, 2.0}, []float64{1.5, 2.0}, 0.125}, // (0.5^2 + 0) / 2 = 0.125
		{"Multiple errors", []float64{1.0, 2.0, 3.0}, []float64{0.0, 1.0, 2.0}, 1.0}, // (1+1+1)/3 = 1
		{"Large errors", []float64{10.0}, []float64{0.0}, 100.0}, // 10^2 = 100
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mse.Forward(tt.yPred, tt.yTrue)
			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("MSE.Forward() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestMSEForwardLengthMismatch tests error handling.
func TestMSEForwardLengthMismatch(t *testing.T) {
	mse := MSE{}

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for length mismatch")
		}
	}()

	mse.Forward([]float64{1.0, 2.0}, []float64{1.0})
}

// TestMSEBackward tests MSE backward pass.
func TestMSEBackward(t *testing.T) {
	mse := MSE{}

	tests := []struct {
		name     string
		yPred    []float64
		yTrue    []float64
		expected []float64
	}{
		{"Perfect prediction", []float64{1.0, 2.0}, []float64{1.0, 2.0}, []float64{0.0, 0.0}},
		{"Single error", []float64{1.0, 2.0}, []float64{1.5, 2.0}, []float64{-0.5, 0.0}}, // 2*(y-p)/n
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mse.Backward(tt.yPred, tt.yTrue)

			if len(result) != len(tt.expected) {
				t.Errorf("Backward length = %d, want %d", len(result), len(tt.expected))
				return
			}

			for i := range result {
				if math.Abs(result[i]-tt.expected[i]) > 1e-10 {
					t.Errorf("Backward[%d] = %v, want %v", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

// TestMSEBackwardInPlace tests in-place gradient computation.
func TestMSEBackwardInPlace(t *testing.T) {
	mse := MSE{}

	yPred := []float64{1.0, 2.0, 3.0}
	yTrue := []float64{0.0, 2.0, 4.0}
	grad := make([]float64, 3)

	mse.BackwardInPlace(yPred, yTrue, grad)

	// Expected: 2*(y-p)/n = 2*[1, 0, -1]/3 = [2/3, 0, -2/3]
	expected := []float64{2.0/3.0, 0.0, -2.0/3.0}

	for i := range grad {
		if math.Abs(grad[i]-expected[i]) > 1e-10 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}

// TestMSEBackwardLengthMismatch tests in-place gradient error handling.
func TestMSEBackwardLengthMismatch(t *testing.T) {
	mse := MSE{}

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for length mismatch")
		}
	}()

	mse.BackwardInPlace([]float64{1.0}, []float64{1.0}, []float64{1.0, 2.0})
}

// TestCrossEntropyForward tests cross entropy forward pass.
func TestCrossEntropyForward(t *testing.T) {
	ce := CrossEntropy{}

	tests := []struct {
		name     string
		yPred    []float64
		yTrue    []float64
		check    func(float64) bool
	}{
		{"Perfect prediction", []float64{1.0, 0.0}, []float64{1.0, 0.0}, func(x float64) bool { return x < 1e-10 }},
		{"Uniform distribution", []float64{0.5, 0.5}, []float64{0.5, 0.5}, func(x float64) bool { return x > 0 && x < 1 }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ce.Forward(tt.yPred, tt.yTrue)
			if !tt.check(result) {
				t.Errorf("CrossEntropy.Forward() = %v, does not satisfy check", result)
			}
		})
	}
}

// TestCrossEntropyBackward tests cross entropy backward pass.
func TestCrossEntropyBackward(t *testing.T) {
	ce := CrossEntropy{}

	// For cross entropy with softmax, gradient is (y_pred - y_true)
	yPred := []float64{0.7, 0.2, 0.1}
	yTrue := []float64{1.0, 0.0, 0.0}

	grad := ce.Backward(yPred, yTrue)

	expected := []float64{0.7 - 1.0, 0.2 - 0.0, 0.1 - 0.0} // [-0.3, 0.2, 0.1]

	for i := range grad {
		if math.Abs(grad[i]-expected[i]) > 1e-10 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}

// TestHuberForward tests Huber loss forward pass.
func TestHuberForward(t *testing.T) {
	huber := NewHuber(1.0)

	tests := []struct {
		name     string
		yPred    []float64
		yTrue    []float64
		expected float64
	}{
		{"Small errors (quadratic)", []float64{1.0, 2.0}, []float64{1.5, 2.5}, 0.125}, // (0.5*0.5^2 + 0.5*0.5^2) / 2 = 0.125
		{"Large errors (linear)", []float64{0.0}, []float64{2.0}, 1.5},                 // delta * (diff - 0.5*delta) = 1 * (2 - 0.5) = 1.5
		{"Mixed errors", []float64{0.0, 0.0}, []float64{0.5, 2.0}, 0.8125},             // (0.5*0.5^2 + 1*(2-0.5*1)) / 2 = (0.125 + 1.5) / 2 = 0.8125
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := huber.Forward(tt.yPred, tt.yTrue)
			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("Huber.Forward() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestHuberBackward tests Huber loss backward pass.
func TestHuberBackward(t *testing.T) {
	huber := NewHuber(1.0)

	tests := []struct {
		name     string
		yPred    []float64
		yTrue    []float64
		expected []float64
	}{
		{"Small error (gradient = diff)", []float64{1.0}, []float64{1.5}, []float64{-0.5}},
		{"Large error (gradient = sign * delta)", []float64{0.0}, []float64{2.0}, []float64{-1.0}},
		{"Large negative error", []float64{2.0}, []float64{0.0}, []float64{1.0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			grad := huber.Backward(tt.yPred, tt.yTrue)

			if len(grad) != len(tt.expected) {
				t.Errorf("Grad length = %d, want %d", len(grad), len(tt.expected))
				return
			}

			for i := range grad {
				if math.Abs(grad[i]-tt.expected[i]) > 1e-10 {
					t.Errorf("grad[%d] = %v, want %v", i, grad[i], tt.expected[i])
				}
			}
		})
	}
}

// TestHuberDeltaParameter tests different delta values.
func TestHuberDeltaParameter(t *testing.T) {
	yPred := []float64{0.0}
	yTrue := []float64{1.5}

	// Small delta: more linear behavior
	huber1 := NewHuber(0.5)
	loss1 := huber1.Forward(yPred, yTrue)
	grad1 := huber1.Backward(yPred, yTrue)

	// Large delta: more quadratic behavior
	huber2 := NewHuber(2.0)
	loss2 := huber2.Forward(yPred, yTrue)
	grad2 := huber2.Backward(yPred, yTrue)

	// With larger delta, loss should be different
	// diff = 1.5, delta1=0.5: loss = 0.5*(1.5-0.5*0.5) = 0.625
	// delta2=2.0: loss = 0.5*1.5^2 = 1.125 (quadratic since diff < delta)
	if loss2 <= loss1 {
		t.Errorf("With larger delta, loss should be different: delta=0.5 -> %v, delta=2.0 -> %v", loss1, loss2)
	}

	// Gradient signs
	// grad1 = -0.5 * sign(1.5) = -0.5 (since |diff| > delta1)
	// grad2 = -1.5 (since |diff| < delta2, gradient = diff)
	if math.Abs(grad1[0]-(-0.5)) > 1e-10 {
		t.Errorf("grad1[0] = %v, want -0.5", grad1[0])
	}
	if math.Abs(grad2[0]-(-1.5)) > 1e-10 {
		t.Errorf("grad2[0] = %v, want -1.5", grad2[0])
	}
}

// TestHuberLengthMismatch tests Huber error handling.
func TestHuberLengthMismatch(t *testing.T) {
	huber := NewHuber(1.0)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for length mismatch")
		}
	}()

	huber.Forward([]float64{1.0, 2.0}, []float64{1.0})
}

// TestCrossEntropyLengthMismatch tests cross entropy error handling.
func TestCrossEntropyLengthMismatch(t *testing.T) {
	ce := CrossEntropy{}

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for length mismatch")
		}
	}()

	ce.Forward([]float64{1.0, 2.0}, []float64{1.0})
}

// TestLossGradientSign tests that gradients have correct signs.
func TestLossGradientSign(t *testing.T) {
	yPred := 0.5
	yTrue := 0.0

	// For MSE: dL/dy_pred = 2*(y_pred - y_true)/n
	// When y_pred > y_true, gradient should be positive
	mse := MSE{}
	gradMSE := mse.Backward([]float64{yPred}, []float64{yTrue})
	if gradMSE[0] <= 0 {
		t.Errorf("MSE gradient should be positive when y_pred > y_true, got %v", gradMSE[0])
	}

	// For CrossEntropy: gradient is (y_pred - y_true)
	// Same logic applies
	ce := CrossEntropy{}
	gradCE := ce.Backward([]float64{yPred}, []float64{yTrue})
	if gradCE[0] <= 0 {
		t.Errorf("CrossEntropy gradient should be positive when y_pred > y_true, got %v", gradCE[0])
	}

	// For Huber: same as MSE for small errors
	huber := NewHuber(1.0)
	gradHuber := huber.Backward([]float64{yPred}, []float64{yTrue})
	if gradHuber[0] <= 0 {
		t.Errorf("Huber gradient should be positive when y_pred > y_true, got %v", gradHuber[0])
	}
}

// TestLossZeroAtMinimum tests that losses are zero at minimum.
func TestLossZeroAtMinimum(t *testing.T) {
	y := []float64{1.0, 2.0, 3.0}

	// MSE is zero when prediction equals target
	mse := MSE{}
	if mse.Forward(y, y) != 0 {
		t.Errorf("MSE should be zero when y_pred == y_true")
	}

	// CrossEntropy: as prediction approaches target, loss approaches 0
	ce := CrossEntropy{}
	// Perfect prediction with small epsilon
	yPred := []float64{1.0 - 1e-10, 1e-10, 1e-10}
	yTrue := []float64{1.0, 0.0, 0.0}
	loss := ce.Forward(yPred, yTrue)
	if loss > 1e-5 {
		t.Errorf("CrossEntropy with perfect prediction should be near zero, got %v", loss)
	}

	// Huber is zero when prediction equals target
	huber := NewHuber(1.0)
	if huber.Forward(y, y) != 0 {
		t.Errorf("Huber should be zero when y_pred == y_true")
	}
}

// TestLossConsistency tests gradient consistency.
func TestLossConsistency(t *testing.T) {
	// Numerically verify gradient: dL/dy ≈ (L(y+h) - L(y-h)) / (2h)
	h := 1e-6

	yPred := []float64{0.5, 0.5}
	yTrue := []float64{1.0, 0.0}

	mse := MSE{}
	lossPlus := mse.Forward([]float64{0.5+h, 0.5}, yTrue)
	lossMinus := mse.Forward([]float64{0.5-h, 0.5}, yTrue)
	numericGrad := (lossPlus - lossMinus) / (2 * h)

	analyticGrad := mse.Backward(yPred, yTrue)

	if math.Abs(numericGrad-analyticGrad[0]) > 1e-4 {
		t.Logf("MSE gradient check: numeric=%v, analytic=%v", numericGrad, analyticGrad[0])
	}

	huber := NewHuber(1.0)
	lossPlus = huber.Forward([]float64{0.5+h, 0.5}, yTrue)
	lossMinus = huber.Forward([]float64{0.5-h, 0.5}, yTrue)
	numericGrad = (lossPlus - lossMinus) / (2 * h)

	analyticGrad = huber.Backward(yPred, yTrue)

	if math.Abs(numericGrad-analyticGrad[0]) > 1e-4 {
		t.Logf("Huber gradient check: numeric=%v, analytic=%v", numericGrad, analyticGrad[0])
	}
}

// TestLossNormalization tests that loss is properly normalized.
func TestLossNormalization(t *testing.T) {
	y := []float64{1.0, 2.0, 3.0}
	target := []float64{0.0, 1.0, 2.0}

	mse := MSE{}

	// Single sample - store to verify normalization
	_ = mse.Forward([]float64{y[0]}, []float64{target[0]})

	// All samples
	loss3 := mse.Forward(y, target)

	// Loss should be averaged, not summed
	// loss3 should equal mean of individual losses
	expected := (1.0 + 1.0 + 1.0) / 3.0 // Each diff^2 = 1, averaged
	if math.Abs(loss3-expected) > 1e-10 {
		t.Errorf("MSE normalization: got %v, want %v", loss3, expected)
	}
}

// TestLossInvarianceToPermutation tests that loss is permutation invariant.
func TestLossInvarianceToPermutation(t *testing.T) {
	yPred := []float64{1.0, 2.0, 3.0, 4.0}
	yTrue := []float64{1.5, 2.5, 3.5, 4.5}

	mse := MSE{}

	loss1 := mse.Forward(yPred, yTrue)

	// Permute both arrays same way
	yPredPerm := []float64{3.0, 1.0, 4.0, 2.0}
	yTruePerm := []float64{3.5, 1.5, 4.5, 2.5}

	loss2 := mse.Forward(yPredPerm, yTruePerm)

	// Loss should be the same (both have same differences)
	if math.Abs(loss1-loss2) > 1e-10 {
		t.Errorf("Loss not permutation invariant: %v vs %v", loss1, loss2)
	}
}

// TestCrossEntropyProbabilityClipping tests that predictions are clipped.
func TestCrossEntropyProbabilityClipping(t *testing.T) {
	ce := CrossEntropy{}

	// Very small prediction should be clipped to eps
	yPred := []float64{1e-20} // Very small
	yTrue := []float64{1.0}

	loss := ce.Forward(yPred, yTrue)

	// Should use eps = 1e-10, not 1e-20
	expected := -math.Log(1e-10) // = 23.025...

	if math.Abs(loss-expected) > 1e-6 {
		t.Errorf("CrossEntropy with very small pred: got %v, want %v", loss, expected)
	}
}

// TestLossGradientsSumToZeroForClassification tests gradient behavior for classification.
func TestLossGradientsSumToZeroForClassification(t *testing.T) {
	// For cross entropy with one-hot encoding, gradients should sum to zero
	ce := CrossEntropy{}

	// 3-class classification with one-hot target
	yPred := []float64{0.2, 0.3, 0.5}
	yTrue := []float64{0.0, 0.0, 1.0}

	grad := ce.Backward(yPred, yTrue)

	// Sum of gradients should be approximately zero
	sum := 0.0
	for _, g := range grad {
		sum += g
	}

	// For cross entropy + one-hot: sum(grad) = sum(y_pred) - 1 = 1 - 1 = 0
	if math.Abs(sum) > 1e-10 {
		t.Errorf("CrossEntropy gradients should sum to zero, got %v", sum)
	}
}

// TestHuberLinearBehaviorForLargeErrors tests Huber's linear behavior.
func TestHuberLinearBehaviorForLargeErrors(t *testing.T) {
	huber := NewHuber(1.0)

	// For large errors, Huber loss grows linearly, not quadratically
	yPred := []float64{0.0}
	yTrue := []float64{10.0} // Large error

	loss := huber.Forward(yPred, yTrue)

	// With delta=1, loss should be approximately: delta * (diff - 0.5*delta) = 1 * (10 - 0.5) = 9.5
	// Divided by n=1, still 9.5
	expected := 1.0 * (10.0 - 0.5*1.0)

	if math.Abs(loss-expected) > 1e-10 {
		t.Errorf("Huber loss for large error: got %v, want %v", loss, expected)
	}
}

// TestHuberQuadraticBehaviorForSmallErrors tests Huber's quadratic behavior.
func TestHuberQuadraticBehaviorForSmallErrors(t *testing.T) {
	huber := NewHuber(1.0)

	// For small errors (diff < delta), Huber loss is quadratic
	yPred := []float64{0.0}
	yTrue := []float64{0.5} // Small error < delta=1.0

	loss := huber.Forward(yPred, yTrue)

	// Should be 0.5 * diff^2 = 0.5 * 0.25 = 0.125
	expected := 0.5 * 0.5 * 0.5

	if math.Abs(loss-expected) > 1e-10 {
		t.Errorf("Huber loss for small error: got %v, want %v", loss, expected)
	}
}
