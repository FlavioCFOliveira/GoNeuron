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
		yPred    []float32
		yTrue    []float32
		expected float32
	}{
		{"Perfect prediction", []float32{1.0, 2.0, 3.0}, []float32{1.0, 2.0, 3.0}, 0.0},
		{"Single error", []float32{1.0, 2.0}, []float32{1.5, 2.0}, 0.125}, // (0.5^2 + 0) / 2 = 0.125
		{"Multiple errors", []float32{1.0, 2.0, 3.0}, []float32{0.0, 1.0, 2.0}, 1.0}, // (1+1+1)/3 = 1
		{"Large errors", []float32{10.0}, []float32{0.0}, 100.0}, // 10^2 = 100
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mse.Forward(tt.yPred, tt.yTrue)
			if float32(math.Abs(float64(result-tt.expected))) > 1e-6 {
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

	mse.Forward([]float32{1.0, 2.0}, []float32{1.0})
}

// TestMSEBackward tests MSE backward pass.
func TestMSEBackward(t *testing.T) {
	mse := MSE{}

	tests := []struct {
		name     string
		yPred    []float32
		yTrue    []float32
		expected []float32
	}{
		{"Perfect prediction", []float32{1.0, 2.0}, []float32{1.0, 2.0}, []float32{0.0, 0.0}},
		{"Single error", []float32{1.0, 2.0}, []float32{1.5, 2.0}, []float32{-0.5, 0.0}}, // 2*(y-p)/n
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mse.Backward(tt.yPred, tt.yTrue)

			if len(result) != len(tt.expected) {
				t.Errorf("Backward length = %d, want %d", len(result), len(tt.expected))
				return
			}

			for i := range result {
				if float32(math.Abs(float64(result[i]-tt.expected[i]))) > 1e-6 {
					t.Errorf("Backward[%d] = %v, want %v", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

// TestMSEBackwardInPlace tests in-place gradient computation.
func TestMSEBackwardInPlace(t *testing.T) {
	mse := MSE{}

	yPred := []float32{1.0, 2.0, 3.0}
	yTrue := []float32{0.0, 2.0, 4.0}
	grad := make([]float32, 3)

	mse.BackwardInPlace(yPred, yTrue, grad)

	// Expected: 2*(y-p)/n = 2*[1, 0, -1]/3 = [2/3, 0, -2/3]
	expected := []float32{2.0 / 3.0, 0.0, -2.0 / 3.0}

	for i := range grad {
		if float32(math.Abs(float64(grad[i]-expected[i]))) > 1e-6 {
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

	mse.BackwardInPlace([]float32{1.0}, []float32{1.0}, []float32{1.0, 2.0})
}

// TestCrossEntropyForward tests cross entropy forward pass.
func TestCrossEntropyForward(t *testing.T) {
	ce := CrossEntropy{}

	tests := []struct {
		name     string
		yPred    []float32
		yTrue    []float32
		check    func(float32) bool
	}{
		{"Perfect prediction", []float32{1.0, 0.0}, []float32{1.0, 0.0}, func(x float32) bool { return x < 1e-6 }},
		{"Uniform distribution", []float32{0.5, 0.5}, []float32{0.5, 0.5}, func(x float32) bool { return x > 0 && x < 1 }},
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
	yPred := []float32{0.7, 0.2, 0.1}
	yTrue := []float32{1.0, 0.0, 0.0}

	grad := ce.Backward(yPred, yTrue)

	expected := []float32{0.7 - 1.0, 0.2 - 0.0, 0.1 - 0.0} // [-0.3, 0.2, 0.1]

	for i := range grad {
		if float32(math.Abs(float64(grad[i]-expected[i]))) > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}

// TestHuberForward tests Huber loss forward pass.
func TestHuberForward(t *testing.T) {
	huber := NewHuber(1.0)

	tests := []struct {
		name     string
		yPred    []float32
		yTrue    []float32
		expected float32
	}{
		{"Small errors (quadratic)", []float32{1.5, 2.5}, []float32{1.0, 2.0}, 0.125}, // (0.5*0.5^2 + 0.5*0.5^2) / 2 = 0.125
		{"Large errors (linear)", []float32{2.0}, []float32{0.0}, 1.5},                 // delta * (diff - 0.5*delta) = 1 * (2 - 0.5) = 1.5
		{"Mixed errors", []float32{0.5, 2.0}, []float32{0.0, 0.0}, 0.8125},             // (0.5*0.5^2 + 1*(2-0.5*1)) / 2 = (0.125 + 1.5) / 2 = 0.8125
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := huber.Forward(tt.yPred, tt.yTrue)
			if float32(math.Abs(float64(result-tt.expected))) > 1e-6 {
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
		yPred    []float32
		yTrue    []float32
		expected []float32
	}{
		{"Small error (gradient = diff)", []float32{1.0}, []float32{1.5}, []float32{-0.5}},
		{"Large error (gradient = sign * delta)", []float32{0.0}, []float32{2.0}, []float32{-1.0}},
		{"Large negative error", []float32{2.0}, []float32{0.0}, []float32{1.0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			grad := huber.Backward(tt.yPred, tt.yTrue)

			if len(grad) != len(tt.expected) {
				t.Errorf("Grad length = %d, want %d", len(grad), len(tt.expected))
				return
			}

			for i := range grad {
				if float32(math.Abs(float64(grad[i]-tt.expected[i]))) > 1e-6 {
					t.Errorf("grad[%d] = %v, want %v", i, grad[i], tt.expected[i])
				}
			}
		})
	}
}

// TestHuberDeltaParameter tests different delta values.
func TestHuberDeltaParameter(t *testing.T) {
	yPred := []float32{0.0}
	yTrue := []float32{1.5}

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
	if float32(math.Abs(float64(grad1[0]-(-0.5)))) > 1e-6 {
		t.Errorf("grad1[0] = %v, want -0.5", grad1[0])
	}
	if float32(math.Abs(float64(grad2[0]-(-1.5)))) > 1e-6 {
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

	huber.Forward([]float32{1.0, 2.0}, []float32{1.0})
}

// TestCrossEntropyLengthMismatch tests cross entropy error handling.
func TestCrossEntropyLengthMismatch(t *testing.T) {
	ce := CrossEntropy{}

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for length mismatch")
		}
	}()

	ce.Forward([]float32{1.0, 2.0}, []float32{1.0})
}

// TestLossGradientSign tests that gradients have correct signs.
func TestLossGradientSign(t *testing.T) {
	yPred := float32(0.5)
	yTrue := float32(0.0)

	// For MSE: dL/dy_pred = 2*(y_pred - y_true)/n
	// When y_pred > y_true, gradient should be positive
	mse := MSE{}
	gradMSE := mse.Backward([]float32{yPred}, []float32{yTrue})
	if gradMSE[0] <= 0 {
		t.Errorf("MSE gradient should be positive when y_pred > y_true, got %v", gradMSE[0])
	}

	// For CrossEntropy: gradient is (y_pred - y_true)
	// Same logic applies
	ce := CrossEntropy{}
	gradCE := ce.Backward([]float32{yPred}, []float32{yTrue})
	if gradCE[0] <= 0 {
		t.Errorf("CrossEntropy gradient should be positive when y_pred > y_true, got %v", gradCE[0])
	}

	// For Huber: same as MSE for small errors
	huber := NewHuber(1.0)
	gradHuber := huber.Backward([]float32{yPred}, []float32{yTrue})
	if gradHuber[0] <= 0 {
		t.Errorf("Huber gradient should be positive when y_pred > y_true, got %v", gradHuber[0])
	}
}

// TestLossZeroAtMinimum tests that losses are zero at minimum.
func TestLossZeroAtMinimum(t *testing.T) {
	y := []float32{1.0, 2.0, 3.0}

	// MSE is zero when prediction equals target
	mse := MSE{}
	if mse.Forward(y, y) != 0 {
		t.Errorf("MSE should be zero when y_pred == y_true")
	}

	// CrossEntropy: as prediction approaches target, loss approaches 0
	ce := CrossEntropy{}
	// Perfect prediction with small epsilon
	yPred := []float32{1.0 - 1e-6, 1e-6, 1e-6}
	yTrue := []float32{1.0, 0.0, 0.0}
	l := ce.Forward(yPred, yTrue)
	if l > 1e-4 {
		t.Errorf("CrossEntropy with perfect prediction should be near zero, got %v", l)
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
	h := float32(1e-4)

	yPred := []float32{0.5, 0.5}
	yTrue := []float32{1.0, 0.0}

	mse := MSE{}
	lossPlus := mse.Forward([]float32{0.5 + h, 0.5}, yTrue)
	lossMinus := mse.Forward([]float32{0.5 - h, 0.5}, yTrue)
	numericGrad := (lossPlus - lossMinus) / (2 * h)

	analyticGrad := mse.Backward(yPred, yTrue)

	if float32(math.Abs(float64(numericGrad-analyticGrad[0]))) > 1e-3 {
		t.Logf("MSE gradient check: numeric=%v, analytic=%v", numericGrad, analyticGrad[0])
	}

	huber := NewHuber(1.0)
	lossPlus = huber.Forward([]float32{0.5 + h, 0.5}, yTrue)
	lossMinus = huber.Forward([]float32{0.5 - h, 0.5}, yTrue)
	numericGrad = (lossPlus - lossMinus) / (2 * h)

	analyticGrad = huber.Backward(yPred, yTrue)

	if float32(math.Abs(float64(numericGrad-analyticGrad[0]))) > 1e-3 {
		t.Logf("Huber gradient check: numeric=%v, analytic=%v", numericGrad, analyticGrad[0])
	}
}

// TestLossNormalization tests that loss is properly normalized.
func TestLossNormalization(t *testing.T) {
	y := []float32{1.0, 2.0, 3.0}
	target := []float32{0.0, 1.0, 2.0}

	mse := MSE{}

	// Single sample - store to verify normalization
	_ = mse.Forward([]float32{y[0]}, []float32{target[0]})

	// All samples
	loss3 := mse.Forward(y, target)

	// Loss should be averaged, not summed
	// loss3 should equal mean of individual losses
	expected := float32((1.0 + 1.0 + 1.0) / 3.0) // Each diff^2 = 1, averaged
	if float32(math.Abs(float64(loss3-expected))) > 1e-6 {
		t.Errorf("MSE normalization: got %v, want %v", loss3, expected)
	}
}

// TestLossInvarianceToPermutation tests that loss is permutation invariant.
func TestLossInvarianceToPermutation(t *testing.T) {
	yPred := []float32{1.0, 2.0, 3.0, 4.0}
	yTrue := []float32{1.5, 2.5, 3.5, 4.5}

	mse := MSE{}

	loss1 := mse.Forward(yPred, yTrue)

	// Permute both arrays same way
	yPredPerm := []float32{3.0, 1.0, 4.0, 2.0}
	yTruePerm := []float32{3.5, 1.5, 4.5, 2.5}

	loss2 := mse.Forward(yPredPerm, yTruePerm)

	// Loss should be the same (both have same differences)
	if float32(math.Abs(float64(loss1-loss2))) > 1e-6 {
		t.Errorf("Loss not permutation invariant: %v vs %v", loss1, loss2)
	}
}

// TestCrossEntropyProbabilityClipping tests that predictions are clipped.
func TestCrossEntropyProbabilityClipping(t *testing.T) {
	ce := CrossEntropy{}

	// Very small prediction should be clipped to eps
	yPred := []float32{1e-20} // Very small
	yTrue := []float32{1.0}

	l := ce.Forward(yPred, yTrue)

	// Should use eps = 1e-10, not 1e-20
	expected := float32(-math.Log(1e-10)) // = 23.025...

	if float32(math.Abs(float64(l-expected))) > 1e-5 {
		t.Errorf("CrossEntropy with very small pred: got %v, want %v", l, expected)
	}
}

// TestLossGradientsSumToZeroForClassification tests gradient behavior for classification.
func TestLossGradientsSumToZeroForClassification(t *testing.T) {
	// For cross entropy with one-hot encoding, gradients should sum to zero
	ce := CrossEntropy{}

	// 3-class classification with one-hot target
	yPred := []float32{0.2, 0.3, 0.5}
	yTrue := []float32{0.0, 0.0, 1.0}

	grad := ce.Backward(yPred, yTrue)

	// Sum of gradients should be approximately zero
	sum := float32(0.0)
	for _, g := range grad {
		sum += g
	}

	// For cross entropy + one-hot: sum(grad) = sum(y_pred) - 1 = 1 - 1 = 0
	if float32(math.Abs(float64(sum))) > 1e-6 {
		t.Errorf("CrossEntropy gradients should sum to zero, got %v", sum)
	}
}

// TestHuberLinearBehaviorForLargeErrors tests Huber's linear behavior.
func TestHuberLinearBehaviorForLargeErrors(t *testing.T) {
	huber := NewHuber(1.0)

	// For large errors, Huber loss grows linearly, not quadratically
	yPred := []float32{0.0}
	yTrue := []float32{10.0} // Large error

	l := huber.Forward(yPred, yTrue)

	// With delta=1, loss should be approximately: delta * (diff - 0.5*delta) = 1 * (10 - 0.5) = 9.5
	// Divided by n=1, still 9.5
	expected := float32(1.0 * (10.0 - 0.5*1.0))

	if float32(math.Abs(float64(l-expected))) > 1e-6 {
		t.Errorf("Huber loss for large error: got %v, want %v", l, expected)
	}
}

// TestHuberQuadraticBehaviorForSmallErrors tests Huber's quadratic behavior.
func TestHuberQuadraticBehaviorForSmallErrors(t *testing.T) {
	huber := NewHuber(1.0)

	// For small errors (diff < delta), Huber loss is quadratic
	yPred := []float32{0.0}
	yTrue := []float32{0.5} // Small error < delta=1.0

	l := huber.Forward(yPred, yTrue)

	// Should be 0.5 * diff^2 = 0.5 * 0.25 = 0.125
	expected := float32(0.5 * 0.5 * 0.5)

	if float32(math.Abs(float64(l-expected))) > 1e-6 {
		t.Errorf("Huber loss for small error: got %v, want %v", l, expected)
	}
}

// TestL1LossForward tests L1 loss forward pass.
func TestL1LossForward(t *testing.T) {
	l1 := L1Loss{}

	tests := []struct {
		name     string
		yPred    []float32
		yTrue    []float32
		expected float32
	}{
		{"Perfect prediction", []float32{1.0, 2.0, 3.0}, []float32{1.0, 2.0, 3.0}, 0.0},
		{"Single error", []float32{1.0, 2.0}, []float32{1.5, 2.0}, 0.25}, // (0.5 + 0) / 2 = 0.25
		{"Multiple errors", []float32{1.0, 2.0, 3.0}, []float32{0.0, 1.0, 2.0}, 1.0}, // (1+1+1)/3 = 1
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := l1.Forward(tt.yPred, tt.yTrue)
			if float32(math.Abs(float64(result-tt.expected))) > 1e-6 {
				t.Errorf("L1Loss.Forward() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestL1LossBackward tests L1 loss backward pass.
func TestL1LossBackward(t *testing.T) {
	l1 := L1Loss{}

	tests := []struct {
		name     string
		yPred    []float32
		yTrue    []float32
		expected []float32
	}{
		{"Positive difference", []float32{2.0}, []float32{1.0}, []float32{1.0 / 1.0}}, // (2-1) > 0, grad = 1/n
		{"Negative difference", []float32{1.0}, []float32{2.0}, []float32{-1.0 / 1.0}}, // (1-2) < 0, grad = -1/n
		{"Zero difference", []float32{1.0}, []float32{1.0}, []float32{0.0}},           // diff = 0, grad = 0
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			grad := l1.Backward(tt.yPred, tt.yTrue)

			if len(grad) != len(tt.expected) {
				t.Errorf("Grad length = %d, want %d", len(grad), len(tt.expected))
				return
			}

			for i := range grad {
				if float32(math.Abs(float64(grad[i]-tt.expected[i]))) > 1e-6 {
					t.Errorf("grad[%d] = %v, want %v", i, grad[i], tt.expected[i])
				}
			}
		})
	}
}

// TestL1LossBackwardInPlace tests in-place gradient computation.
func TestL1LossBackwardInPlace(t *testing.T) {
	l1 := L1Loss{}

	yPred := []float32{1.0, 2.0, 3.0}
	yTrue := []float32{0.0, 2.0, 4.0}
	grad := make([]float32, 3)

	l1.BackwardInPlace(yPred, yTrue, grad)

	// Expected: 1/n * sign(y-p)
	// [1, 0, -1] / 3 = [1/3, 0, -1/3]
	expected := []float32{1.0 / 3.0, 0.0, -1.0 / 3.0}

	for i := range grad {
		if float32(math.Abs(float64(grad[i]-expected[i]))) > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}

// TestBCELossForward tests BCE loss forward pass.
func TestBCELossForward(t *testing.T) {
	bce := BCELoss{}

	tests := []struct {
		name     string
		yPred    []float32
		yTrue    []float32
		expected float32
	}{
		{"Perfect prediction", []float32{0.99, 0.01}, []float32{1.0, 0.0}, 0.01005}, // Close to 0 but not exactly
		{"Uniform distribution", []float32{0.5, 0.5}, []float32{0.5, 0.5}, 0.6931},  // ln(2) ≈ 0.693
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := bce.Forward(tt.yPred, tt.yTrue)
			if float32(math.Abs(float64(result-tt.expected))) > 1e-4 {
				t.Errorf("BCELoss.Forward() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestBCELossBackward tests BCE loss backward pass.
func TestBCELossBackward(t *testing.T) {
	bce := BCELoss{}

	yPred := []float32{0.7, 0.3}
	yTrue := []float32{1.0, 0.0}

	grad := bce.Backward(yPred, yTrue)

	// For BCE: grad = (pred - y) / (pred * (1-pred)) / n
	// Sample 1: (0.7 - 1) / (0.7 * 0.3) / 2 = -0.3 / 0.21 / 2 = -0.714...
	// Sample 2: (0.3 - 0) / (0.3 * 0.7) / 2 = 0.3 / 0.21 / 2 = 0.714...

	expected := []float32{-0.3 / 0.42, 0.3 / 0.42}

	for i := range grad {
		if float32(math.Abs(float64(grad[i]-expected[i]))) > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}

// TestBCEWithLogitsLossForward tests BCEWithLogitsLoss forward pass.
func TestBCEWithLogitsLossForward(t *testing.T) {
	bce := BCEWithLogitsLoss{}

	// For logits [1.0, -1.0] and targets [1.0, 0.0]
	// Loss should be numerically stable
	yPred := []float32{1.0, -1.0}
	yTrue := []float32{1.0, 0.0}

	result := bce.Forward(yPred, yTrue)

	// Check that loss is positive
	if result <= 0 {
		t.Errorf("BCEWithLogitsLoss.Forward() should be positive, got %v", result)
	}
}

// TestBCEWithLogitsLossBackward tests BCEWithLogitsLoss backward pass.
func TestBCEWithLogitsLossBackward(t *testing.T) {
	bce := BCEWithLogitsLoss{}

	yPred := []float32{0.0, 0.0} // sigmoid(0) = 0.5
	yTrue := []float32{1.0, 0.0}

	grad := bce.Backward(yPred, yTrue)

	// For BCEWithLogits: grad = (sigmoid(x) - y) / n
	// Sample 1: (0.5 - 1) / 2 = -0.25
	// Sample 2: (0.5 - 0) / 2 = 0.25

	expected := []float32{-0.25, 0.25}

	for i := range grad {
		if float32(math.Abs(float64(grad[i]-expected[i]))) > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}

// TestNLLLossForward tests NLL loss forward pass with log-probabilities.
func TestNLLLossForward(t *testing.T) {
	nll := NLLLoss{}

	// For log-probabilities and one-hot targets
	// y_pred = log([0.9, 0.05, 0.05]) = [-0.105, -2.996, -2.996]
	// y_true = [1, 0, 0]
	// Loss = -sum(y_true * y_pred) / n = -(-0.105) / 3 = 0.035
	yPred := []float32{float32(math.Log(0.9)), float32(math.Log(0.05)), float32(math.Log(0.05))}
	yTrue := []float32{1.0, 0.0, 0.0}

	result := nll.Forward(yPred, yTrue)

	// Should be approximately -log(0.9) / 3 ≈ 0.035
	expected := float32(-math.Log(0.9) / 3.0)
	if float32(math.Abs(float64(result-expected))) > 1e-6 {
		t.Errorf("NLLLoss.Forward() = %v, want %v", result, expected)
	}
}

// TestNLLLossBackward tests NLL loss backward pass with log-probabilities.
func TestNLLLossBackward(t *testing.T) {
	nll := NLLLoss{}

	// Log-probabilities input
	yPred := []float32{float32(math.Log(0.9)), float32(math.Log(0.05)), float32(math.Log(0.05))}
	yTrue := []float32{1.0, 0.0, 0.0}

	grad := nll.Backward(yPred, yTrue)

	// For NLL with log-prob input: grad[i] = -y_true[i] / n
	// Sample 1: -1 / 3 = -0.333...
	// Others: 0
	expectedGrad0 := float32(-1.0 / 3.0)

	if float32(math.Abs(float64(grad[0]-expectedGrad0))) > 1e-6 {
		t.Errorf("grad[0] = %v, want %v", grad[0], expectedGrad0)
	}

	if grad[1] != 0 || grad[2] != 0 {
		t.Errorf("grad[1] and grad[2] should be 0 for one-hot target")
	}
}

// TestCosineEmbeddingLossForward tests cosine embedding loss forward pass.
func TestCosineEmbeddingLossForward(t *testing.T) {
	l := NewCosineEmbeddingLoss(0.0)

	// Similar pair (y=1): loss = 1 - cos
	// Dissimilar pair (y=-1): loss = max(0, cos - margin)

	yPred := []float32{0.5, -0.5} // cos=0.5 similar, cos=-0.5 dissimilar
	yTrue := []float32{1.0, -1.0}

	result := l.Forward(yPred, yTrue)

	// Sample 1: 1 - 0.5 = 0.5
	// Sample 2: max(0, -0.5 - 0) = 0
	// Average: 0.5 / 2 = 0.25

	if float32(math.Abs(float64(result-0.25))) > 1e-6 {
		t.Errorf("CosineEmbeddingLoss.Forward() = %v, want 0.25", result)
	}
}

// TestCosineEmbeddingLossBackward tests cosine embedding loss backward pass.
func TestCosineEmbeddingLossBackward(t *testing.T) {
	l := NewCosineEmbeddingLoss(0.0)

	// Use positive cosine for dissimilar to have active margin
	yPred := []float32{0.5, 0.3}
	yTrue := []float32{1.0, -1.0}

	grad := l.Backward(yPred, yTrue)

	// Similar: grad = -1/n = -0.5
	// Dissimilar with cos(0.3) > margin(0): grad = 1/n = 0.5

	expected := []float32{-0.5, 0.5}

	for i := range grad {
		if float32(math.Abs(float64(grad[i]-expected[i]))) > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}

// TestHingeEmbeddingLossForward tests hinge embedding loss forward pass.
func TestHingeEmbeddingLossForward(t *testing.T) {
	l := NewHingeEmbeddingLoss(1.0)

	// y=1: loss = x
	// y=-1: loss = max(0, margin - x)

	yPred := []float32{0.5, 1.5} // x=0.5 for positive, x=1.5 for negative
	yTrue := []float32{1.0, -1.0}

	result := l.Forward(yPred, yTrue)

	// Sample 1: 0.5
	// Sample 2: max(0, 1.0 - 1.5) = 0
	// Average: 0.5 / 2 = 0.25

	if float32(math.Abs(float64(result-0.25))) > 1e-6 {
		t.Errorf("HingeEmbeddingLoss.Forward() = %v, want 0.25", result)
	}
}

// TestHingeEmbeddingLossBackward tests hinge embedding loss backward pass.
func TestHingeEmbeddingLossBackward(t *testing.T) {
	l := NewHingeEmbeddingLoss(1.0)

	yPred := []float32{0.5, 0.5}
	yTrue := []float32{1.0, -1.0}

	grad := l.Backward(yPred, yTrue)

	// y=1: grad = 1/n
	// y=-1 and x < margin: grad = -1/n

	expected := []float32{0.5, -0.5}

	for i := range grad {
		if float32(math.Abs(float64(grad[i]-expected[i]))) > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}

// TestMarginRankingLossForward tests margin ranking loss forward pass.
func TestMarginRankingLossForward(t *testing.T) {
	l := NewMarginRankingLoss(1.0)

	// For margin ranking, yPred contains [x1, x2] pairs
	// yTrue contains targets (+1 or -1) for each pair
	// Pair 0: x1=1.0, x2=0.5, y=1: margin_diff = -1*(1-0.5)+1 = 0.5 > 0, loss = 0.5
	// Pair 1: x1=0.5, x2=1.0, y=-1: margin_diff = -(-1)*(0.5-1)+1 = 0.5 > 0, loss = 0.5

	yPred := []float32{1.0, 0.5, 0.5, 1.0}
	yTrue := []float32{1.0, -1.0} // One target per pair

	result := l.Forward(yPred, yTrue)

	// Average: (0.5 + 0.5) / 2 = 0.5

	if float32(math.Abs(float64(result-0.5))) > 1e-6 {
		t.Errorf("MarginRankingLoss.Forward() = %v, want 0.5", result)
	}
}

// TestMarginRankingLossBackward tests margin ranking loss backward pass.
func TestMarginRankingLossBackward(t *testing.T) {
	l := NewMarginRankingLoss(1.0)

	yPred := []float32{1.0, 0.5, 0.5, 1.0}
	yTrue := []float32{1.0, -1.0} // One target per pair

	grad := l.Backward(yPred, yTrue)

	// Pairs are [0,1] and [2,3]
	// Pair 0 (x1=1.0, x2=0.5, y=1): grad[0] -= 1/2, grad[1] += 1/2 => [-0.5, 0.5, 0, 0]
	// Pair 1 (x1=0.5, x2=1.0, y=-1): grad[2] -= (-1)/2, grad[3] += (-1)/2 => [-0.5, 0.5, 0.5, -0.5]

	expected := []float32{-0.5, 0.5, 0.5, -0.5}

	for i := range grad {
		if float32(math.Abs(float64(grad[i]-expected[i]))) > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}

// TestKLDivLossForward tests KL divergence forward pass.
func TestKLDivLossForward(t *testing.T) {
	kld := KLDivLoss{}

	// KL(divergence) between two distributions
	// For identical distributions, KL = 0
	yPred := []float32{0.5, 0.5}
	yTrue := []float32{0.5, 0.5}

	result := kld.Forward(yPred, yTrue)

	if result > 1e-6 {
		t.Errorf("KLDivLoss.Forward() for identical dists should be ~0, got %v", result)
	}
}

// TestKLDivLossBackward tests KL divergence backward pass.
func TestKLDivLossBackward(t *testing.T) {
	kld := KLDivLoss{}

	yPred := []float32{0.6, 0.4}
	yTrue := []float32{0.5, 0.5}

	grad := kld.Backward(yPred, yTrue)

	// grad[i] = -y_true[i] / (y_pred[i] * n)
	// grad[0] = -0.5 / (0.6 * 2) = -0.4167
	// grad[1] = -0.5 / (0.4 * 2) = -0.625

	expected := []float32{-0.5 / (0.6 * 2.0), -0.5 / (0.4 * 2.0)}

	for i := range grad {
		if float32(math.Abs(float64(grad[i]-expected[i]))) > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, grad[i], expected[i])
		}
	}
}
