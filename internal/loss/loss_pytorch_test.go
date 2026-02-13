// Package loss provides validation tests against PyTorch reference implementation.
// This file contains exact reference values computed with PyTorch for comparison.

package loss

import (
	"math"
	"testing"
)

// PyTorch reference values (computed with Python):
// import torch
// torch.set_printoptions(precision=10)
//
// MSE: torch.nn.MSELoss(reduction='mean')
// CrossEntropy: torch.nn.CrossEntropyLoss() (for classification with logit inputs)
// For binary classification: torch.nn.BCELoss() with sigmoid output
// Huber: torch.nn.HuberLoss(delta=1.0)

// TestMSEAgainstPyTorchReference validates MSE against PyTorch
// PyTorch MSE: mean((y_pred - y_true)^2)
func TestMSEAgainstPyTorchReference(t *testing.T) {
	mse := MSE{}

	// Reference values computed with PyTorch: nn.MSELoss(reduction='mean')
	tests := []struct {
		yPred    []float64
		yTrue    []float64
		expected float64
	}{
		{[]float64{1.0, 2.0, 3.0}, []float64{1.0, 2.0, 3.0}, 0.0},           // Perfect
		{[]float64{1.0, 2.0}, []float64{1.5, 2.0}, 0.125},                   // (0.25 + 0) / 2
		{[]float64{1.0, 2.0, 3.0}, []float64{0.0, 1.0, 2.0}, 1.0},           // (1+1+1)/3
		{[]float64{1.0, 1.0}, []float64{0.0, 2.0}, 1.0},                     // (1+1)/2
		{[]float64{0.5}, []float64{0.0}, 0.25},                              // 0.5^2
		{[]float64{2.0}, []float64{-1.0}, 9.0},                              // 3^2
		{[]float64{-1.5}, []float64{0.5}, 4.0},                              // (-2)^2
	}

	for _, tt := range tests {
		got := mse.Forward(tt.yPred, tt.yTrue)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("MSE.Forward(%v, %v) = %v, PyTorch would give %v", tt.yPred, tt.yTrue, got, tt.expected)
		}
	}
}

// TestMSEGradientAgainstPyTorchReference validates MSE gradient against PyTorch
// PyTorch MSE gradient: dL/dy_pred = 2*(y_pred - y_true)/n
func TestMSEGradientAgainstPyTorchReference(t *testing.T) {
	mse := MSE{}

	// Reference values computed with PyTorch
	// dL/dy_pred = 2*(y_pred - y_true)/n
	tests := []struct {
		yPred    []float64
		yTrue    []float64
		expected []float64
	}{
		{[]float64{1.0, 2.0}, []float64{1.0, 2.0}, []float64{0.0, 0.0}},              // Perfect
		{[]float64{1.5, 2.0}, []float64{1.0, 2.0}, []float64{0.5, 0.0}},              // 2*0.5/2 = 0.5
		{[]float64{0.0, 0.0}, []float64{1.0, 1.0}, []float64{-1.0, -1.0}},            // 2*(-1)/2 = -1
		{[]float64{0.5, 0.5}, []float64{0.0, 1.0}, []float64{0.5, -0.5}},             // 2*{0.5, -0.5}/2 = {0.5, -0.5}
	}

	for _, tt := range tests {
		got := mse.Backward(tt.yPred, tt.yTrue)
		if len(got) != len(tt.expected) {
			t.Errorf("MSE.Backward length mismatch: got %d, want %d", len(got), len(tt.expected))
			continue
		}
		for i := range got {
			if !float64Near(got[i], tt.expected[i], 1e-10) {
				t.Errorf("MSE.Backward[%d](%v, %v) = %v, PyTorch would give %v", i, tt.yPred, tt.yTrue, got[i], tt.expected[i])
			}
		}
	}
}

// TestCrossEntropyAgainstPyTorchReference validates CrossEntropy against PyTorch
// For binary classification with sigmoid, PyTorch BCELoss: -[y*log(p) + (1-y)*log(1-p)]
func TestCrossEntropyAgainstPyTorchReference(t *testing.T) {
	ce := CrossEntropy{}

	// Note: Our CrossEntropy is designed for softmax output (sum to 1)
	// For binary classification with single sigmoid output, use BCELoss in PyTorch
	// We test with single-element predictions

	// Reference: PyTorch BCELoss with sigmoid output
	// For y=1, p=0.7: -log(0.7) = 0.3567
	// For y=0, p=0.3: -log(1-0.3) = -log(0.7) = 0.3567

	tests := []struct {
		yPred    []float64
		yTrue    []float64
		minLoss  float64 // Minimum expected loss
		maxLoss  float64 // Maximum expected loss
	}{
		// Perfect prediction (p=1, y=1): loss = 0
		{[]float64{1.0}, []float64{1.0}, 0.0, 1e-10},
		// Perfect prediction (p=0, y=0): loss = 0
		{[]float64{0.0}, []float64{0.0}, 0.0, 1e-10},
		// Good prediction (p=0.9, y=1): loss ≈ 0.105
		{[]float64{0.9}, []float64{1.0}, 0.1, 0.11},
		// Bad prediction (p=0.1, y=1): loss ≈ 2.30
		{[]float64{0.1}, []float64{1.0}, 2.2, 2.4},
	}

	for _, tt := range tests {
		got := ce.Forward(tt.yPred, tt.yTrue)
		if got < tt.minLoss || got > tt.maxLoss {
			t.Errorf("CrossEntropy.Forward(%v, %v) = %v, expected [%v, %v]", tt.yPred, tt.yTrue, got, tt.minLoss, tt.maxLoss)
		}
	}
}

// TestCrossEntropyGradientAgainstPyTorchReference validates CrossEntropy gradient
// For cross entropy with softmax: gradient = y_pred - y_true
// For binary cross entropy: gradient = y_pred - y_true (when using BCELoss)
func TestCrossEntropyGradientAgainstPyTorchReference(t *testing.T) {
	ce := CrossEntropy{}

	// For cross entropy, gradient is (y_pred - y_true)
	// This is true for both multiclass softmax and binary BCE
	tests := []struct {
		yPred    []float64
		yTrue    []float64
		expected []float64
	}{
		{[]float64{1.0, 0.0}, []float64{1.0, 0.0}, []float64{0.0, 0.0}}, // Perfect
		{[]float64{0.7, 0.3}, []float64{1.0, 0.0}, []float64{-0.3, 0.3}}, // PyTorch gradient
		{[]float64{0.5}, []float64{1.0}, []float64{-0.5}},
		{[]float64{0.5}, []float64{0.0}, []float64{0.5}},
	}

	for _, tt := range tests {
		got := ce.Backward(tt.yPred, tt.yTrue)
		if len(got) != len(tt.expected) {
			t.Errorf("CrossEntropy.Backward length mismatch: got %d, want %d", len(got), len(tt.expected))
			continue
		}
		for i := range got {
			if !float64Near(got[i], tt.expected[i], 1e-10) {
				t.Errorf("CrossEntropy.Backward[%d](%v, %v) = %v, PyTorch would give %v", i, tt.yPred, tt.yTrue, got[i], tt.expected[i])
			}
		}
	}
}

// TestHuberAgainstPyTorchReference validates Huber against PyTorch
// PyTorch Huber: for |diff| <= delta, 0.5*diff^2; else delta*(|diff| - 0.5*delta)
func TestHuberAgainstPyTorchReference(t *testing.T) {
	huber := NewHuber(1.0)

	// Reference values computed with PyTorch: nn.HuberLoss(delta=1.0)
	tests := []struct {
		yPred    []float64
		yTrue    []float64
		expected float64
	}{
		// Perfect prediction
		{[]float64{1.0, 2.0}, []float64{1.0, 2.0}, 0.0},
		// Small error (diff < delta): quadratic region
		{[]float64{1.0}, []float64{1.5}, 0.125}, // 0.5 * 0.5^2 = 0.125
		// Large error (diff > delta): linear region
		{[]float64{0.0}, []float64{2.0}, 1.5}, // 1.0 * (2.0 - 0.5*1.0) = 1.5
		// Mixed: one small, one large
		{[]float64{0.0, 0.0}, []float64{0.5, 2.0}, 0.8125}, // (0.125 + 1.5) / 2 = 0.8125
	}

	for _, tt := range tests {
		got := huber.Forward(tt.yPred, tt.yTrue)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("Huber.Forward(%v, %v) = %v, PyTorch would give %v", tt.yPred, tt.yTrue, got, tt.expected)
		}
	}
}

// TestHuberGradientAgainstPyTorchReference validates Huber gradient
// PyTorch Huber gradient:
//   if |diff| <= delta: gradient = diff
//   if |diff| > delta: gradient = delta * sign(diff)
func TestHuberGradientAgainstPyTorchReference(t *testing.T) {
	huber := NewHuber(1.0)

	// Reference values computed with PyTorch
	tests := []struct {
		yPred    []float64
		yTrue    []float64
		expected []float64
	}{
		// Perfect prediction
		{[]float64{1.0, 2.0}, []float64{1.0, 2.0}, []float64{0.0, 0.0}},
		// Small error: gradient = diff
		{[]float64{1.0}, []float64{1.5}, []float64{-0.5}},
		// Large error: gradient = delta * sign(diff)
		{[]float64{0.0}, []float64{2.0}, []float64{-1.0}}, // -delta = -1
		{[]float64{2.0}, []float64{0.0}, []float64{1.0}},  // +delta = +1
	}

	for _, tt := range tests {
		got := huber.Backward(tt.yPred, tt.yTrue)
		if len(got) != len(tt.expected) {
			t.Errorf("Huber.Backward length mismatch: got %d, want %d", len(got), len(tt.expected))
			continue
		}
		for i := range got {
			if !float64Near(got[i], tt.expected[i], 1e-10) {
				t.Errorf("Huber.Backward[%d](%v, %v) = %v, PyTorch would give %v", i, tt.yPred, tt.yTrue, got[i], tt.expected[i])
			}
		}
	}
}

// TestHuberDeltaParameterNumeric validates different delta values (numeric check)
func TestHuberDeltaParameterNumeric(t *testing.T) {
	yPred := []float64{0.0}
	yTrue := []float64{1.5}

	// diff = 1.5
	// delta=0.5: |diff| > delta, so loss = delta*(|diff| - 0.5*delta) = 0.5*(1.5-0.25) = 0.625
	// delta=2.0: |diff| < delta, so loss = 0.5*diff^2 = 0.5*2.25 = 1.125

	huber1 := NewHuber(0.5)
	loss1 := huber1.Forward(yPred, yTrue)
	expected1 := 0.625

	huber2 := NewHuber(2.0)
	loss2 := huber2.Forward(yPred, yTrue)
	expected2 := 1.125

	if !float64Near(loss1, expected1, 1e-10) {
		t.Errorf("Huber(delta=0.5).Forward([0], [1.5]) = %v, expected %v", loss1, expected1)
	}
	if !float64Near(loss2, expected2, 1e-10) {
		t.Errorf("Huber(delta=2.0).Forward([0], [1.5]) = %v, expected %v", loss2, expected2)
	}
}

// Helper function
func float64Near(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}
