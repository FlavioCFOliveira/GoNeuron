// Package activations provides validation tests against PyTorch reference implementation.
// This file contains exact reference values computed with PyTorch for comparison.

package activations

import (
	"math"
	"testing"
)

// PyTorch reference values (computed with Python):
// import torch
// torch.set_printoptions(precision=10)

// TestPReLUAgainstPyTorchReference validates PReLU against PyTorch
// PyTorch: torch.nn.PReLU(num_parameters=1)
func TestPReLUAgainstPyTorchReference(t *testing.T) {
	prelu := NewPReLU(0.25)

	// Values from PyTorch: torch.nn.functional.prelu(torch.tensor(x), torch.tensor([0.25]))
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, -0.5},
		{-1.0, -0.25},
		{-0.5, -0.125},
		{0.0, 0.0},
		{0.5, 0.5},
		{1.0, 1.0},
		{2.0, 2.0},
	}

	for _, tt := range tests {
		got := prelu.Activate(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("PReLU.Activate(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestPReLUDerivativeAgainstPyTorchReference validates PReLU derivative
// PyTorch: f'(x) = 1 if x > 0, alpha if x <= 0
func TestPReLUDerivativeAgainstPyTorchReference(t *testing.T) {
	prelu := NewPReLU(0.25)

	// PyTorch PReLU: f'(x) = 1 if x > 0, alpha if x <= 0
	tests := []struct {
		x        float64
		expected float64
	}{
		{-2.0, 0.25},
		{-0.1, 0.25},
		{0.0, 0.25},
		{0.1, 1.0},
		{1.0, 1.0},
	}

	for _, tt := range tests {
		got := prelu.Derivative(tt.x)
		if !float64Near(got, tt.expected, 1e-10) {
			t.Errorf("PReLU.Derivative(%v) = %v, PyTorch would give %v", tt.x, got, tt.expected)
		}
	}
}

// TestPReLUAlphaAccess tests PReLU alpha getter/setter
func TestPReLUAlphaAccess(t *testing.T) {
	prelu := NewPReLU(0.25)

	// Test GetAlpha
	if got := prelu.GetAlpha(); got != 0.25 {
		t.Errorf("PReLU.GetAlpha() = %v, want 0.25", got)
	}

	// Test SetAlpha
	prelu.SetAlpha(0.5)
	if got := prelu.GetAlpha(); got != 0.5 {
		t.Errorf("PReLU.GetAlpha() after SetAlpha(0.5) = %v, want 0.5", got)
	}

	// Verify activation uses updated alpha
	got := prelu.Activate(-1.0)
	expected := -0.5
	if !float64Near(got, expected, 1e-10) {
		t.Errorf("PReLU(alpha=0.5).Activate(-1.0) = %v, want %v", got, expected)
	}
}

// Helper function
func float64Near(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}
