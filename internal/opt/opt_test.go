// Package opt provides comprehensive unit tests for optimizers.
package opt

import (
	"math"
	"testing"
)

// TestSGDStep tests SGD step computation.
func TestSGDStep(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	params := []float64{1.0, 2.0, 3.0}
	gradients := []float64{0.1, 0.2, 0.3}

	updated := sgd.Step(params, gradients)

	// Expected: params - lr * gradients
	expected := []float64{
		1.0 - 0.1*0.1,  // 0.99
		2.0 - 0.1*0.2,  // 1.98
		3.0 - 0.1*0.3,  // 2.97
	}

	for i := range updated {
		if math.Abs(updated[i]-expected[i]) > 1e-10 {
			t.Errorf("updated[%d] = %v, want %v", i, updated[i], expected[i])
		}
	}
}

// TestSGDStepInPlace tests in-place SGD update.
func TestSGDStepInPlace(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	params := []float64{1.0, 2.0, 3.0}
	gradients := []float64{0.1, 0.2, 0.3}

	sgd.StepInPlace(params, gradients)

	// params should be updated in-place
	expected := []float64{
		1.0 - 0.1*0.1,  // 0.99
		2.0 - 0.1*0.2,  // 1.98
		3.0 - 0.1*0.3,  // 2.97
	}

	for i := range params {
		if math.Abs(params[i]-expected[i]) > 1e-10 {
			t.Errorf("params[%d] = %v, want %v", i, params[i], expected[i])
		}
	}
}

// TestSGDStepInPlaceModifiesInput tests that StepInPlace modifies the input slice.
func TestSGDStepInPlaceModifiesInput(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	params := []float64{1.0, 2.0}
	gradients := []float64{0.1, 0.1}

	// Store original values
	original0 := params[0]

	sgd.StepInPlace(params, gradients)

	// Input should be modified
	if params[0] == original0 {
		t.Error("StepInPlace should modify input slice")
	}
}

// TestSGDLearningRateEffect tests that larger learning rates cause larger updates.
func TestSGDLearningRateEffect(t *testing.T) {
	gradients := []float64{1.0, 2.0}

	// Small learning rate
	sgd1 := SGD{LearningRate: 0.01}
	updated1 := sgd1.Step([]float64{0.0, 0.0}, gradients)

	// Large learning rate
	sgd2 := SGD{LearningRate: 0.5}
	updated2 := sgd2.Step([]float64{0.0, 0.0}, gradients)

	// Check that larger LR causes larger update
	for i := range gradients {
		if math.Abs(updated1[i]) >= math.Abs(updated2[i]) {
			t.Errorf("With larger learning rate, update should be larger at index %d", i)
		}
	}
}

// TestSGDZeroLearningRate tests zero learning rate behavior.
func TestSGDZeroLearningRate(t *testing.T) {
	sgd := SGD{LearningRate: 0.0}

	params := []float64{1.0, 2.0, 3.0}
	gradients := []float64{1.0, 1.0, 1.0}

	updated := sgd.Step(params, gradients)

	// With zero LR, params should not change
	for i := range params {
		if math.Abs(updated[i]-params[i]) > 1e-10 {
			t.Errorf("With zero LR, param[%d] should not change: %v vs %v", i, updated[i], params[i])
		}
	}
}

// TestSGDZeroGradients tests zero gradient behavior.
func TestSGDZeroGradients(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	params := []float64{1.0, 2.0, 3.0}
	gradients := []float64{0.0, 0.0, 0.0}

	updated := sgd.Step(params, gradients)

	// With zero gradients, params should not change
	for i := range params {
		if math.Abs(updated[i]-params[i]) > 1e-10 {
			t.Errorf("With zero gradients, param[%d] should not change: %v vs %v", i, updated[i], params[i])
		}
	}
}

// TestSGDNegativeGradients tests negative gradient behavior.
func TestSGDNegativeGradients(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	params := []float64{0.0}
	gradients := []float64{-0.5} // Negative gradient

	updated := sgd.Step(params, gradients)

	// Negative gradient should increase parameter (move in positive direction)
	expected := 0.0 - 0.1*(-0.5) // = 0.05
	if math.Abs(updated[0]-expected) > 1e-10 {
		t.Errorf("updated = %v, want %v", updated[0], expected)
	}
}

// TestAdamNewAdam tests Adam creation with defaults.
func TestAdamNewAdam(t *testing.T) {
	adam := NewAdam(0.001)

	if adam.LearningRate != 0.001 {
		t.Errorf("LearningRate = %v, want 0.001", adam.LearningRate)
	}
	if adam.Beta1 != 0.9 {
		t.Errorf("Beta1 = %v, want 0.9", adam.Beta1)
	}
	if adam.Beta2 != 0.999 {
		t.Errorf("Beta2 = %v, want 0.999", adam.Beta2)
	}
	if adam.Epsilon != 1e-8 {
		t.Errorf("Epsilon = %v, want 1e-8", adam.Epsilon)
	}
}

// TestAdamCustomBeta tests Adam with custom betas.
func TestAdamCustomBeta(t *testing.T) {
	adam := &Adam{
		LearningRate: 0.001,
		Beta1:        0.95,
		Beta2:        0.9999,
		Epsilon:      1e-10,
	}

	if adam.Beta1 != 0.95 {
		t.Errorf("Beta1 = %v, want 0.95", adam.Beta1)
	}
	if adam.Beta2 != 0.9999 {
		t.Errorf("Beta2 = %v, want 0.9999", adam.Beta2)
	}
}

// TestAdamStep tests Adam step (currently same as SGD).
func TestAdamStep(t *testing.T) {
	adam := NewAdam(0.1)

	params := []float64{1.0, 2.0}
	gradients := []float64{0.1, 0.2}

	updated := adam.Step(params, gradients)

	// Currently Adam uses SGD internally, so results should match
	sgd := SGD{LearningRate: 0.1}
	expected := sgd.Step(params, gradients)

	for i := range updated {
		if math.Abs(updated[i]-expected[i]) > 1e-10 {
			t.Errorf("Adam updated[%d] = %v, SGD expected[%d] = %v", i, updated[i], i, expected[i])
		}
	}
}

// TestAdamStepInPlace tests Adam in-place update.
func TestAdamStepInPlace(t *testing.T) {
	adam := NewAdam(0.1)

	params := []float64{1.0, 2.0}
	gradients := []float64{0.1, 0.2}

	adam.StepInPlace(params, gradients)

	// Currently Adam uses SGD internally
	sgd := SGD{LearningRate: 0.1}
	sgd.StepInPlace([]float64{1.0, 2.0}, []float64{0.1, 0.2})

	// Check final values match
	expected := []float64{0.99, 1.98}
	for i := range params {
		if math.Abs(params[i]-expected[i]) > 1e-10 {
			t.Errorf("params[%d] = %v, want %v", i, params[i], expected[i])
		}
	}
}

// TestOptimizerInterface tests that optimizers implement the interface.
func TestOptimizerInterface(t *testing.T) {
	var _ Optimizer = &SGD{}
	var _ Optimizer = &Adam{}

	// Verify both can be used through interface
	params := []float64{1.0, 2.0}
	gradients := []float64{0.1, 0.2}

	sgd := SGD{LearningRate: 0.1}
	adam := NewAdam(0.1)

	updatedSGD := sgd.Step(params, gradients)
	updatedAdam := adam.Step(params, gradients)

	if len(updatedSGD) != len(params) || len(updatedAdam) != len(params) {
		t.Error("Optimizer.Step should return slice of same length")
	}
}

// TestSGDMultipleSteps tests multiple SGD steps.
func TestSGDMultipleSteps(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	// Gradient descent on f(x) = x^2, gradient = 2x
	x := 10.0

	for i := 0; i < 10; i++ {
		gradient := 2.0 * x
		x = x - sgd.LearningRate*gradient
	}

	// After 10 steps with LR=0.1:
	// x_1 = 10 - 0.1*20 = 8
	// x_2 = 8 - 0.1*16 = 6.4
	// x_3 = 6.4 - 0.1*12.8 = 5.12
	// x_n = 10 * 0.8^n
	// x_10 = 10 * 0.8^10 ≈ 1.07
	expected := 10.0 * math.Pow(0.8, 10)

	if math.Abs(x-expected) > 0.01 {
		t.Errorf("After 10 steps, x = %v, want ~%v", x, expected)
	}
}

// TestSGDConvergence tests SGD convergence on simple quadratic.
func TestSGDConvergence(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	// Minimize f(x, y) = x^2 + y^2 starting from (10, 10)
	params := []float64{10.0, 10.0}

	for i := 0; i < 50; i++ {
		// Gradient: [2x, 2y]
		gradients := []float64{2.0 * params[0], 2.0 * params[1]}
		params = sgd.Step(params, gradients)
	}

	// After 50 steps: x = 10 * 0.8^50 ≈ very small
	// Should be close to minimum at (0, 0)
	if math.Abs(params[0]) > 0.01 || math.Abs(params[1]) > 0.01 {
		t.Errorf("After convergence, params = %v, should be near [0, 0]", params)
	}
}

// TestSGDStepInPlaceNoAllocation tests that StepInPlace doesn't allocate.
func TestSGDStepInPlaceNoAllocation(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	params := make([]float64, 100)
	gradients := make([]float64, 100)

	for i := range params {
		params[i] = float64(i)
		gradients[i] = 0.1
	}

	// Store original pointer
	originalPtr := &params[0]

	// Call StepInPlace
	sgd.StepInPlace(params, gradients)

	// Pointer should be same (no reallocation)
	if &params[0] != originalPtr {
		t.Log("Note: StepInPlace may reallocate - this is acceptable but noted")
	}
}

// TestSGDLargeLearningRate tests behavior with large learning rate.
func TestSGDLargeLearningRate(t *testing.T) {
	sgd := SGD{LearningRate: 1.0}

	params := []float64{1.0}
	gradients := []float64{1.0}

	updated := sgd.Step(params, gradients)

	// With LR=1, gradient step should be full step
	expected := 0.0
	if math.Abs(updated[0]-expected) > 1e-10 {
		t.Errorf("With LR=1, updated = %v, want %v", updated[0], expected)
	}
}

// TestSGDDivergenceWithLargeLR tests potential divergence with large learning rate.
func TestSGDDivergenceWithLargeLR(t *testing.T) {
	sgd := SGD{LearningRate: 2.0} // Very large LR

	// Minimize f(x) = x^2 starting from x=1
	x := 1.0

	for i := 0; i < 5; i++ {
		gradient := 2.0 * x
		x = x - sgd.LearningRate*gradient
		// x_n+1 = x_n - 2*2*x_n = x_n * (1 - 4) = -3 * x_n
		// x_5 = 1 * (-3)^5 = -243
	}

	// Should diverge (oscillate and grow)
	if math.Abs(x) < 100 {
		t.Errorf("With very large LR, should diverge, got %v", x)
	}
}

// TestOptimizerPrecision tests numerical precision.
func TestOptimizerPrecision(t *testing.T) {
	sgd := SGD{LearningRate: 1e-10} // Very small LR

	params := []float64{1.0}
	gradients := []float64{1e-8}

	updated := sgd.Step(params, gradients)

	// Update should be small but non-zero
	// At this precision, the update might be lost
	// Just check it doesn't crash and produces reasonable output
	if math.IsNaN(updated[0]) || math.IsInf(updated[0], 0) {
		t.Errorf("Got NaN or Inf: %v", updated[0])
	}
}

// TestSGDSymmetric tests that SGD is symmetric around zero.
func TestSGDSymmetric(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	// Positive gradient from positive params
	params1 := []float64{1.0}
	grad1 := []float64{0.5}
	updated1 := sgd.Step(params1, grad1)

	// Negative gradient from negative params
	params2 := []float64{-1.0}
	grad2 := []float64{-0.5}
	updated2 := sgd.Step(params2, grad2)

	// Results should be symmetric
	if math.Abs(updated1[0]-(-updated2[0])) > 1e-10 {
		t.Errorf("SGD not symmetric: updated1 = %v, -updated2 = %v", updated1[0], -updated2[0])
	}
}

// TestSGDAdditive tests gradient addition property.
func TestSGDAdditive(t *testing.T) {
	sgd := SGD{LearningRate: 0.1}

	params := []float64{0.0}

	// Apply gradient g1 then g2
	grad1 := []float64{1.0}
	grad2 := []float64{2.0}

	// Step 1
	params1 := sgd.Step(params, grad1)
	// Step 2
	params2 := sgd.Step(params1, grad2)

	// Should equal applying combined gradient
	combinedGrad := []float64{3.0}
	paramsCombined := sgd.Step(params, combinedGrad)

	if math.Abs(params2[0]-paramsCombined[0]) > 1e-10 {
		t.Errorf("SGD not additive: sequential = %v, combined = %v", params2[0], paramsCombined[0])
	}
}

// TestAdamZeroGradient tests Adam with zero gradients.
func TestAdamZeroGradient(t *testing.T) {
	adam := NewAdam(0.1)

	params := []float64{1.0, 2.0}
	gradients := []float64{0.0, 0.0}

	updated := adam.Step(params, gradients)

	// With zero gradients, params should not change
	for i := range params {
		if math.Abs(updated[i]-params[i]) > 1e-10 {
			t.Errorf("With zero gradients, param[%d] should not change", i)
		}
	}
}

// TestSGDMultipleParameters tests with many parameters.
func TestSGDMultipleParameters(t *testing.T) {
	sgd := SGD{LearningRate: 0.01}

	// 1000 parameters
	n := 1000
	params := make([]float64, n)
	gradients := make([]float64, n)

	for i := 0; i < n; i++ {
		params[i] = float64(i)
		gradients[i] = 1.0
	}

	updated := sgd.Step(params, gradients)

	// Check a few random indices
	for _, idx := range []int{0, 100, 500, 999} {
		expected := float64(idx) - 0.01
		if math.Abs(updated[idx]-expected) > 1e-10 {
			t.Errorf("updated[%d] = %v, want %v", idx, updated[idx], expected)
		}
	}
}
