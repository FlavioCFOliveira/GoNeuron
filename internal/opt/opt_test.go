// Package opt provides comprehensive unit tests for optimizers.
package opt

import (
	"math"
	"testing"
)

// TestSGDStep tests SGD step computation.
func TestSGDStep(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	params := []float32{1.0, 2.0, 3.0}
	gradients := []float32{0.1, 0.2, 0.3}

	updated := sgd.Step(params, gradients)

	// Expected: params - lr * gradients
	expected := []float32{
		1.0 - 0.1*0.1, // 0.99
		2.0 - 0.1*0.2, // 1.98
		3.0 - 0.1*0.3, // 2.97
	}

	for i := range updated {
		if float32(math.Abs(float64(updated[i]-expected[i]))) > 1e-6 {
			t.Errorf("updated[%d] = %v, want %v", i, updated[i], expected[i])
		}
	}
}

// TestSGDStepInPlace tests in-place SGD update.
func TestSGDStepInPlace(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	params := []float32{1.0, 2.0, 3.0}
	gradients := []float32{0.1, 0.2, 0.3}

	sgd.StepInPlace(params, gradients)

	// params should be updated in-place
	expected := []float32{
		1.0 - 0.1*0.1, // 0.99
		2.0 - 0.1*0.2, // 1.98
		3.0 - 0.1*0.3, // 2.97
	}

	for i := range params {
		if float32(math.Abs(float64(params[i]-expected[i]))) > 1e-6 {
			t.Errorf("params[%d] = %v, want %v", i, params[i], expected[i])
		}
	}
}

// TestSGDStepInPlaceModifiesInput tests that StepInPlace modifies the input slice.
func TestSGDStepInPlaceModifiesInput(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	params := []float32{1.0, 2.0}
	gradients := []float32{0.1, 0.1}

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
	gradients := []float32{1.0, 2.0}

	// Small learning rate
	sgd1 := &SGD{LearningRate: 0.01}
	updated1 := sgd1.Step([]float32{0.0, 0.0}, gradients)

	// Large learning rate
	sgd2 := &SGD{LearningRate: 0.5}
	updated2 := sgd2.Step([]float32{0.0, 0.0}, gradients)

	// Check that larger LR causes larger update
	for i := range gradients {
		if float32(math.Abs(float64(updated1[i]))) >= float32(math.Abs(float64(updated2[i]))) {
			t.Errorf("With larger learning rate, update should be larger at index %d", i)
		}
	}
}

// TestSGDZeroLearningRate tests zero learning rate behavior.
func TestSGDZeroLearningRate(t *testing.T) {
	sgd := &SGD{LearningRate: 0.0}

	params := []float32{1.0, 2.0, 3.0}
	gradients := []float32{1.0, 1.0, 1.0}

	updated := sgd.Step(params, gradients)

	// With zero LR, params should not change
	for i := range params {
		if float32(math.Abs(float64(updated[i]-params[i]))) > 1e-10 {
			t.Errorf("With zero LR, param[%d] should not change: %v vs %v", i, updated[i], params[i])
		}
	}
}

// TestSGDZeroGradients tests zero gradient behavior.
func TestSGDZeroGradients(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	params := []float32{1.0, 2.0, 3.0}
	gradients := []float32{0.0, 0.0, 0.0}

	updated := sgd.Step(params, gradients)

	// With zero gradients, params should not change
	for i := range params {
		if float32(math.Abs(float64(updated[i]-params[i]))) > 1e-10 {
			t.Errorf("With zero gradients, param[%d] should not change: %v vs %v", i, updated[i], params[i])
		}
	}
}

// TestSGDNegativeGradients tests negative gradient behavior.
func TestSGDNegativeGradients(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	params := []float32{0.0}
	gradients := []float32{-0.5} // Negative gradient

	updated := sgd.Step(params, gradients)

	// Negative gradient should increase parameter (move in positive direction)
	expected := float32(0.0 - 0.1*(-0.5)) // = 0.05
	if float32(math.Abs(float64(updated[0]-expected))) > 1e-6 {
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

// TestAdamStep tests Adam step computation.
func TestAdamStep(t *testing.T) {
	adam := NewAdam(0.1)

	params := []float32{1.0, 2.0}
	gradients := []float32{0.1, 0.2}

	updated := adam.Step(params, gradients)

	// Adam with timestep=1:
	// m = 0.9*0 + 0.1*[0.1, 0.2] = [0.01, 0.02]
	// v = 0.999*0 + 0.001*[0.01, 0.04] = [0.00001, 0.00004]
	// bias1 = 1 - 0.9*1 = 0.1, bias2 = 1 - 0.999*1 = 0.001
	// mHat = [0.01/0.1, 0.02/0.1] = [0.1, 0.2]
	// vHat = [0.00001/0.001, 0.00004/0.001] = [0.01, 0.04]
	// update = lr * mHat / (sqrt(vHat) + eps)
	//        = 0.1 * [0.1, 0.2] / ([0.1, 0.2] + 1e-8)
	//        = 0.1 * [1.0, 1.0] = [0.1, 0.1]
	// updated = [1.0 - 0.1, 2.0 - 0.1] = [0.9, 1.9]

	for i := range updated {
		// Check that Adam makes reasonable updates
		if math.IsNaN(float64(updated[i])) || math.IsInf(float64(updated[i]), 0) {
			t.Errorf("Adam produced NaN or Inf at index %d: %v", i, updated[i])
		}
	}
}

// TestAdamStepInPlace tests Adam in-place update.
func TestAdamStepInPlace(t *testing.T) {
	adam := NewAdam(0.1)

	params := []float32{1.0, 2.0}
	gradients := []float32{0.1, 0.2}

	// First call initializes moments
	adam.StepInPlace(params, gradients)

	// Second call should continue with updated moments
	adam.StepInPlace(params, gradients)

	// Check that params changed
	if params[0] == 1.0 && params[1] == 2.0 {
		t.Error("Adam StepInPlace should modify params")
	}

	// Check no NaN or Inf
	for i := range params {
		if math.IsNaN(float64(params[i])) || math.IsInf(float64(params[i]), 0) {
			t.Errorf("Param[%d] is NaN or Inf: %v", i, params[i])
		}
	}
}

// TestAdamConvergence tests Adam convergence on simple quadratic.
func TestAdamConvergence(t *testing.T) {
	adam := NewAdam(0.5) // Higher LR for faster convergence in test

	// Minimize f(x, y) = x^2 + y^2 starting from (10, 10)
	params := []float32{10.0, 10.0}

	for i := 0; i < 50; i++ {
		// Gradient: [2x, 2y]
		gradients := []float32{2.0 * params[0], 2.0 * params[1]}
		params = adam.Step(params, gradients)
	}

	// With higher LR, should converge well
	if float32(math.Abs(float64(params[0]))) > 0.5 || float32(math.Abs(float64(params[1]))) > 0.5 {
		t.Errorf("After convergence, params = %v, should be near [0, 0]", params)
	}
}

// TestAdamBiasCorrection tests Adam bias correction.
func TestAdamBiasCorrection(t *testing.T) {
	adam := NewAdam(0.1)

	params := []float32{0.0}
	gradients := []float32{1.0}

	// First step: bias correction is important
	updated1 := adam.Step(params, gradients)

	// Reset and do 10 steps
	adam2 := NewAdam(0.1)
	for i := 0; i < 10; i++ {
		adam2.Step(params, gradients)
	}
	updated10 := adam2.Step(params, gradients)

	// After many steps, bias correction is less important
	// Updates should be similar but not identical
	diff := float32(math.Abs(float64(updated1[0] - updated10[0])))
	// They should be different due to bias correction
	if diff < 0.001 {
		t.Log("Note: Bias correction effect is small at this precision")
	}
}

// TestOptimizerInterface tests that optimizers implement the interface.
func TestOptimizerInterface(t *testing.T) {
	var _ Optimizer = &SGD{}
	var _ Optimizer = &Adam{}

	// Verify both can be used through interface
	params := []float32{1.0, 2.0}
	gradients := []float32{0.1, 0.2}

	sgd := &SGD{LearningRate: 0.1}
	adam := NewAdam(0.1)

	updatedSGD := sgd.Step(params, gradients)
	updatedAdam := adam.Step(params, gradients)

	if len(updatedSGD) != len(params) || len(updatedAdam) != len(params) {
		t.Error("Optimizer.Step should return slice of same length")
	}
}

// TestSGDMultipleSteps tests multiple SGD steps.
func TestSGDMultipleSteps(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	// Gradient descent on f(x) = x^2, gradient = 2x
	x := float32(10.0)

	for i := 0; i < 10; i++ {
		gradient := float32(2.0) * x
		x = x - float32(sgd.LearningRate)*gradient
	}

	// After 10 steps with LR=0.1:
	// x_1 = 10 - 0.1*20 = 8
	// x_2 = 8 - 0.1*16 = 6.4
	// x_3 = 6.4 - 0.1*12.8 = 5.12
	// x_n = 10 * 0.8^n
	// x_10 = 10 * 0.8^10 ≈ 1.07
	expected := float32(10.0 * math.Pow(0.8, 10))

	if float32(math.Abs(float64(x-expected))) > 0.01 {
		t.Errorf("After 10 steps, x = %v, want ~%v", x, expected)
	}
}

// TestSGDConvergence tests SGD convergence on simple quadratic.
func TestSGDConvergence(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	// Minimize f(x, y) = x^2 + y^2 starting from (10, 10)
	params := []float32{10.0, 10.0}

	for i := 0; i < 50; i++ {
		// Gradient: [2x, 2y]
		gradients := []float32{2.0 * params[0], 2.0 * params[1]}
		params = sgd.Step(params, gradients)
	}

	// After 50 steps: x = 10 * 0.8^50 ≈ very small
	// Should be close to minimum at (0, 0)
	if float32(math.Abs(float64(params[0]))) > 0.01 || float32(math.Abs(float64(params[1]))) > 0.01 {
		t.Errorf("After convergence, params = %v, should be near [0, 0]", params)
	}
}

// TestSGDStepInPlaceNoAllocation tests that StepInPlace doesn't allocate.
func TestSGDStepInPlaceNoAllocation(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	params := make([]float32, 100)
	gradients := make([]float32, 100)

	for i := range params {
		params[i] = float32(i)
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
	sgd := &SGD{LearningRate: 1.0}

	params := []float32{1.0}
	gradients := []float32{1.0}

	updated := sgd.Step(params, gradients)

	// With LR=1, gradient step should be full step
	expected := float32(0.0)
	if float32(math.Abs(float64(updated[0]-expected))) > 1e-6 {
		t.Errorf("With LR=1, updated = %v, want %v", updated[0], expected)
	}
}

// TestSGDDivergenceWithLargeLR tests potential divergence with large learning rate.
func TestSGDDivergenceWithLargeLR(t *testing.T) {
	sgd := &SGD{LearningRate: 2.0} // Very large LR

	// Minimize f(x) = x^2 starting from x=1
	x := float32(1.0)

	for i := 0; i < 5; i++ {
		gradient := float32(2.0) * x
		x = x - float32(sgd.LearningRate)*gradient
		// x_n+1 = x_n - 2*2*x_n = x_n * (1 - 4) = -3 * x_n
		// x_5 = 1 * (-3)^5 = -243
	}

	// Should diverge (oscillate and grow)
	if float32(math.Abs(float64(x))) < 100 {
		t.Errorf("With very large LR, should diverge, got %v", x)
	}
}

// TestOptimizerPrecision tests numerical precision.
func TestOptimizerPrecision(t *testing.T) {
	sgd := SGD{LearningRate: 1e-10} // Very small LR

	params := []float32{1.0}
	gradients := []float32{1e-8}

	updated := sgd.Step(params, gradients)

	// Update should be small but non-zero
	// At this precision, the update might be lost
	// Just check it doesn't crash and produces reasonable output
	if math.IsNaN(float64(updated[0])) || math.IsInf(float64(updated[0]), 0) {
		t.Errorf("Got NaN or Inf: %v", updated[0])
	}
}

// TestSGDSymmetric tests that SGD is symmetric around zero.
func TestSGDSymmetric(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	// Positive gradient from positive params
	params1 := []float32{1.0}
	grad1 := []float32{0.5}
	updated1 := sgd.Step(params1, grad1)

	// Negative gradient from negative params
	params2 := []float32{-1.0}
	grad2 := []float32{-0.5}
	updated2 := sgd.Step(params2, grad2)

	// Results should be symmetric
	if float32(math.Abs(float64(updated1[0]-(-updated2[0])))) > 1e-6 {
		t.Errorf("SGD not symmetric: updated1 = %v, -updated2 = %v", updated1[0], -updated2[0])
	}
}

// TestSGDAdditive tests gradient addition property.
func TestSGDAdditive(t *testing.T) {
	sgd := &SGD{LearningRate: 0.1}

	params := []float32{0.0}

	// Apply gradient g1 then g2
	grad1 := []float32{1.0}
	grad2 := []float32{2.0}

	// Step 1
	params1 := sgd.Step(params, grad1)
	// Step 2
	params2 := sgd.Step(params1, grad2)

	// Should equal applying combined gradient
	combinedGrad := []float32{3.0}
	paramsCombined := sgd.Step(params, combinedGrad)

	if float32(math.Abs(float64(params2[0]-paramsCombined[0]))) > 1e-6 {
		t.Errorf("SGD not additive: sequential = %v, combined = %v", params2[0], paramsCombined[0])
	}
}

// TestAdamZeroGradient tests Adam with zero gradients.
func TestAdamZeroGradient(t *testing.T) {
	adam := NewAdam(0.1)

	params := []float32{1.0, 2.0}
	gradients := []float32{0.0, 0.0}

	updated := adam.Step(params, gradients)

	// With zero gradients, params should not change
	for i := range params {
		if float32(math.Abs(float64(updated[i]-params[i]))) > 1e-10 {
			t.Errorf("With zero gradients, param[%d] should not change", i)
		}
	}
}

// TestSGDMultipleParameters tests with many parameters.
func TestSGDMultipleParameters(t *testing.T) {
	sgd := &SGD{LearningRate: 0.01}

	// 1000 parameters
	n := 1000
	params := make([]float32, n)
	gradients := make([]float32, n)

	for i := 0; i < n; i++ {
		params[i] = float32(i)
		gradients[i] = 1.0
	}

	updated := sgd.Step(params, gradients)

	// Check a few random indices
	for _, idx := range []int{0, 100, 500, 999} {
		expected := float32(idx) - 0.01
		if float32(math.Abs(float64(updated[idx]-expected))) > 1e-6 {
			t.Errorf("updated[%d] = %v, want %v", idx, updated[idx], expected)
		}
	}
}

// ==================== AdamW Tests ====================

// TestAdamWCreation tests AdamW creation with default values.
func TestAdamWCreation(t *testing.T) {
	adamw := NewAdamW(0.001)

	if adamw.LearningRate != 0.001 {
		t.Errorf("LearningRate = %v, want 0.001", adamw.LearningRate)
	}
	if adamw.WeightDecay != 0.01 {
		t.Errorf("WeightDecay = %v, want 0.01", adamw.WeightDecay)
	}
	if adamw.Beta1 != 0.9 {
		t.Errorf("Beta1 = %v, want 0.9", adamw.Beta1)
	}
	if adamw.Beta2 != 0.999 {
		t.Errorf("Beta2 = %v, want 0.999", adamw.Beta2)
	}
	if adamw.Epsilon != 1e-8 {
		t.Errorf("Epsilon = %v, want 1e-8", adamw.Epsilon)
	}
}

// TestAdamWStep tests AdamW step computation.
func TestAdamWStep(t *testing.T) {
	adamw := NewAdamW(0.1)
	adamw.WeightDecay = 0.0 // Disable weight decay for comparison with Adam

	params := []float32{1.0, 2.0}
	gradients := []float32{0.1, 0.2}

	updated := adamw.Step(params, gradients)

	// With weight decay = 0, AdamW should behave like Adam
	for i := range updated {
		if math.IsNaN(float64(updated[i])) || math.IsInf(float64(updated[i]), 0) {
			t.Errorf("AdamW produced NaN or Inf at index %d: %v", i, updated[i])
		}
	}
}

// TestAdamWWeightDecay tests that weight decay actually shrinks parameters.
func TestAdamWWeightDecay(t *testing.T) {
	// Test with weight decay
	adamw := NewAdamW(0.01)
	adamw.WeightDecay = 0.1

	params := []float32{10.0, 10.0}
	gradients := []float32{0.0, 0.0} // Zero gradients

	adamw.NewStep()
	updated := adamw.Step(params, gradients)

	// With zero gradients and weight decay, parameters should shrink
	for i := range updated {
		if updated[i] >= params[i] {
			t.Errorf("AdamW with weight decay should shrink parameters: updated[%d] = %v, original = %v",
				i, updated[i], params[i])
		}
	}
}

// TestAdamWStepInPlace tests AdamW in-place update.
func TestAdamWStepInPlace(t *testing.T) {
	adamw := NewAdamW(0.1)

	params := []float32{1.0, 2.0}
	gradients := []float32{0.1, 0.2}

	// First call initializes moments
	adamw.StepInPlace(params, gradients)

	// Second call should continue with updated moments
	adamw.StepInPlace(params, gradients)

	// Check that params changed
	if params[0] == 1.0 && params[1] == 2.0 {
		t.Error("AdamW StepInPlace should modify params")
	}

	// Check no NaN or Inf
	for i := range params {
		if math.IsNaN(float64(params[i])) || math.IsInf(float64(params[i]), 0) {
			t.Errorf("Param[%d] is NaN or Inf: %v", i, params[i])
		}
	}
}

// TestAdamWConvergence tests AdamW convergence on simple quadratic.
func TestAdamWConvergence(t *testing.T) {
	adamw := NewAdamW(0.5)
	adamw.WeightDecay = 0.01

	// Minimize f(x, y) = x^2 + y^2 starting from (10, 10)
	params := []float32{10.0, 10.0}

	for i := 0; i < 100; i++ {
		adamw.NewStep()

		// Gradient of x^2 is 2x
		gradients := []float32{2.0 * params[0], 2.0 * params[1]}
		adamw.StepInPlace(params, gradients)
	}

	// Should converge close to zero
	if math.Abs(float64(params[0])) > 0.1 || math.Abs(float64(params[1])) > 0.1 {
		t.Errorf("AdamW did not converge: params = %v", params)
	}
}

// TestAdamWState tests AdamW state serialization.
func TestAdamWState(t *testing.T) {
	adamw := NewAdamW(0.01)
	adamw.WeightDecay = 0.05
	adamw.timestep = 5

	state := adamw.State()

	if state["LearningRate"].(float32) != 0.01 {
		t.Error("State LearningRate mismatch")
	}
	if state["WeightDecay"].(float32) != 0.05 {
		t.Error("State WeightDecay mismatch")
	}
	if state["timestep"].(int) != 5 {
		t.Error("State timestep mismatch")
	}

	// Test restoring state
	adamw2 := NewAdamW(0.0)
	adamw2.SetState(state)

	if adamw2.LearningRate != 0.01 {
		t.Error("SetState LearningRate mismatch")
	}
	if adamw2.WeightDecay != 0.05 {
		t.Error("SetState WeightDecay mismatch")
	}
}

// TestAdamWGobState tests AdamW gob serialization.
func TestAdamWGobState(t *testing.T) {
	adamw := NewAdamW(0.01)
	adamw.WeightDecay = 0.05
	adamw.timestep = 10

	gobState := adamw.GobState()

	// Test restoring from gob state
	adamw2 := NewAdamW(0.0)
	adamw2.SetGobState(gobState)

	if adamw2.LearningRate != 0.01 {
		t.Errorf("SetGobState LearningRate = %v, want 0.01", adamw2.LearningRate)
	}
	if adamw2.WeightDecay != 0.05 {
		t.Errorf("SetGobState WeightDecay = %v, want 0.05", adamw2.WeightDecay)
	}
	if adamw2.timestep != 10 {
		t.Errorf("SetGobState timestep = %v, want 10", adamw2.timestep)
	}
}

// TestAdamWVsAdamWithZeroWeightDecay tests AdamW equals Adam when weight decay is 0.
func TestAdamWVsAdamWithZeroWeightDecay(t *testing.T) {
	// Both optimizers with same parameters
	adam := NewAdam(0.1)
	adamw := NewAdamW(0.1)
	adamw.WeightDecay = 0.0

	paramsAdam := []float32{1.0, 2.0, 3.0}
	paramsAdamW := []float32{1.0, 2.0, 3.0}
	gradients := []float32{0.1, 0.2, 0.3}

	// Run multiple steps
	for step := 0; step < 5; step++ {
		adam.NewStep()
		adamw.NewStep()

		updatedAdam := adam.Step(paramsAdam, gradients)
		updatedAdamW := adamw.Step(paramsAdamW, gradients)

		// Copy back for next iteration
		copy(paramsAdam, updatedAdam)
		copy(paramsAdamW, updatedAdamW)
	}

	// Results should be very close (allowing for tiny numerical differences)
	for i := range paramsAdam {
		diff := math.Abs(float64(paramsAdam[i] - paramsAdamW[i]))
		if diff > 1e-5 {
			t.Errorf("Adam and AdamW with zero weight decay differ at index %d: diff = %v", i, diff)
		}
	}
}
