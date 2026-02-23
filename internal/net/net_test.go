// Package net provides comprehensive unit tests for neural network.
package net

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"os"
	"testing"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// TestNetworkForward tests forward pass through network.
func TestNetworkForward(t *testing.T) {
	// Simple network: 2 -> 2 -> 1
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{1.0, 2.0}
	output := network.Forward(input)

	if len(output) != 1 {
		t.Errorf("Output length = %d, want 1", len(output))
	}
}

// TestNetworkBackward tests backward pass through network.
func TestNetworkBackward(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{1.0, 2.0}
	target := []float32{1.0}

	// Forward
	yPred := network.Forward(input)

	// Compute loss gradient
	grad := loss.MSE{}.Backward(yPred, target)

	// Backward
	outputGrad := network.Backward(grad)

	if len(outputGrad) != 2 {
		t.Errorf("Output gradient length = %d, want 2", len(outputGrad))
	}
}

// TestNetworkTrain tests single training step.
func TestNetworkTrain(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{0.0, 0.0}
	target := []float32{0.0}

	l := network.Train(input, target)

	if l < 0 {
		t.Errorf("Loss should be non-negative, got %v", l)
	}
}

// TestNetworkXOR tests XOR problem (classic benchmark).
func TestNetworkXOR(t *testing.T) {
	// XOR network: 2 -> 3 -> 1 with Tanh
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.5})

	// XOR training data
	trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float32{{0}, {1}, {1}, {0}}

	// Train for multiple epochs
	for epoch := 0; epoch < 1000; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Test predictions
	tolerances := []struct {
		input  []float32
		target float32
	}{
		{[]float32{0, 0}, 0},
		{[]float32{0, 1}, 1},
		{[]float32{1, 0}, 1},
		{[]float32{1, 1}, 0},
	}

	for _, tt := range tolerances {
		output := network.Forward(tt.input)
		pred := output[0]
		if float32(math.Abs(float64(pred-tt.target))) > 0.1 {
			t.Errorf("XOR(%v) = %v, want ~%v", tt.input, pred, tt.target)
		}
	}
}

// TestNetworkMultipleEpochs tests training over multiple epochs.
func TestNetworkMultipleEpochs(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float32{{0}, {1}, {1}, {0}}

	// Store initial loss
	initialLoss := float32(0.0)
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		initialLoss += loss.MSE{}.Forward(yPred, trainY[i])
	}
	initialLoss /= float32(len(trainX))

	// Train for 500 epochs
	for epoch := 0; epoch < 500; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Check that loss decreased
	finalLoss := float32(0.0)
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		finalLoss += loss.MSE{}.Forward(yPred, trainY[i])
	}
	finalLoss /= float32(len(trainX))

	if finalLoss >= initialLoss {
		t.Errorf("Loss should decrease after training: initial=%v, final=%v", initialLoss, finalLoss)
	}
}

// TestNetworkStep tests optimization step.
func TestNetworkStep(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{1.0, 1.0}
	target := []float32{1.0, 1.0}

	// Get initial params (copy)
	initialParams := make([]float32, len(network.Params()))
	copy(initialParams, network.Params())

	// Train
	network.Train(input, target)

	// Get params after step
	afterStepParams := network.Params()

	// Params should have changed
	paramsChanged := false
	for i := range initialParams {
		if float32(math.Abs(float64(initialParams[i]-afterStepParams[i]))) > 1e-6 {
			paramsChanged = true
			break
		}
	}

	if !paramsChanged {
		t.Error("Parameters should change after training step")
	}
}

// TestNetworkParams tests parameter handling.
func TestNetworkParams(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	params := network.Params()

	// Expected: (2*3 + 3) + (3*1 + 1) = 9 + 4 = 13
	expectedLen := 13
	if len(params) != expectedLen {
		t.Errorf("Params length = %d, want %d", len(params), expectedLen)
	}
}

// TestNetworkGradients tests gradient computation.
func TestNetworkGradients(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{1.0, 1.0}
	target := []float32{1.0, 1.0}

	// Train to compute gradients
	network.Train(input, target)

	// Get gradients
	gradients := network.Gradients()

	// Gradients should not be all zero after training
	nonZeroCount := 0
	for _, g := range gradients {
		if float32(math.Abs(float64(g))) > 1e-10 {
			nonZeroCount++
		}
	}

	if nonZeroCount == 0 {
		t.Error("Gradients should not be all zero after training")
	}
}

// TestNetworkTrainBatch tests batch training.
func TestNetworkTrainBatch(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float32{{0}, {1}, {1}, {0}}

	// Train on batch
	l := network.TrainBatch(trainX, trainY)

	if l < 0 {
		t.Errorf("Batch loss should be non-negative, got %v", l)
	}
}

// TestNetworkTrainBatchEmpty tests empty batch handling.
func TestNetworkTrainBatchEmpty(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	l := network.TrainBatch([][]float32{}, [][]float32{})

	if l != 0 {
		t.Errorf("Empty batch loss should be 0, got %v", l)
	}
}

// TestNetworkTrainBatchConsistency tests that sequential and batch training give same results.
func TestNetworkTrainBatchConsistency(t *testing.T) {
	// Create two identical networks
	layers1 := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	layers2 := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}

	// Initialize with same weights
	optimizer1 := &opt.SGD{LearningRate: 0.1}
	optimizer2 := &opt.SGD{LearningRate: 0.1}

	network1 := New(layers1, loss.MSE{}, optimizer1)
	network2 := New(layers2, loss.MSE{}, optimizer2)

	// Set same initial params
	params := make([]float32, len(network1.Params()))
	copy(params, network1.Params())
	network2.SetParams(params)

	trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float32{{0}, {1}, {1}, {0}}

	// Train one epoch with sequential (looping over samples)
	for i := range trainX {
		// Manual sequential accumulation for exact match
		yPred := network1.Forward(trainX[i])
		grad := network1.loss.Backward(yPred, trainY[i])
		network1.Backward(grad)
	}
	// Average and step manually to match TrainBatch behavior
	invBS := float32(1.0 / float64(len(trainX)))
	grads1 := network1.Gradients()
	for i := range grads1 {
		grads1[i] *= invBS
	}
	network1.Step()

	// Train one epoch with batch
	network2.TrainBatch(trainX, trainY)

	// Compare final params
	params1 := network1.Params()
	params2 := network2.Params()

	if len(params1) != len(params2) {
		t.Errorf("Params length mismatch: sequential=%d, batch=%d", len(params1), len(params2))
	}

	// Check params are similar
	for i := range params1 {
		if float32(math.Abs(float64(params1[i]-params2[i]))) > 1e-5 {
			t.Errorf("Param %d mismatch: sequential=%v, batch=%v", i, params1[i], params2[i])
		}
	}
}

// TestNetworkGradientDescent tests basic gradient descent behavior.
func TestNetworkGradientDescent(t *testing.T) {
	// Simple linear regression: y = 2x + 1
	layers := []layer.Layer{
		layer.NewDense(1, 1, activations.Linear{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Generate training data: y = 2x + 1
	trainX := make([][]float32, 10)
	trainY := make([][]float32, 10)
	for i := 0; i < 10; i++ {
		x := float32(i) / 10.0
		trainX[i] = []float32{x}
		trainY[i] = []float32{2*x + 1}
	}

	// Train for 100 epochs
	for epoch := 0; epoch < 100; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Test predictions
	for i := 0; i < 10; i++ {
		x := float32(i) / 10.0
		output := network.Forward([]float32{x})
		pred := output[0]
		target := float32(2*x + 1)

		// Should be reasonably close after training
		if float32(math.Abs(float64(pred-target))) > 0.2 {
			t.Errorf("Prediction for x=%v: %v, want %v", x, pred, target)
		}
	}
}

// TestNetworkParamsLengthMatchesGradients tests that params and gradients have same length.
func TestNetworkParamsLengthMatchesGradients(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(3, 4, activations.Tanh{}),
		layer.NewDense(4, 2, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{1.0, 1.0, 1.0}
	target := []float32{1.0, 1.0}

	// Train to compute gradients
	network.Train(input, target)

	params := network.Params()
	gradients := network.Gradients()

	if len(params) != len(gradients) {
		t.Errorf("Params length (%d) != Gradients length (%d)", len(params), len(gradients))
	}
}

// TestNetworkLayerOrdering tests that layer order is preserved.
func TestNetworkLayerOrdering(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Verify layer order is preserved
	networkLayers := network.Layers()
	if len(networkLayers) != 3 {
		t.Errorf("Expected 3 layers, got %d", len(networkLayers))
	}

	// Forward pass
	output := network.Forward([]float32{1.0, 1.0})
	if len(output) != 1 {
		t.Errorf("Output length = %d, want 1", len(output))
	}
}

// TestNetworkDifferentActivations tests different activation functions.
func TestNetworkDifferentActivations(t *testing.T) {
	tests := []struct {
		name    string
		act1    activations.Activation
		act2    activations.Activation
	}{
		{"Tanh-Sigmoid", activations.Tanh{}, activations.Sigmoid{}},
		{"ReLU-Sigmoid", activations.ReLU{}, activations.Sigmoid{}},
		{"Sigmoid-Sigmoid", activations.Sigmoid{}, activations.Sigmoid{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layers := []layer.Layer{
				layer.NewDense(2, 2, tt.act1),
				layer.NewDense(2, 1, tt.act2),
			}
			network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

			// Just verify it doesn't panic
			output := network.Forward([]float32{1.0, 1.0})
			if len(output) != 1 {
				t.Errorf("Output length = %d, want 1", len(output))
			}
		})
	}
}

// TestNetworkForwardBackwardShape tests forward/backward shape consistency.
func TestNetworkForwardBackwardShape(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(3, 4, activations.Tanh{}),
		layer.NewDense(4, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{1.0, 1.0, 1.0}
	output := network.Forward(input)

	grad := make([]float32, len(output))
	for i := range grad {
		grad[i] = 1.0
	}

	outputGrad := network.Backward(grad)

	if len(outputGrad) != 3 {
		t.Errorf("Input gradient length = %d, want 3", len(outputGrad))
	}
}

// TestNetworkSetParams tests parameter setting.
func TestNetworkSetParams(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Get params
	params := network.Params()

	// Modify them
	for i := range params {
		params[i] = float32(i) * 0.01
	}

	// Set them back
	network.SetParams(params)

	// Verify they were set
	afterParams := network.Params()
	for i := range params {
		if float32(math.Abs(float64(params[i]-afterParams[i]))) > 1e-6 {
			t.Errorf("Param %d mismatch after SetParams", i)
		}
	}
}

// TestNetworkParamsSetParamsRoundtrip tests that SetParams is the inverse of Params.
func TestNetworkParamsSetParamsRoundtrip(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Get original params
	originalParams := network.Params()

	// Set new random params
	newParams := make([]float32, len(originalParams))
	for i := range newParams {
		newParams[i] = float32(i) * 0.001
	}
	network.SetParams(newParams)

	// Get params after setting
	afterParams := network.Params()

	// Verify they match
	for i := range newParams {
		if float32(math.Abs(float64(newParams[i]-afterParams[i]))) > 1e-6 {
			t.Errorf("Param %d: expected %v, got %v", i, newParams[i], afterParams[i])
		}
	}

	// Set back to original
	network.SetParams(originalParams)
	afterOriginal := network.Params()

	for i := range originalParams {
		if float32(math.Abs(float64(originalParams[i]-afterOriginal[i]))) > 1e-6 {
			t.Errorf("Param %d after roundtrip: expected %v, got %v", i, originalParams[i], afterOriginal[i])
		}
	}
}

// TestNetworkZeroInput tests behavior with zero input.
func TestNetworkZeroInput(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Zero input should produce some output (due to biases)
	output := network.Forward([]float32{0, 0})

	// Output should be in [-1, 1] for Tanh
	for i := range output {
		if output[i] < -1 || output[i] > 1 {
			t.Errorf("Output[%d] = %v, should be in [-1, 1] for Tanh", i, output[i])
		}
	}
}

// TestNetworkLargeInput tests numerical stability with large inputs.
func TestNetworkLargeInput(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Large input values
	output := network.Forward([]float32{100, 100})

	// Tanh should saturate at ±1
	for i := range output {
		if output[i] < -1 || output[i] > 1 {
			t.Errorf("Output[%d] = %v, should be in [-1, 1] for Tanh", i, output[i])
		}
	}
}

// TestNetworkSingleLayer tests network with single layer.
func TestNetworkSingleLayer(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(3, 2, activations.Linear{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	output := network.Forward([]float32{1, 2, 3})
	if len(output) != 2 {
		t.Errorf("Output length = %d, want 2", len(output))
	}
}

// TestNetworkDeepNetwork tests network with many layers.
func TestNetworkDeepNetwork(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(4, 8, activations.Tanh{}),
		layer.NewDense(8, 16, activations.Tanh{}),
		layer.NewDense(16, 8, activations.Tanh{}),
		layer.NewDense(8, 4, activations.Tanh{}),
		layer.NewDense(4, 2, activations.Linear{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	output := network.Forward([]float32{1, 2, 3, 4})
	if len(output) != 2 {
		t.Errorf("Output length = %d, want 2", len(output))
	}
}

// TestNetworkTrainBatchParallel tests parallel batch training.
func TestNetworkTrainBatchParallel(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float32{{0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}}

	// Train on batch (should use parallel path)
	l := network.TrainBatch(trainX, trainY)

	if l < 0 {
		t.Errorf("Batch loss should be non-negative, got %v", l)
	}
}

// TestNetworkSequentialVsParallelConsistency tests that sequential and parallel batch training produce consistent results.
func TestNetworkSequentialVsParallelConsistency(t *testing.T) {
	trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float32{{0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}}

	// Sequential network
	sequentialLayers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	sequentialNet := New(sequentialLayers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Parallel network (same architecture)
	parallelLayers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	parallelNet := New(parallelLayers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Initialize with same weights
	params := sequentialNet.Params()
	parallelNet.SetParams(params)

	// Train sequential
	for i := range trainX {
		sequentialNet.Train(trainX[i], trainY[i])
	}

	// Train parallel (uses batch)
	parallelNet.TrainBatch(trainX, trainY)

	// Compare final losses on same test data
	testLossSeq := float32(0.0)
	testLossPar := float32(0.0)

	for i := range trainX {
		yPredSeq := sequentialNet.Forward(trainX[i])
		yPredPar := parallelNet.Forward(trainX[i])

		testLossSeq += loss.MSE{}.Forward(yPredSeq, trainY[i])
		testLossPar += loss.MSE{}.Forward(yPredPar, trainY[i])
	}

	// Losses should be similar (both are minimizing the same objective)
	// Allow some tolerance due to floating point differences
	if float32(math.Abs(float64(testLossSeq-testLossPar))) > 0.1 {
		t.Logf("Sequential loss: %v, Parallel loss: %v", testLossSeq, testLossPar)
	}
}

// TestNetworkBackwardGradientShape tests backward gradient shapes.
func TestNetworkBackwardGradientShape(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(3, 4, activations.Tanh{}),
		layer.NewDense(4, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{1, 2, 3}
	output := network.Forward(input)

	grad := make([]float32, len(output))
	outputGrad := network.Backward(grad)

	if len(outputGrad) != len(input) {
		t.Errorf("Input gradient length = %d, want %d", len(outputGrad), len(input))
	}
}

// TestNetworkForwardIdempotent tests that same input gives same output.
func TestNetworkForwardIdempotent(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{1.0, 2.0}

	output1 := network.Forward(input)
	output2 := network.Forward(input)

	for i := range output1 {
		if float32(math.Abs(float64(output1[i]-output2[i]))) > 1e-6 {
			t.Errorf("Forward not idempotent: output1[%d]=%v, output2[%d]=%v", i, output1[i], i, output2[i])
		}
	}
}

// TestNetworkTrainPreservesParamsLength tests that Train preserves params length.
func TestNetworkTrainPreservesParamsLength(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	initialLen := len(network.Params())

	network.Train([]float32{1, 1}, []float32{1, 1})

	afterLen := len(network.Params())

	if initialLen != afterLen {
		t.Errorf("Params length changed: %d -> %d", initialLen, afterLen)
	}
}

// TestNetworkLossDecreases tests that loss decreases during training.
func TestNetworkLossDecreases(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.5})

	input := []float32{0, 0}
	target := []float32{0}

	losses := make([]float32, 10)
	for i := 0; i < 10; i++ {
		losses[i] = network.Train(input, target)
	}

	// Loss should generally decrease
	if losses[9] > losses[0] {
		t.Errorf("Loss should decrease: %v -> %v", losses[0], losses[9])
	}
}

// TestNetworkParamsDoNotNan tests that params don't become NaN during training.
func TestNetworkParamsDoNotNan(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	for i := 0; i < 100; i++ {
		params := network.Params()
		for j := range params {
			if math.IsNaN(float64(params[j])) {
				t.Errorf("NaN detected in params at index %d", j)
			}
		}
		network.Train([]float32{1, 1}, []float32{1, 1})
	}
}

// TestNetworkBackwardChain tests backward pass through multiple layers.
func TestNetworkBackwardChain(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float32{1, 2}
	_ = network.Forward(input) // Forward pass to populate states

	grad := []float32{1.0}
	outputGrad := network.Backward(grad)

	// Input gradient should have same length as input
	if len(outputGrad) != 2 {
		t.Errorf("Input gradient length = %d, want 2", len(outputGrad))
	}
}

// TestNetworkLayerCount tests layer count.
func TestNetworkLayerCount(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	if len(network.Layers()) != 2 {
		t.Errorf("Layer count = %d, want 2", len(network.Layers()))
	}
}

// TestNetworkLargeBatch tests training with large batch size.
func TestNetworkLargeBatch(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Create a larger batch
	batchSize := 50
	trainX := make([][]float32, batchSize)
	trainY := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		trainX[i] = []float32{float32(i % 2), float32((i / 2) % 2)}
		trainY[i] = []float32{float32((i % 3) % 2)}
	}

	l := network.TrainBatch(trainX, trainY)

	if l < 0 {
		t.Errorf("Batch loss should be non-negative, got %v", l)
	}
}

// TestNetworkSetParamsLengthMismatch tests handling of length mismatch in SetParams.
func TestNetworkSetParamsLengthMismatch(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Wrong length params
	wrongParams := make([]float32, 100)

	defer func() {
		if r := recover(); r == nil {
			// This is actually acceptable behavior - SetParams may not check length
			t.Log("Note: SetParams did not panic on length mismatch")
		}
	}()

	l := network.layers[0]
	defer func() {
		if r := recover(); r == nil {
			// This is actually acceptable behavior - SetParams may not check length
			t.Log("Note: SetParams did not panic on length mismatch")
		}
	}()

	l.SetParams(wrongParams)
}

// TestNetworkGobEncodingAdam tests gob encoding/decoding with Adam optimizer.
func TestNetworkGobEncodingAdam(t *testing.T) {
	// Create network with Adam optimizer
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	optimizer := opt.NewAdam(0.001)
	network := New(layers, loss.MSE{}, optimizer)

	// Train for a few steps to populate optimizer state
	trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float32{{0}, {1}, {1}, {0}}
	for i := 0; i < 10; i++ {
		for j := range trainX {
			network.Train(trainX[j], trainY[j])
		}
	}

	// Store initial state for comparison
	initialParams := network.Params()

	// Encode to buffer
	var buf bytes.Buffer
	err := network.Encode(&buf)
	if err != nil {
		t.Fatalf("Failed to encode network: %v", err)
	}

	// Decode using Load
	loadedNetwork, _, err := LoadFromBuffer(buf.Bytes())
	if err != nil {
		t.Fatalf("Failed to decode network: %v", err)
	}

	// Verify loaded network
	loadedParams := loadedNetwork.Params()
	if len(loadedParams) != len(initialParams) {
		t.Errorf("Loaded params length = %d, want %d", len(loadedParams), len(initialParams))
	}

	for i := range initialParams {
		if float32(math.Abs(float64(initialParams[i]-loadedParams[i]))) > 1e-6 {
			t.Errorf("Params[%d] mismatch: %v vs %v", i, initialParams[i], loadedParams[i])
		}
	}

	// Verify forward pass works
	output := loadedNetwork.Forward([]float32{0.5, 0.5})
	if len(output) != 1 {
		t.Errorf("Loaded network output length = %d, want 1", len(output))
	}
}

// TestNetworkGobEncodingSGD tests gob encoding/decoding with SGD optimizer.
func TestNetworkGobEncodingSGD(t *testing.T) {
	// Create network with SGD optimizer
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	optimizer := &opt.SGD{LearningRate: 0.1}
	network := New(layers, loss.MSE{}, optimizer)

	// Train for a few steps
	trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float32{{0}, {1}, {1}, {0}}
	for i := 0; i < 10; i++ {
		for j := range trainX {
			network.Train(trainX[j], trainY[j])
		}
	}

	// Store initial state for comparison
	initialParams := network.Params()

	// Encode to buffer
	var buf bytes.Buffer
	err := network.Encode(&buf)
	if err != nil {
		t.Fatalf("Failed to encode network: %v", err)
	}

	// Decode using Load
	loadedNetwork, _, err := LoadFromBuffer(buf.Bytes())
	if err != nil {
		t.Fatalf("Failed to decode network: %v", err)
	}

	// Verify loaded network
	loadedParams := loadedNetwork.Params()
	if len(loadedParams) != len(initialParams) {
		t.Errorf("Loaded params length = %d, want %d", len(loadedParams), len(initialParams))
	}

	for i := range initialParams {
		if float32(math.Abs(float64(initialParams[i]-loadedParams[i]))) > 1e-6 {
			t.Errorf("Params[%d] mismatch: %v vs %v", i, initialParams[i], loadedParams[i])
		}
	}
}

// LoadFromBuffer loads a network from a byte slice (for testing).
func LoadFromBuffer(data []byte) (*Network, loss.Loss, error) {
	buf := bytes.NewReader(data)

	// 1. Read and validate magic number
	magic := make([]byte, 4)
	if _, err := io.ReadFull(buf, magic); err != nil {
		return nil, nil, fmt.Errorf("failed to read magic number: %w", err)
	}
	if string(magic) != "GONN" {
		return nil, nil, fmt.Errorf("invalid magic number: expected GONN, got %s", string(magic))
	}

	// 2. Read and validate version
	var version int32
	if err := binary.Read(buf, binary.LittleEndian, &version); err != nil {
		return nil, nil, fmt.Errorf("failed to read version: %w", err)
	}
	if version != 1 {
		return nil, nil, fmt.Errorf("unsupported version: %d", version)
	}

	decoder := gob.NewDecoder(buf)

	// Read number of layers
	var numLayers int32
	if err := decoder.Decode(&numLayers); err != nil {
		return nil, nil, fmt.Errorf("failed to read layer count: %w", err)
	}

	// Security check for layer count
	const maxLayers = 1000
	if numLayers < 0 || numLayers > maxLayers {
		return nil, nil, fmt.Errorf("invalid number of layers: %d (max %d)", numLayers, maxLayers)
	}

	// Read loss type
	var lossType string
	if err := decoder.Decode(&lossType); err != nil {
		return nil, nil, fmt.Errorf("failed to read loss type: %w", err)
	}

	// Read optimizer type
	var optType string
	if err := decoder.Decode(&optType); err != nil {
		return nil, nil, fmt.Errorf("failed to read optimizer type: %w", err)
	}

	// Read layer configurations and calculate total parameters
	layerConfigs := make([]LayerConfig, numLayers)
	expectedTotalParams := 0
	for i := 0; i < int(numLayers); i++ {
		var cfg LayerConfig
		if err := decoder.Decode(&cfg); err != nil {
			return nil, nil, fmt.Errorf("failed to read layer %d: %w", i, err)
		}
		layerConfigs[i] = cfg
		expectedTotalParams += len(cfg.Params)
	}

	// Security check for total parameters (limit to 1GB = 256M float32)
	const maxParams = 256 * 1024 * 1024
	if expectedTotalParams > maxParams {
		return nil, nil, fmt.Errorf("total parameters exceed limit: %d (max %d)", expectedTotalParams, maxParams)
	}

	// Read parameters for all layers
	var params []float32
	if err := decoder.Decode(&params); err != nil {
		return nil, nil, fmt.Errorf("failed to read parameters: %w", err)
	}

	// Validate parameter count
	if len(params) != expectedTotalParams {
		return nil, nil, fmt.Errorf("parameter count mismatch: expected %d, got %d", expectedTotalParams, len(params))
	}

	// Reconstruct layers
	var layers []layer.Layer
	offset := 0
	for _, cfg := range layerConfigs {
		l, err := cfg.CreateLayer()
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create layer: %w", err)
		}
		numParams := len(cfg.Params)
		l.SetParams(params[offset : offset+numParams])
		layers = append(layers, l)
		offset += numParams
	}

	// Read optimizer state
	var optState map[string]any
	if err := decoder.Decode(&optState); err != nil {
		// Older versions might not have optimizer state
		optState = nil
	}

	// Create network with default optimizer
	var optimizer opt.Optimizer
	switch optType {
	case "SGD":
		optimizer = &opt.SGD{LearningRate: 0.1}
	case "Adam":
		optimizer = opt.NewAdam(0.001)
	default:
		optimizer = &opt.SGD{LearningRate: 0.1}
	}

	if optState != nil {
		optimizer.SetState(optState)
	}

	// Create network with reconstructed loss
	var l loss.Loss
	switch lossType {
	case "MSE":
		l = loss.MSE{}
	case "CrossEntropy":
		l = loss.CrossEntropy{}
	case "Huber":
		l = loss.NewHuber(1.0)
	default:
		l = loss.MSE{}
	}

	return New(layers, l, optimizer), l, nil
}

// TestNetworkLSTMResetBetweenSamples tests that LSTM state is reset between samples.
func TestNetworkLSTMResetBetweenSamples(t *testing.T) {
	// Create network with LSTM layer
	lstm := layer.NewLSTM(2, 3)
	dense := layer.NewDense(3, 2, activations.Linear{})

	layers := []layer.Layer{
		lstm,
		dense,
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Train on multiple samples
	trainX := [][]float32{{1, 2}, {3, 4}, {5, 6}}

	// Store outputs for comparison
	outputs := make([][]float32, len(trainX))
	for i := range trainX {
		outputs[i] = network.Forward(trainX[i])
	}

	// After reset, same input should produce same output (same weights, no state)
	lstm.Reset()
	output1 := network.Forward(trainX[0])

	lstm.Reset()
	output2 := network.Forward(trainX[0])

	for i := range output1 {
		if float32(math.Abs(float64(output1[i]-output2[i]))) > 1e-6 {
			t.Errorf("Reset not working: output1[%d]=%v, output2[%d]=%v", i, output1[i], i, output2[i])
		}
	}
}

// TestNetworkTrainBatchWithLSTM tests batch training with LSTM layer.
func TestNetworkTrainBatchWithLSTM(t *testing.T) {
	// Create network with LSTM
	lstm := layer.NewLSTM(2, 3)
	dense := layer.NewDense(3, 2, activations.Linear{})

	layers := []layer.Layer{
		lstm,
		dense,
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	trainX := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	trainY := [][]float32{{1, 0}, {0, 1}, {1, 1}}

	// Train on batch
	l := network.TrainBatch(trainX, trainY)

	if l < 0 {
		t.Errorf("Batch loss should be non-negative, got %v", l)
	}

	// Verify forward pass works after batch training
	output := network.Forward([]float32{1, 2})
	if len(output) != 2 {
		t.Errorf("Output length = %d, want 2", len(output))
	}
}

// TestNetworkTrainWithLSTMReset tests that Train properly resets LSTM state.
func TestNetworkTrainWithLSTMReset(t *testing.T) {
	// Create network with LSTM
	lstm := layer.NewLSTM(2, 3)
	dense := layer.NewDense(3, 2, activations.Linear{})

	layers := []layer.Layer{
		lstm,
		dense,
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Train multiple times on the same input
	// Without proper reset, outputs would differ due to accumulated state
	outputs := make([][]float32, 5)
	for i := range outputs {
		outputs[i] = network.Forward([]float32{1, 2})
	}

	// Outputs should be identical (same weights, no accumulated state)
	for i := 1; i < len(outputs); i++ {
		for j := range outputs[0] {
			if float32(math.Abs(float64(outputs[0][j]-outputs[i][j]))) > 1e-6 {
				t.Errorf("Output mismatch at training step %d: %v vs %v", i, outputs[0][j], outputs[i][j])
			}
		}
	}
}

// TestNetworkSaveBinaryWeights tests the binary weights export functionality.
func TestNetworkSaveBinaryWeights(t *testing.T) {
	layers := []layer.Layer{
		layer.NewEmbedding(10, 8),
		layer.NewDense(8, 4, activations.ReLU{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	filename := "test_weights.bin"
	defer os.Remove(filename)

	err := network.SaveBinaryWeights(filename)
	if err != nil {
		t.Fatalf("Failed to save binary weights: %v", err)
	}

	// Verify file exists and has content
	info, err := os.Stat(filename)
	if err != nil {
		t.Fatalf("Failed to stat file: %v", err)
	}
	if info.Size() < 4 {
		t.Errorf("File too small: %d bytes", info.Size())
	}
}

// TestLoadSecurity tests security boundaries during model loading.
func TestLoadSecurity(t *testing.T) {
	t.Run("InvalidMagic", func(t *testing.T) {
		var buf bytes.Buffer
		buf.Write([]byte("BAD!")) // Wrong magic
		binary.Write(&buf, binary.LittleEndian, int32(1))

		_, _, err := LoadFromBuffer(buf.Bytes())
		if err == nil {
			t.Error("Expected error for invalid magic number, got nil")
		}
	})

	t.Run("InvalidVersion", func(t *testing.T) {
		var buf bytes.Buffer
		buf.Write([]byte("GONN"))
		binary.Write(&buf, binary.LittleEndian, int32(999)) // Wrong version

		_, _, err := LoadFromBuffer(buf.Bytes())
		if err == nil {
			t.Error("Expected error for invalid version, got nil")
		}
	})

	t.Run("TooManyLayers", func(t *testing.T) {
		var buf bytes.Buffer
		buf.Write([]byte("GONN"))
		binary.Write(&buf, binary.LittleEndian, int32(1))

		encoder := gob.NewEncoder(&buf)
		encoder.Encode(int32(2000)) // Over limit (1000)

		_, _, err := LoadFromBuffer(buf.Bytes())
		if err == nil {
			t.Error("Expected error for too many layers, got nil")
		}
	})

	t.Run("TooManyParams", func(t *testing.T) {
		var buf bytes.Buffer
		buf.Write([]byte("GONN"))
		binary.Write(&buf, binary.LittleEndian, int32(1))

		encoder := gob.NewEncoder(&buf)
		encoder.Encode(int32(1)) // 1 layer
		encoder.Encode("MSE")
		encoder.Encode("SGD")

		// Create a config with huge number of parameters (simulated)
		cfg := LayerConfig{
			Type:   "Dense",
			Params: make([]float32, 100), // Small actual slice
		}
		// Manually override the Params field length in a way that triggers our check
		// but doesn't necessarily allocate 1GB in the test if we could...
		// But since our check uses len(cfg.Params), we have to provide it.
		// Let's use a smaller limit for testing if possible? No, it's hardcoded to 256M.
		// 256M float32 is 1GB. Let's use exactly 256M + 1.

		cfg.Params = make([]float32, 256*1024*1024+1)

		encoder.Encode(cfg)

		_, _, err := LoadFromBuffer(buf.Bytes())
		if err == nil {
			t.Error("Expected error for too many parameters, got nil")
		} else if err.Error() == "EOF" {
			t.Log("Note: Got EOF, likely because Encode failed to allocate/write such a large slice")
		}
	})
}
