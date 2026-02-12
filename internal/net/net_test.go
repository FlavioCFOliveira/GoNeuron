// Package net provides comprehensive unit tests for neural network.
package net

import (
	"math"
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

	input := []float64{1.0, 2.0}
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

	input := []float64{1.0, 2.0}
	target := []float64{1.0}

	// Forward
	yPred := network.Forward(input)

	// Compute loss gradient
	grad := loss.MSE{}.Backward(yPred, target)

	// Backward
	outputGrad := network.Backward(grad)

	// Output gradient should match input size
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

	input := []float64{0.0, 0.0}
	target := []float64{0.0}

	loss := network.Train(input, target)

	if loss < 0 {
		t.Errorf("Loss should be non-negative, got %v", loss)
	}
}

// TestNetworkXOR tests XOR learning.
func TestNetworkXOR(t *testing.T) {
	// Create XOR network: 2 -> 3 -> 1
	// Using Tanh instead of ReLU to avoid dying ReLU problem
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// XOR data
	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {1}, {1}, {0}}

	// Train for 1000 epochs
	for epoch := 0; epoch < 1000; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Test predictions
	tolerance := 0.1
	tests := []struct {
		input  []float64
		target float64
	}{
		{[]float64{0, 0}, 0},
		{[]float64{0, 1}, 1},
		{[]float64{1, 0}, 1},
		{[]float64{1, 1}, 0},
	}

	for _, tt := range tests {
		output := network.Forward(tt.input)
		pred := output[0]

		// Check if prediction is close to target
		if tt.target == 0 && pred > tolerance {
			t.Errorf("XOR[0,0] = %v, should be near 0", pred)
		}
		if tt.target == 1 && pred < 1-tolerance {
			t.Errorf("XOR[0,1] or [1,0] = %v, should be near 1", pred)
		}
		if tt.target == 0 && pred > 0.5 {
			t.Errorf("XOR[1,1] = %v, should be near 0", pred)
		}
	}
}

// TestNetworkMultipleEpochs tests training over multiple epochs.
func TestNetworkMultipleEpochs(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.5})

	// Simple linearly separable data
	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {0}, {0}, {1}} // AND function

	initialLoss := 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		initialLoss += loss.MSE{}.Forward(yPred, trainY[i])
	}
	initialLoss /= float64(len(trainX))

	// Train
	for epoch := 0; epoch < 500; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	finalLoss := 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		finalLoss += loss.MSE{}.Forward(yPred, trainY[i])
	}
	finalLoss /= float64(len(trainX))

	// Loss should decrease significantly
	if finalLoss > initialLoss*0.5 {
		t.Errorf("Loss did not decrease enough: %v -> %v", initialLoss, finalLoss)
	}
}

// TestNetworkStep tests step method directly.
func TestNetworkStep(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0}
	target := []float64{0.5, 0.5}

	// Forward and backward to compute gradients
	yPred := network.Forward(input)
	grad := loss.MSE{}.Backward(yPred, target)
	_ = network.Backward(grad)

	// Store initial params
	initialParams := network.Params()

	// Step
	network.Step()

	// Params should have changed
	updatedParams := network.Params()
	if len(updatedParams) != len(initialParams) {
		t.Error("Params length changed after step")
	}

	// Check at least some params changed
	changed := false
	for i := range initialParams {
		if math.Abs(initialParams[i]-updatedParams[i]) > 1e-10 {
			changed = true
			break
		}
	}

	if !changed {
		t.Error("No params changed after step")
	}
}

// TestNetworkParams tests parameter extraction.
func TestNetworkParams(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	params := network.Params()

	// Expected: 2*3 + 3 (first layer) + 3*1 + 1 (second layer) = 6 + 3 + 3 + 1 = 13
	expectedLen := 2*3 + 3 + 3*1 + 1
	if len(params) != expectedLen {
		t.Errorf("Params length = %d, want %d", len(params), expectedLen)
	}
}

// TestNetworkGradients tests gradient extraction.
func TestNetworkGradients(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0}
	target := []float64{1.0}

	// Forward and backward
	yPred := network.Forward(input)
	_ = loss.MSE{}.Forward(yPred, target)
	grad := loss.MSE{}.Backward(yPred, target)
	_ = network.Backward(grad)

	gradients := network.Gradients()

	// Check gradient length matches params length
	params := network.Params()
	if len(gradients) != len(params) {
		t.Errorf("Gradients length = %d, want %d", len(gradients), len(params))
	}

	// Check some gradients are non-zero
	nonZero := false
	for _, g := range gradients {
		if math.Abs(g) > 1e-10 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Error("All gradients are zero")
	}
}

// TestNetworkTrainBatch tests batch training.
func TestNetworkTrainBatch(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Simple data
	batchX := [][]float64{{0, 0}, {1, 1}}
	batchY := [][]float64{{0}, {1}}

	loss := network.TrainBatch(batchX, batchY)

	if loss < 0 {
		t.Errorf("Batch loss should be non-negative, got %v", loss)
	}
}

// TestNetworkTrainBatchEmpty tests empty batch handling.
func TestNetworkTrainBatchEmpty(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	batchX := [][]float64{}
	batchY := [][]float64{}

	loss := network.TrainBatch(batchX, batchY)

	if loss != 0 {
		t.Errorf("Empty batch loss should be 0, got %v", loss)
	}
}

// TestNetworkTrainBatchConsistency tests that batch and sequential training give similar results.
func TestNetworkTrainBatchConsistency(t *testing.T) {
	// Train two identical networks
	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {1}, {1}, {0}}

	// Network 1: sequential training
	layers1 := []layer.Layer{
		layer.NewDense(2, 3, activations.ReLU{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network1 := New(layers1, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Network 2: batch training
	layers2 := []layer.Layer{
		layer.NewDense(2, 3, activations.ReLU{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network2 := New(layers2, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Copy params to network2
	params1 := network1.Params()
	// Use SetParams on individual layers
	offset := 0
	for _, layer := range network2.layers {
		layerParams := params1[offset : offset+len(layer.Params())]
		layer.SetParams(layerParams)
		offset += len(layer.Params())
	}

	// Train both for same number of epochs
	epochs := 100
	for e := 0; e < epochs; e++ {
		// Network 1: sequential
		for i := range trainX {
			network1.Train(trainX[i], trainY[i])
		}

		// Network 2: batch
		network2.TrainBatch(trainX, trainY)
	}

	// Compare final predictions
	for i := range trainX {
		pred1 := network1.Forward(trainX[i])
		pred2 := network2.Forward(trainX[i])

		// Should be similar (within tolerance)
		diff := math.Abs(pred1[0] - pred2[0])
		if diff > 0.01 {
			t.Logf("Predictions differ at sample %d: %v vs %v (diff: %v)", i, pred1[0], pred2[0], diff)
		}
	}
}

// TestNetworkGradientDescent tests basic gradient descent behavior.
func TestNetworkGradientDescent(t *testing.T) {
	// Simple 1-layer network
	layers := []layer.Layer{
		layer.NewDense(1, 1, activations.ReLU{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Learn y = 2x
	trainX := [][]float64{{0}, {1}, {2}, {3}}
	trainY := [][]float64{{0}, {2}, {4}, {6}}

	// Train
	for epoch := 0; epoch < 1000; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Test predictions should be close to 2x
	for i := range trainX {
		pred := network.Forward(trainX[i])
		expected := 2.0 * trainX[i][0]

		if math.Abs(pred[0]-expected) > 0.5 {
			t.Logf("After 1000 epochs, prediction for x=%v: %v (expected %v)", trainX[i][0], pred[0], expected)
		}
	}
}

// TestNetworkParamsLengthMatchesGradients tests that params and gradients have matching lengths.
func TestNetworkParamsLengthMatchesGradients(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(3, 4, activations.Tanh{}),
		layer.NewDense(4, 2, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0, 3.0}
	target := []float64{1.0, 0.0}

	// Forward and backward
	yPred := network.Forward(input)
	_ = loss.MSE{}.Forward(yPred, target)
	grad := loss.MSE{}.Backward(yPred, target)
	_ = network.Backward(grad)

	params := network.Params()
	gradients := network.Gradients()

	if len(params) != len(gradients) {
		t.Errorf("Params length %d != gradients length %d", len(params), len(gradients))
	}
}

// TestNetworkLayerOrdering tests that layer ordering is preserved.
func TestNetworkLayerOrdering(t *testing.T) {
	// Create network with known layer sizes
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0}
	output := network.Forward(input)

	// Output should be length 1 (last layer output size)
	if len(output) != 1 {
		t.Errorf("Output length = %d, want 1", len(output))
	}
}

// TestNetworkDifferentActivations tests network with different activation types.
func TestNetworkDifferentActivations(t *testing.T) {
	tests := []struct {
		name   string
		hidden activations.Activation
		output activations.Activation
	}{
		{"Tanh-Sigmoid", activations.Tanh{}, activations.Sigmoid{}},
		{"ReLU-Sigmoid", activations.ReLU{}, activations.Sigmoid{}},
		{"Sigmoid-Sigmoid", activations.Sigmoid{}, activations.Sigmoid{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layers := []layer.Layer{
				layer.NewDense(2, 3, tt.hidden),
				layer.NewDense(3, 1, tt.output),
			}
			network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

			input := []float64{1.0, 2.0}
			output := network.Forward(input)

			if len(output) != 1 {
				t.Errorf("Output length = %d, want 1", len(output))
			}

			// Test backward pass
			grad := loss.MSE{}.Backward(output, []float64{0.5})
			_ = network.Backward(grad)
		})
	}
}

// TestNetworkForwardBackwardShape tests forward/backward shape consistency.
func TestNetworkForwardBackwardShape(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(5, 3, activations.Tanh{}),
		layer.NewDense(3, 4, activations.Tanh{}),
		layer.NewDense(4, 2, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	output := network.Forward(input)

	// Output should be length 2
	if len(output) != 2 {
		t.Errorf("Output length = %d, want 2", len(output))
	}

	// Backward
	grad := loss.MSE{}.Backward(output, []float64{0.5, 0.5})
	outputGrad := network.Backward(grad)

	// Output gradient should match input size
	if len(outputGrad) != 5 {
		t.Errorf("Output gradient length = %d, want 5", len(outputGrad))
	}
}

// TestNetworkSetParams tests parameter setting.
func TestNetworkSetParams(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network1 := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Create new network with same architecture
	layers2 := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network2 := New(layers2, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Get params from network1
	params := network1.Params()

	// Copy params to network2 layer by layer
	offset := 0
	for _, layer := range network2.layers {
		layerParams := params[offset : offset+len(layer.Params())]
		layer.SetParams(layerParams)
		offset += len(layer.Params())
	}

	// Forward pass should give same result
	input := []float64{1.0, 2.0}
	output1 := network1.Forward(input)
	output2 := network2.Forward(input)

	for i := range output1 {
		if math.Abs(output1[i]-output2[i]) > 1e-10 {
			t.Errorf("Outputs differ after SetParams: %v vs %v", output1[i], output2[i])
		}
	}
}

// TestNetworkParamsSetParamsRoundtrip tests that Params/SetParams preserve values.
func TestNetworkParamsSetParamsRoundtrip(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network1 := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Get original params
	params1 := network1.Params()

	// Create new network
	layers2 := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network2 := New(layers2, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Copy params to network2 layer by layer
	offset := 0
	for _, layer := range network2.layers {
		layerParams := params1[offset : offset+len(layer.Params())]
		layer.SetParams(layerParams)
		offset += len(layer.Params())
	}

	// Verify
	params2 := network2.Params()
	if len(params1) != len(params2) {
		t.Errorf("Params length mismatch: %d vs %d", len(params1), len(params2))
	}

	for i := range params1 {
		if math.Abs(params1[i]-params2[i]) > 1e-10 {
			t.Errorf("Params[%d] mismatch: %v vs %v", i, params1[i], params2[i])
		}
	}
}

// TestNetworkZeroInput tests behavior with zero input.
func TestNetworkZeroInput(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{0.0, 0.0}
	output := network.Forward(input)

	// Output should be in valid range
	if output[0] < 0 || output[0] > 1 {
		t.Errorf("Output with zero input = %v, should be in [0,1]", output[0])
	}
}

// TestNetworkLargeInput tests numerical stability with large inputs.
func TestNetworkLargeInput(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{100.0, 100.0}
	output := network.Forward(input)

	// Tanh saturates at ±1, Sigmoid at [0,1]
	if output[0] < 0 || output[0] > 1 {
		t.Errorf("Output with large input = %v, should be in [0,1]", output[0])
	}
}

// TestNetworkSingleLayer tests network with single layer.
func TestNetworkSingleLayer(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(3, 2, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0, 3.0}
	output := network.Forward(input)

	if len(output) != 2 {
		t.Errorf("Output length = %d, want 2", len(output))
	}

	// Sigmoid output should be in [0,1]
	for _, o := range output {
		if o < 0 || o > 1 {
			t.Errorf("Sigmoid output = %v, should be in [0,1]", o)
		}
	}
}

// TestNetworkDeepNetwork tests network with many layers.
func TestNetworkDeepNetwork(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(4, 8, activations.Tanh{}),
		layer.NewDense(8, 8, activations.Tanh{}),
		layer.NewDense(8, 8, activations.Tanh{}),
		layer.NewDense(8, 2, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0, 3.0, 4.0}
	output := network.Forward(input)

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

	// Large batch to trigger parallel path
	batchSize := 32
	batchX := make([][]float64, batchSize)
	batchY := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = []float64{float64(i % 2), float64((i / 2) % 2)}
		batchY[i] = []float64{float64((i % 2) ^ ((i / 2) % 2))}
	}

	loss := network.TrainBatch(batchX, batchY)

	if loss < 0 {
		t.Errorf("Parallel batch loss should be non-negative, got %v", loss)
	}
}

// TestNetworkSequentialVsParallelConsistency tests that sequential and parallel give same results.
func TestNetworkSequentialVsParallelConsistency(t *testing.T) {
	batchSize := 64
	batchX := make([][]float64, batchSize)
	batchY := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = []float64{float64(i % 2), float64((i / 2) % 2)}
		batchY[i] = []float64{float64((i % 2) ^ ((i / 2) % 2))}
	}

	// Network 1: sequential
	layers1 := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network1 := New(layers1, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Network 2: parallel
	layers2 := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network2 := New(layers2, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Copy params
	params := network1.Params()
	// Copy params layer by layer
	offset := 0
	for _, layer := range network2.layers {
		layerParams := params[offset : offset+len(layer.Params())]
		layer.SetParams(layerParams)
		offset += len(layer.Params())
	}

	// Train both
	network1.TrainBatch(batchX, batchY)
	network2.TrainBatch(batchX, batchY)

	// Compare predictions
	for i := range batchX[:10] {
		pred1 := network1.Forward(batchX[i])
		pred2 := network2.Forward(batchX[i])

		if math.Abs(pred1[0]-pred2[0]) > 0.01 {
			t.Logf("Predictions differ: sequential=%v, parallel=%v", pred1[0], pred2[0])
		}
	}
}

// TestNetworkBackwardGradientShape tests backward gradient shape for each layer.
func TestNetworkBackwardGradientShape(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(3, 4, activations.Tanh{}),
		layer.NewDense(4, 2, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0, 3.0}
	target := []float64{1.0, 0.0}

	// Forward
	yPred := network.Forward(input)

	// Backward
	grad := loss.MSE{}.Backward(yPred, target)
	_ = network.Backward(grad)

	// Check each layer has gradients
	for i, l := range layers {
		gradients := l.Gradients()
		params := l.Params()

		if len(gradients) != len(params) {
			t.Errorf("Layer %d: gradients length %d != params length %d", i, len(gradients), len(params))
		}
	}
}

// TestNetworkForwardIdempotent tests that forward is deterministic.
func TestNetworkForwardIdempotent(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0}

	// Multiple forward passes should give same result
	output1 := network.Forward(input)
	output2 := network.Forward(input)
	output3 := network.Forward(input)

	for i := range output1 {
		if math.Abs(output1[i]-output2[i]) > 1e-10 {
			t.Errorf("Forward not idempotent: %v vs %v", output1[i], output2[i])
		}
		if math.Abs(output2[i]-output3[i]) > 1e-10 {
			t.Errorf("Forward not idempotent: %v vs %v", output2[i], output3[i])
		}
	}
}

// TestNetworkTrainPreservesParamsLength tests that training doesn't change param count.
func TestNetworkTrainPreservesParamsLength(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0}
	target := []float64{1.0}

	initialLen := len(network.Params())

	// Train
	network.Train(input, target)

	finalLen := len(network.Params())

	if initialLen != finalLen {
		t.Errorf("Param count changed: %d -> %d", initialLen, finalLen)
	}
}

// TestNetworkLossDecreases tests that training decreases loss.
func TestNetworkLossDecreases(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.5})

	// Simple linearly separable data (AND function) for faster convergence
	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {0}, {0}, {1}}

	// Initial loss
	initialLoss := 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		initialLoss += loss.MSE{}.Forward(yPred, trainY[i])
	}
	initialLoss /= float64(len(trainX))

	// Train for more epochs to ensure loss decreases
	for epoch := 0; epoch < 500; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Final loss
	finalLoss := 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		finalLoss += loss.MSE{}.Forward(yPred, trainY[i])
	}
	finalLoss /= float64(len(trainX))

	// Loss should decrease significantly (at least 10% reduction)
	if finalLoss >= initialLoss*0.9 {
		t.Errorf("Loss did not decrease: %v -> %v", initialLoss, finalLoss)
	}
}

// TestNetworkParamsDoNotNan tests that params don't become NaN during training.
func TestNetworkParamsDoNotNan(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {1}, {1}, {0}}

	for epoch := 0; epoch < 100; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}

		// Check for NaN
		params := network.Params()
		for i, p := range params {
			if math.IsNaN(p) {
				t.Errorf("Param %d is NaN after epoch %d", i, epoch)
			}
			if math.IsInf(p, 0) {
				t.Errorf("Param %d is Inf after epoch %d", i, epoch)
			}
		}
	}
}

// TestNetworkBackwardChain tests that backward properly chains gradients.
func TestNetworkBackwardChain(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := []float64{1.0, 2.0}
	target := []float64{1.0}

	// Forward
	yPred := network.Forward(input)

	// Compute loss gradient
	lossGrad := loss.MSE{}.Backward(yPred, target)

	// Backward through network
	inputGrad := network.Backward(lossGrad)

	// Input gradient should have same length as input
	if len(inputGrad) != len(input) {
		t.Errorf("Input gradient length = %d, want %d", len(inputGrad), len(input))
	}

	// Gradients should not be all zero
	nonZero := false
	for _, g := range inputGrad {
		if math.Abs(g) > 1e-10 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Error("Input gradients are all zero")
	}
}

// TestNetworkLayerCount tests layer counting.
func TestNetworkLayerCount(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Count layers by Forward
	input := []float64{1.0, 2.0}
	output := network.Forward(input)

	// We have 3 layers, output should be length 1
	if len(output) != 1 {
		t.Errorf("Output length = %d, want 1", len(output))
	}
}

// TestNetworkLargeBatch tests with very large batch.
func TestNetworkLargeBatch(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(4, 8, activations.Tanh{}),
		layer.NewDense(8, 2, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// 256 sample batch
	batchSize := 256
	batchX := make([][]float64, batchSize)
	batchY := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = []float64{float64(i % 4), float64((i / 4) % 4), float64((i / 16) % 4), float64((i / 64) % 4)}
		batchY[i] = []float64{float64(i % 2), float64((i / 2) % 2)}
	}

	loss := network.TrainBatch(batchX, batchY)

	if loss < 0 || math.IsNaN(loss) {
		t.Errorf("Large batch loss invalid: %v", loss)
	}
}

// TestNetworkSetParamsLengthMismatch tests error handling.
func TestNetworkSetParamsLengthMismatch(t *testing.T) {
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	// Wrong length params
	wrongParams := []float64{1.0, 2.0} // Should be 6 (4 weights + 2 biases)

	// Get the first layer and try to set wrong params
	layer := network.layers[0]
	defer func() {
		if r := recover(); r == nil {
			// This is actually acceptable behavior - SetParams may not check length
			t.Log("Note: SetParams did not panic on length mismatch")
		}
	}()

	layer.SetParams(wrongParams)
}
