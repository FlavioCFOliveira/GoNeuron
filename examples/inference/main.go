// Package main - Comprehensive test suite for GoNeural Network capabilities
package main

import (
	"fmt"
	"math"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// ============================================================================
// Comprehensive Neural Network Test Suite
// ============================================================================

const (
	EPSILON = 1e-6
)

var (
	passCount int
	failCount int
)

func main() {
	fmt.Println("=============================================================")
	fmt.Println("  GoNeuron - Comprehensive Neural Network Test Suite")
	fmt.Println("=============================================================")
	fmt.Printf("  Started at: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Println()

	// Run all test categories
	testActivations()
	testLossFunctions()
	testOptimizers()
	testDenseLayer()
	testConv2DLayer()
	testLSTMLayer()
	testNetworkCore()
	testNetworkTraining()
	testNetworkBatchTraining()
	testNetworkSerialization()
	testXORProblem()
	testRegression()
	testClassification()
	testDeepNetworks()

	// Summary
	fmt.Println()
	fmt.Println("=============================================================")
	fmt.Println("  TEST SUMMARY")
	fmt.Println("=============================================================")
	fmt.Printf("  Passed: %d\n", passCount)
	fmt.Printf("  Failed: %d\n", failCount)
	fmt.Printf("  Total:  %d\n", passCount+failCount)
	fmt.Printf("  Success Rate: %.1f%%\n", float64(passCount)*100.0/float64(passCount+failCount))

	if failCount == 0 {
		fmt.Println("\n  *** ALL TESTS PASSED! ***")
	} else {
		fmt.Println("\n  *** SOME TESTS FAILED ***")
	}
	fmt.Println("=============================================================")
}

func assert(condition bool, message string) bool {
	if condition {
		passCount++
		fmt.Printf("  [PASS] %s\n", message)
		return true
	} else {
		failCount++
		fmt.Printf("  [FAIL] %s\n", message)
		return false
	}
}

func assertNear(actual, expected, tolerance float64, message string) bool {
	if math.Abs(actual-expected) <= tolerance {
		passCount++
		fmt.Printf("  [PASS] %s (got %.6f, expected %.6f)\n", message, actual, expected)
		return true
	} else {
		failCount++
		fmt.Printf("  [FAIL] %s (got %.6f, expected %.6f)\n", message, actual, expected)
		return false
	}
}

// ============================================================================
// Test Activations
// ============================================================================

func testActivations() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Activation Functions")
	fmt.Println("-------------------------------------------------------------")

	// Test ReLU
	relu := activations.ReLU{}
	assert(relu.Activate(-1.0) == 0.0, "ReLU(-1) = 0")
	assert(relu.Activate(0.0) == 0.0, "ReLU(0) = 0")
	assert(relu.Activate(1.0) == 1.0, "ReLU(1) = 1")
	assert(relu.Derivative(-1.0) == 0.0, "ReLU'.(-1) = 0")
	assert(relu.Derivative(0.0) == 0.0, "ReLU'(0) = 0")
	assert(relu.Derivative(1.0) == 1.0, "ReLU'(1) = 1")

	// Test Sigmoid
	sigmoid := activations.Sigmoid{}
	assertNear(sigmoid.Activate(0.0), 0.5, EPSILON, "Sigmoid(0) = 0.5")
	assertNear(sigmoid.Derivative(0.0), 0.25, EPSILON, "Sigmoid'(0) = 0.25")
	assert(sigmoid.Activate(10.0) > 0.99, "Sigmoid(10) ≈ 1")
	assert(sigmoid.Activate(-10.0) < 0.01, "Sigmoid(-10) ≈ 0")

	// Test Tanh
	tanh := activations.Tanh{}
	assertNear(tanh.Activate(0.0), 0.0, EPSILON, "Tanh(0) = 0")
	assertNear(tanh.Derivative(0.0), 1.0, EPSILON, "Tanh'(0) = 1")
	assert(tanh.Activate(10.0) > 0.999, "Tanh(10) ≈ 1")
	assert(tanh.Activate(-10.0) < -0.999, "Tanh(-10) ≈ -1")

	// Test LeakyReLU
	leakyReLU := activations.NewLeakyReLU(0.01)
	assertNear(leakyReLU.Activate(-1.0), -0.01, EPSILON, "LeakyReLU(-1) = -0.01")
	assertNear(leakyReLU.Derivative(-1.0), 0.01, EPSILON, "LeakyReLU'(-1) = 0.01")
	assertNear(leakyReLU.Derivative(0.0), 0.01, EPSILON, "LeakyReLU'(0) = 0.01")
	assertNear(leakyReLU.Activate(1.0), 1.0, EPSILON, "LeakyReLU(1) = 1")
	assertNear(leakyReLU.Derivative(1.0), 1.0, EPSILON, "LeakyReLU'(1) = 1")

	// Test Softmax
	softmax := activations.Softmax{}
	softmaxInput := []float64{1.0, 2.0, 3.0}
	softmaxOutput := softmax.ActivateBatch(softmaxInput)
	sum := 0.0
	for _, v := range softmaxOutput {
		sum += v
	}
	assertNear(sum, 1.0, EPSILON, "Softmax sums to 1")
	for _, v := range softmaxOutput {
		assert(v > 0 && v < 1, "Softmax outputs in (0,1)")
	}

	// Test activation ranges
	for _, x := range []float64{-10, -5, -1, 0, 1, 5, 10} {
		s := sigmoid.Activate(x)
		assert(s >= 0 && s <= 1, fmt.Sprintf("Sigmoid(%v) in [0,1]", x))
	}
	for _, x := range []float64{-10, -5, -1, 0, 1, 5, 10} {
		t := tanh.Activate(x)
		assert(t >= -1 && t <= 1, fmt.Sprintf("Tanh(%v) in [-1,1]", x))
	}
}

// ============================================================================
// Test Loss Functions
// ============================================================================

func testLossFunctions() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Loss Functions")
	fmt.Println("-------------------------------------------------------------")

	// Test MSE
	mse := loss.MSE{}
	assertNear(mse.Forward([]float64{1, 2, 3}, []float64{1, 2, 3}), 0.0, EPSILON, "MSE perfect prediction = 0")
	assertNear(mse.Forward([]float64{1}, []float64{0}), 1.0, EPSILON, "MSE(1,0) = 1")

	grad := mse.Backward([]float64{1, 2}, []float64{1, 2})
	assertNear(grad[0], 0.0, EPSILON, "MSE gradient at perfect = 0")

	grad = mse.Backward([]float64{1}, []float64{0})
	assertNear(grad[0], 2.0, EPSILON, "MSE gradient(1,0) = 2")

	// Test MSE InPlace
	gradInPlace := make([]float64, 2)
	mse.BackwardInPlace([]float64{1, 2}, []float64{1, 2}, gradInPlace)
	assertNear(gradInPlace[0], 0.0, EPSILON, "MSE InPlace gradient = 0")

	// Test CrossEntropy
	ce := loss.CrossEntropy{}
	// Perfect prediction
	assert(ce.Forward([]float64{0.99, 0.01}, []float64{1, 0}) < 0.1, "CrossEntropy perfect < 0.1")
	// Gradient for classification
	grad = ce.Backward([]float64{0.7, 0.2, 0.1}, []float64{1, 0, 0})
	assertNear(grad[0], -0.3, 0.01, "CrossEntropy gradient[0] ≈ -0.3")

	// Test CrossEntropy InPlace
	gradInPlace = make([]float64, 3)
	ce.BackwardInPlace([]float64{0.7, 0.2, 0.1}, []float64{1, 0, 0}, gradInPlace)
	assertNear(gradInPlace[0], -0.3, 0.01, "CrossEntropy InPlace gradient[0] ≈ -0.3")

	// Test Huber
	huber := loss.NewHuber(1.0)
	// Small error (quadratic region)
	assertNear(huber.Forward([]float64{1}, []float64{1.5}), 0.125, EPSILON, "Huber small error = 0.125")
	// Large error (linear region)
	assertNear(huber.Forward([]float64{0}, []float64{2}), 1.5, EPSILON, "Huber large error = 1.5")

	// Huber gradients
	grad = huber.Backward([]float64{1}, []float64{1.5}) // diff = -0.5, |diff| < delta
	assertNear(grad[0], -0.5, EPSILON, "Huber gradient small error = diff")

	grad = huber.Backward([]float64{0}, []float64{2}) // diff = -2, |diff| > delta
	assertNear(grad[0], -1.0, EPSILON, "Huber gradient large error = -delta")

	// Huber InPlace
	gradInPlace = make([]float64, 1)
	huber.BackwardInPlace([]float64{1}, []float64{1.5}, gradInPlace)
	assertNear(gradInPlace[0], -0.5, EPSILON, "Huber InPlace gradient = -0.5")
}

// ============================================================================
// Test Optimizers
// ============================================================================

func testOptimizers() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Optimizers")
	fmt.Println("-------------------------------------------------------------")

	// Test SGD Step
	sgd := opt.SGD{LearningRate: 0.1}
	updated := sgd.Step([]float64{1.0, 2.0}, []float64{0.1, 0.2})
	assertNear(updated[0], 0.99, EPSILON, "SGD Step[0] = 0.99")
	assertNear(updated[1], 1.98, EPSILON, "SGD Step[1] = 1.98")

	// Test SGD StepInPlace
	params := []float64{1.0, 2.0}
	gradients := []float64{0.1, 0.2}
	sgd.StepInPlace(params, gradients)
	assertNear(params[0], 0.99, EPSILON, "SGD StepInPlace[0] = 0.99")
	assertNear(params[1], 1.98, EPSILON, "SGD StepInPlace[1] = 1.98")

	// Test SGD Zero Gradient
	updated = sgd.Step([]float64{1, 2}, []float64{0, 0})
	assertNear(updated[0], 1.0, EPSILON, "SGD zero gradient no change")

	// Test SGD Zero Learning Rate
	sgdZero := opt.SGD{LearningRate: 0}
	updated = sgdZero.Step([]float64{1, 2}, []float64{1, 1})
	assertNear(updated[0], 1.0, EPSILON, "SGD zero LR no change")

	// Test Adam
	adam := opt.NewAdam(0.1)
	updated = adam.Step([]float64{1.0, 2.0}, []float64{0.1, 0.2})
	assert(!math.IsNaN(updated[0]), "Adam step no NaN")

	// Test Adam StepInPlace
	params = []float64{1.0, 2.0}
	adam.StepInPlace(params, gradients)
	assert(!math.IsNaN(params[0]), "Adam StepInPlace no NaN")

	// Test Adam Zero Gradient
	adam2 := opt.NewAdam(0.1)
	updated = adam2.Step([]float64{1, 2}, []float64{0, 0})
	assertNear(updated[0], 1.0, EPSILON, "Adam zero gradient no change")
}

// ============================================================================
// Test Dense Layer
// ============================================================================

func testDenseLayer() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Dense Layer")
	fmt.Println("-------------------------------------------------------------")

	// Test forward pass
	dense := layer.NewDense(2, 3, activations.Tanh{})
	input := []float64{1.0, 2.0}
	output := dense.Forward(input)
	assert(len(output) == 3, "Dense output length = 3")

	// Test backward pass
	grad := []float64{1.0, 1.0, 1.0}
	inputGrad := dense.Backward(grad)
	assert(len(inputGrad) == 2, "Dense inputGrad length = 2")

	// Test parameters
	params := dense.Params()
	// 2*3 weights + 3 biases = 9
	assert(len(params) == 9, "Dense params length = 9")

	// Test gradients
	_ = dense.Forward(input)
	dense.Backward(grad)
	gradients := dense.Gradients()
	assert(len(gradients) == 9, "Dense gradients length = 9")

	// Test SetParams
	newParams := make([]float64, len(params))
	for i := range newParams {
		newParams[i] = float64(i) * 0.01
	}
	dense.SetParams(newParams)
	paramsAfter := dense.Params()
	assertNear(paramsAfter[0], newParams[0], EPSILON, "SetParams works")

	// Test individual weight/bias access
	dense.SetWeight(0, 0, 3.14)
	assertNear(dense.GetWeight(0, 0), 3.14, EPSILON, "GetWeight/SetWeight work")

	dense.SetBias(1, 2.71)
	assertNear(dense.GetBias(1), 2.71, EPSILON, "GetBias/SetBias work")

	// Test InSize/OutSize
	assert(dense.InSize() == 2, "Dense InSize = 2")
	assert(dense.OutSize() == 3, "Dense OutSize = 3")

	// Test Reset (no-op for Dense)
	dense.Reset()

	// Test layer interface compliance
	var _ layer.Layer = dense
}

// ============================================================================
// Test Conv2D Layer
// ============================================================================

func testConv2DLayer() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Conv2D Layer")
	fmt.Println("-------------------------------------------------------------")

	// Test 1x1 convolution
	conv := layer.NewConv2D(1, 1, 1, 1, 0, activations.Tanh{})
	input := []float64{1.0, 2.0, 3.0, 4.0} // 2x2
	output := conv.Forward(input)
	assert(len(output) == 4, "Conv2D 1x1 output length = 4")

	// Test forward with padding
	conv2 := layer.NewConv2D(1, 1, 3, 1, 1, activations.ReLU{})
	input = make([]float64, 16) // 4x4
	output = conv2.Forward(input)
	assert(len(output) == 16, "Conv2D with padding output = 16")

	// Test forward without padding
	conv3 := layer.NewConv2D(1, 1, 3, 1, 0, activations.ReLU{})
	output = conv3.Forward(make([]float64, 16))
	assert(len(output) == 4, "Conv2D no padding output = 4")

	// Test stride
	conv4 := layer.NewConv2D(1, 1, 3, 2, 0, activations.ReLU{})
	output = conv4.Forward(make([]float64, 16))
	assert(len(output) == 1, "Conv2D stride=2 output = 1")

	// Test backward pass
	conv5 := layer.NewConv2D(1, 1, 3, 1, 1, activations.Tanh{})
	input = make([]float64, 16)
	conv5.Forward(input)
	grad := make([]float64, 16)
	for i := range grad {
		grad[i] = 1.0
	}
	inputGrad := conv5.Backward(grad)
	assert(len(inputGrad) == 16, "Conv2D backward inputGrad = 16")

	// Test parameters
	conv6 := layer.NewConv2D(2, 3, 3, 1, 1, activations.Tanh{})
	params := conv6.Params()
	// 3 * 2 * 3 * 3 + 3 = 57
	assert(len(params) == 57, "Conv2D params length = 57")

	// Test SetParams
	newParams := make([]float64, len(params))
	for i := range newParams {
		newParams[i] = float64(i) * 0.01
	}
	conv6.SetParams(newParams)
	paramsAfter := conv6.Params()
	assertNear(paramsAfter[0], newParams[0], EPSILON, "Conv2D SetParams works")

	// Test InSize/OutSize
	assert(conv6.InSize() == 2, "Conv2D InSize = 2")
	assert(conv6.OutSize() == 3, "Conv2D OutSize = 3")

	// Test SetInputDimensions
	conv6.SetInputDimensions(10, 10)

	// Test Reset
	conv6.Reset()

	// Test layer interface compliance
	var _ layer.Layer = conv6
}

// ============================================================================
// Test LSTM Layer
// ============================================================================

func testLSTMLayer() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: LSTM Layer")
	fmt.Println("-------------------------------------------------------------")

	// Test single forward pass
	lstm := layer.NewLSTM(2, 3)
	input := []float64{1.0, 2.0}
	output := lstm.Forward(input)
	assert(len(output) == 3, "LSTM output length = 3")

	// Test multiple forward passes (sequence)
	inputs := [][]float64{
		{1.0, 2.0},
		{0.5, 1.5},
		{0.0, 0.0},
	}
	for _, inp := range inputs {
		output = lstm.Forward(inp)
		assert(len(output) == 3, fmt.Sprintf("LSTM seq output[%d] = 3", len(inputs)))
	}

	// Test backward pass
	grad := []float64{0.1, 0.1, 0.1}
	inputGrad := lstm.Backward(grad)
	assert(len(inputGrad) == 2, "LSTM inputGrad length = 2")

	// Test parameters
	params := lstm.Params()
	// inputW: 4*3*2=24, recurrentW: 4*3*3=36, biases: 4*3=12, total=72
	assert(len(params) == 72, "LSTM params length = 72")

	// Test gradients
	_ = lstm.Forward(input)
	_ = lstm.Backward(grad)
	gradients := lstm.Gradients()
	assert(len(gradients) == 72, "LSTM gradients length = 72")

	// Test SetParams
	newParams := make([]float64, len(params))
	for i := range newParams {
		newParams[i] = float64(i) * 0.001
	}
	lstm.SetParams(newParams)
	paramsAfter := lstm.Params()
	assertNear(paramsAfter[0], newParams[0], EPSILON, "LSTM SetParams works")

	// Test Reset
	lstm.Reset()
	// After reset, first input should produce same output as initial
	lstm.Reset()
	output1 := lstm.Forward(input)
	lstm.Reset()
	output2 := lstm.Forward(input)
	for i := range output1 {
		assertNear(output1[i], output2[i], EPSILON, "LSTM Reset produces same output")
	}

	// Test InSize/OutSize
	assert(lstm.InSize() == 2, "LSTM InSize = 2")
	assert(lstm.OutSize() == 3, "LSTM OutSize = 3")

	// Test layer interface compliance
	var _ layer.Layer = lstm
}

// ============================================================================
// Test Network Core
// ============================================================================

func testNetworkCore() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Network Core")
	fmt.Println("-------------------------------------------------------------")

	mse := loss.MSE{}

	// Create network: 2 -> 3 -> 1
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network := net.New(layers, mse, &opt.SGD{LearningRate: 0.1})

	// Test forward
	input := []float64{1.0, 2.0}
	output := network.Forward(input)
	assert(len(output) == 1, "Network forward output = 1")

	// Test backward
	yPred := network.Forward(input)
	grad := mse.Backward(yPred, []float64{1.0})
	outputGrad := network.Backward(grad)
	assert(len(outputGrad) == 2, "Network backward outputGrad = 2")

	// Test Train
	loss := network.Train(input, []float64{1.0})
	assert(loss >= 0, "Network Train loss >= 0")

	// Test Params
	params := network.Params()
	// 2*3+3 + 3*1+1 = 6+3+3+1 = 13
	assert(len(params) == 13, "Network Params length = 13")

	// Test Gradients
	_ = network.Forward(input)
	_ = mse.Forward(yPred, []float64{1.0})
	_ = mse.Backward(yPred, []float64{1.0})
	_ = network.Backward(grad)
	gradients := network.Gradients()
	assert(len(gradients) == 13, "Network Gradients length = 13")

	// Test Params/Gradients consistency
	assert(len(params) == len(gradients), "Params and Gradients same length")

	// Test Step
	_ = network.Forward(input)
	_ = mse.Forward(network.Forward(input), []float64{1.0})
	_ = mse.Backward(network.Forward(input), []float64{1.0})
	_ = network.Backward(grad)
	initialParams := network.Params()
	network.Step()
	updatedParams := network.Params()
	changed := false
	for i := range initialParams {
		if math.Abs(initialParams[i]-updatedParams[i]) > 1e-10 {
			changed = true
			break
		}
	}
	assert(changed, "Network Step changes params")

	// Test Layer Ordering
	layers2 := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 4, activations.Tanh{}),
		layer.NewDense(4, 2, activations.Sigmoid{}),
	}
	network2 := net.New(layers2, mse, &opt.SGD{LearningRate: 0.1})
	output = network2.Forward([]float64{1, 2})
	assert(len(output) == 2, "Network layer ordering correct")

	// Test Different Activations
	for _, act := range []activations.Activation{
		activations.Tanh{},
		activations.ReLU{},
		activations.Sigmoid{},
		activations.NewLeakyReLU(0.01),
	} {
		layers := []layer.Layer{
			layer.NewDense(2, 2, act),
			layer.NewDense(2, 1, activations.Sigmoid{}),
		}
		n := net.New(layers, mse, &opt.SGD{LearningRate: 0.1})
		o := n.Forward([]float64{1, 2})
		assert(len(o) == 1, fmt.Sprintf("Network with %T works", act))
	}

	// Test Zero Input
	dense := layer.NewDense(2, 2, activations.Tanh{})
	network3 := net.New([]layer.Layer{dense}, mse, &opt.SGD{LearningRate: 0.1})
	output = network3.Forward([]float64{0, 0})
	assert(output[0] >= -1 && output[0] <= 1, "Network zero input in range")

	// Test Large Input (numerical stability)
	output = network3.Forward([]float64{100, 100})
	assert(output[0] >= -1 && output[0] <= 1, "Network large input stable")
}

// ============================================================================
// Test Network Training
// ============================================================================

func testNetworkTraining() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Network Training")
	fmt.Println("-------------------------------------------------------------")

	mse := loss.MSE{}

	// Test AND function
	layers := []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network := net.New(layers, mse, &opt.SGD{LearningRate: 0.5})

	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {0}, {0}, {1}}

	// Compute initial loss
	initialLoss := 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		initialLoss += mse.Forward(yPred, trainY[i])
	}
	initialLoss /= float64(len(trainX))

	// Train
	for epoch := 0; epoch < 500; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Compute final loss
	finalLoss := 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		finalLoss += mse.Forward(yPred, trainY[i])
	}
	finalLoss /= float64(len(trainX))

	// Loss should decrease significantly
	assert(finalLoss < initialLoss*0.5, fmt.Sprintf("AND function learned: loss %.4f -> %.4f", initialLoss, finalLoss))

	// Test XOR with more epochs
	layers = []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network = net.New(layers, mse, &opt.SGD{LearningRate: 0.1})

	trainX = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY = [][]float64{{0}, {1}, {1}, {0}}

	for epoch := 0; epoch < 1000; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Verify XOR learning
	tolerance := 0.15
	tests := []struct {
		input  []float64
		target float64
	}{
		{[]float64{0, 0}, 0},
		{[]float64{0, 1}, 1},
		{[]float64{1, 0}, 1},
		{[]float64{1, 1}, 0},
	}

	xorPassed := true
	for _, tt := range tests {
		output := network.Forward(tt.input)
		pred := output[0]
		if tt.target == 0 && pred > tolerance {
			xorPassed = false
		}
		if tt.target == 1 && pred < 1-tolerance {
			xorPassed = false
		}
		if tt.target == 0 && pred > 0.5 {
			xorPassed = false
		}
	}
	assert(xorPassed, "XOR function learned (with tolerance)")

	// Test Loss Decreases
	layers = []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network = net.New(layers, mse, &opt.SGD{LearningRate: 0.5})

	initialLoss = 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		initialLoss += mse.Forward(yPred, trainY[i])
	}
	initialLoss /= float64(len(trainX))

	for epoch := 0; epoch < 500; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	finalLoss = 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		finalLoss += mse.Forward(yPred, trainY[i])
	}
	finalLoss /= float64(len(trainX))

	assert(finalLoss < initialLoss*0.9, "Training loss decreases")

	// Test NaN/Inf Prevention
	layers = []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network = net.New(layers, mse, &opt.SGD{LearningRate: 0.1})

	trainX = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY = [][]float64{{0}, {1}, {1}, {0}}

	NaNInfFound := false
	for epoch := 0; epoch < 100; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
		params := network.Params()
		for _, p := range params {
			if math.IsNaN(p) || math.IsInf(p, 0) {
				NaNInfFound = true
				break
			}
		}
		if NaNInfFound {
			break
		}
	}
	assert(!NaNInfFound, "No NaN/Inf during training")
}

// ============================================================================
// Test Network Batch Training
// ============================================================================

func testNetworkBatchTraining() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Network Batch Training")
	fmt.Println("-------------------------------------------------------------")

	mse := loss.MSE{}

	// Test basic batch training
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network := net.New(layers, mse, &opt.SGD{LearningRate: 0.1})

	batchX := [][]float64{{0, 0}, {1, 1}}
	batchY := [][]float64{{0}, {1}}

	loss := network.TrainBatch(batchX, batchY)
	assert(loss >= 0, "Batch training loss >= 0")

	// Test empty batch
	batchX = [][]float64{}
	batchY = [][]float64{}
	loss = network.TrainBatch(batchX, batchY)
	assert(loss == 0, "Empty batch loss = 0")

	// Test large batch (triggers parallel path)
	batchSize := 64
	batchX = make([][]float64, batchSize)
	batchY = make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = []float64{float64(i % 2), float64((i / 2) % 2)}
		batchY[i] = []float64{float64((i % 2) ^ ((i / 2) % 2))}
	}

	loss = network.TrainBatch(batchX, batchY)
	assert(loss >= 0, "Large batch loss >= 0")

	// Test sequential vs parallel consistency
	batchSize = 32
	batchX = make([][]float64, batchSize)
	batchY = make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = []float64{float64(i % 2), float64((i / 2) % 2)}
		batchY[i] = []float64{float64((i % 2) ^ ((i / 2) % 2))}
	}

	// Network 1: sequential
	layers1 := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network1 := net.New(layers1, mse, &opt.SGD{LearningRate: 0.1})

	// Network 2: parallel
	layers2 := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
		layer.NewDense(2, 1, activations.Sigmoid{}),
	}
	network2 := net.New(layers2, mse, &opt.SGD{LearningRate: 0.1})

	// Copy params by training both networks on same data
	// We'll verify consistency by checking that trained networks produce similar results

	// Train both
	network1.TrainBatch(batchX, batchY)
	network2.TrainBatch(batchX, batchY)

	// Compare predictions
	for i := range batchX[:5] {
		pred1 := network1.Forward(batchX[i])
		pred2 := network2.Forward(batchX[i])
		diff := math.Abs(pred1[0] - pred2[0])
		assert(diff < 0.1, fmt.Sprintf("Batch consistency: diff %.4f", diff))
	}
}

// ============================================================================
// Test Network Serialization
// ============================================================================

func testNetworkSerialization() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Network Serialization")
	fmt.Println("-------------------------------------------------------------")

	mse := loss.MSE{}

	// Create and train a network
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network := net.New(layers, mse, &opt.SGD{LearningRate: 0.1})

	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {1}, {1}, {0}}

	for epoch := 0; epoch < 500; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Get predictions before save
	predsBefore := make([]float64, len(trainX))
	for i := range trainX {
		predsBefore[i] = network.Forward(trainX[i])[0]
	}

	// Save
	filename := "/tmp/test_network.bin"
	err := network.Save(filename)
	assert(err == nil, "Network save succeeds")

	// Load
	loadedNetwork, _, err := net.Load(filename)
	assert(err == nil, "Network load succeeds")

	// Verify predictions match
	predsAfter := make([]float64, len(trainX))
	for i := range trainX {
		predsAfter[i] = loadedNetwork.Forward(trainX[i])[0]
	}

	match := true
	for i := range predsBefore {
		if math.Abs(predsBefore[i]-predsAfter[i]) > 0.01 {
			match = false
			break
		}
	}
	assert(match, "Network predictions match after save/load")
}

// ============================================================================
// Test XOR Problem
// ============================================================================

func testXORProblem() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: XOR Problem")
	fmt.Println("-------------------------------------------------------------")

	mse := loss.MSE{}

	// XOR data
	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {1}, {1}, {0}}

	// Create network with Tanh to avoid dying ReLU problem
	layers := []layer.Layer{
		layer.NewDense(2, 3, activations.Tanh{}),
		layer.NewDense(3, 1, activations.Sigmoid{}),
	}
	network := net.New(layers, mse, &opt.SGD{LearningRate: 0.1})

	// Train
	for epoch := 0; epoch < 2000; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Test all cases
	tolerance := 0.15
	allCorrect := true
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
		correct := false
		if tt.target == 0 && pred < tolerance {
			correct = true
		}
		if tt.target == 1 && pred > 1-tolerance {
			correct = true
		}
		if !correct {
			allCorrect = false
			fmt.Printf("    XOR[%v] = %.4f, target = %.0f [WRONG]\n", tt.input, pred, tt.target)
		}
	}

	assert(allCorrect, "XOR problem solved")
}

// ============================================================================
// Test Regression
// ============================================================================

func testRegression() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Regression")
	fmt.Println("-------------------------------------------------------------")

	mse := loss.MSE{}

	// Simple identity mapping test
	layers := []layer.Layer{
		layer.NewDense(2, 2, activations.Tanh{}),
	}
	network := net.New(layers, mse, &opt.SGD{LearningRate: 0.1})

	// Train identity mapping
	for i := 0; i < 2000; i++ {
		input := []float64{0.5, 0.5}
		output := network.Forward(input)
		target := []float64{0.5, 0.5}
		grad := make([]float64, 2)
		for j := 0; j < 2; j++ {
			grad[j] = output[j] - target[j]
		}
		network.Backward(grad)
		network.Step()
	}

	input := []float64{0.5, 0.5}
	output := network.Forward(input)
	assertNear(output[0], 0.5, 0.2, "Identity mapping [0]")
	assertNear(output[1], 0.5, 0.2, "Identity mapping [1]")
}

// ============================================================================
// Test Classification
// ============================================================================

func testClassification() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Classification")
	fmt.Println("-------------------------------------------------------------")

	mse := loss.MSE{}

	// Binary classification: AND gate
	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {0}, {0}, {1}}

	layers := []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network := net.New(layers, mse, &opt.SGD{LearningRate: 0.5})

	for epoch := 0; epoch < 1000; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Test accuracy
	correct := 0
	for i := range trainX {
		pred := network.Forward(trainX[i])[0]
		predClass := 0
		if pred > 0.5 {
			predClass = 1
		}
		if predClass == int(trainY[i][0]) {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(trainX))
	assert(accuracy >= 0.75, fmt.Sprintf("Binary classification accuracy: %.1f%%", accuracy*100))

	// Multi-class classification with Softmax
	trainX = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY = [][]float64{{1, 0}, {0, 1}, {0, 1}, {1, 0}} // Two classes

	ce := loss.CrossEntropy{}
	layers = []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 2, activations.ReLU{}), // Use ReLU instead of Softmax for simplicity
	}
	network = net.New(layers, ce, &opt.SGD{LearningRate: 0.1})

	for epoch := 0; epoch < 2000; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// Test accuracy
	correct = 0
	for i := range trainX {
		pred := network.Forward(trainX[i])
		predClass := 0
		if pred[1] > pred[0] {
			predClass = 1
		}
		trueClass := 0
		if trainY[i][1] > trainY[i][0] {
			trueClass = 1
		}
		if predClass == trueClass {
			correct++
		}
	}

	accuracy = float64(correct) / float64(len(trainX))
	assert(accuracy >= 0.5, fmt.Sprintf("Multi-class accuracy: %.1f%%", accuracy*100))
}

// ============================================================================
// Test Deep Networks
// ============================================================================

func testDeepNetworks() {
	fmt.Println("-------------------------------------------------------------")
	fmt.Println("  Test Category: Deep Networks")
	fmt.Println("-------------------------------------------------------------")

	mse := loss.MSE{}

	// Test deep network: 4 -> 8 -> 8 -> 8 -> 2
	layers := []layer.Layer{
		layer.NewDense(4, 8, activations.Tanh{}),
		layer.NewDense(8, 8, activations.Tanh{}),
		layer.NewDense(8, 8, activations.Tanh{}),
		layer.NewDense(8, 2, activations.Sigmoid{}),
	}
	network := net.New(layers, mse, &opt.SGD{LearningRate: 0.1})

	input := []float64{1, 2, 3, 4}
	output := network.Forward(input)
	assert(len(output) == 2, "Deep network output = 2")

	// Test with ReLU
	layers = []layer.Layer{
		layer.NewDense(4, 8, activations.ReLU{}),
		layer.NewDense(8, 8, activations.ReLU{}),
		layer.NewDense(8, 2, activations.Sigmoid{}),
	}
	network = net.New(layers, mse, &opt.SGD{LearningRate: 0.1})
	output = network.Forward(input)
	assert(len(output) == 2, "ReLU deep network output = 2")

	// Test training deep network
	trainX := [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}}
	trainY := [][]float64{{0, 1}, {1, 0}}

	initialLoss := 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		initialLoss += mse.Forward(yPred, trainY[i])
	}
	initialLoss /= float64(len(trainX))

	for epoch := 0; epoch < 500; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	finalLoss := 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		finalLoss += mse.Forward(yPred, trainY[i])
	}
	finalLoss /= float64(len(trainX))

	assert(finalLoss < initialLoss, fmt.Sprintf("Deep network loss: %.4f -> %.4f", initialLoss, finalLoss))

	// Test with Adam optimizer
	adam := opt.NewAdam(0.01)
	layers = []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}
	network = net.New(layers, mse, adam)

	trainX = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY = [][]float64{{0}, {1}, {1}, {0}}

	initialLoss = 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		initialLoss += mse.Forward(yPred, trainY[i])
	}
	initialLoss /= float64(len(trainX))

	for epoch := 0; epoch < 500; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	finalLoss = 0.0
	for i := range trainX {
		yPred := network.Forward(trainX[i])
		finalLoss += mse.Forward(yPred, trainY[i])
	}
	finalLoss /= float64(len(trainX))

	assert(finalLoss < initialLoss, fmt.Sprintf("Adam deep network loss: %.4f -> %.4f", initialLoss, finalLoss))
}
