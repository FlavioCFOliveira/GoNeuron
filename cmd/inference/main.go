// Package main - Examples of neural network inference for GoNeuron
// Execute com: go run cmd/inference/main.go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

func main() {
	fmt.Println("=============================================================")
	fmt.Println("  GoNeuron - Neural Network Inference Examples")
	fmt.Println("=============================================================")
	fmt.Println()

	// Seed rand for reproducibility
	rand.Seed(42)

	// Example 1: Dense Network Inference
	fmt.Println("--- Example 1: Dense Network (MLP) ---")
	denseNetworkInference()
	fmt.Println()

	// Example 2: Convolutional Network Inference
	fmt.Println("--- Example 2: Convolutional Network (CNN) ---")
	cnnNetworkInference()
	fmt.Println()

	// Example 3: LSTM Network Inference
	fmt.Println("--- Example 3: Recurrent Network (LSTM) ---")
	lstmNetworkInference()
	fmt.Println()

	// Example 4: Hybrid Network (CNN + Dense)
	fmt.Println("--- Example 4: Hybrid Network (CNN + Dense) ---")
	hybridNetworkInference()
	fmt.Println()

	// Example 5: Network with Different Activations
	fmt.Println("--- Example 5: Network with Multiple Activation Functions ---")
	multiActivationNetworkInference()
	fmt.Println()

	// Example 6: Batch Inference
	fmt.Println("--- Example 6: Batch Inference ---")
	batchInference()
	fmt.Println()
}

// ============================================================================
// Example 1: Dense Network Inference
// ============================================================================

func denseNetworkInference() {
	fmt.Println("Architecture: 4 -> 8 -> 6 -> 3")
	fmt.Println("Task: Classification (3 classes)")
	fmt.Println()

	// Create a trained network (in production, load from file)
	// For demo, we'll create and train a small network first
	layers := []layer.Layer{
		layer.NewDense(4, 8, activations.Tanh{}),
		layer.NewDense(8, 6, activations.Tanh{}),
		layer.NewDense(6, 3, activations.Sigmoid{}), // Using Sigmoid instead of Softmax for compatibility
	}
	ce := loss.CrossEntropy{}
	network := net.New(layers, ce, opt.NewAdam(0.001))

	// Simulate training (in real use: load from file with net.Load())
	trainX := [][]float64{
		{5.1, 3.5, 1.4, 0.2},
		{6.2, 2.9, 4.3, 1.3},
		{7.2, 3.0, 5.8, 1.6},
	}
	trainY := [][]float64{
		{0.9, 0.1, 0.1}, // Class 0
		{0.1, 0.9, 0.1}, // Class 1
		{0.1, 0.1, 0.9}, // Class 2
	}

	for i := 0; i < 100; i++ {
		for j := range trainX {
			network.Train(trainX[j], trainY[j])
		}
	}

	// INFERENCE: Make predictions on new data
	newSamples := [][]float64{
		{5.0, 3.4, 1.5, 0.2},  // Should be class 0
		{6.0, 2.8, 4.0, 1.2},  // Should be class 1
		{7.0, 3.1, 5.5, 1.5},  // Should be class 2
	}

	fmt.Println("Input Features -> Predicted Class (Probabilities)")
	fmt.Println("---------------------------------------------------")

	for i, sample := range newSamples {
		// Single sample inference
		output := network.Forward(sample)

		// Get predicted class (argmax)
		predClass := 0
		maxProb := output[0]
		for j := 1; j < len(output); j++ {
			if output[j] > maxProb {
				maxProb = output[j]
				predClass = j
			}
		}

		fmt.Printf("Sample %d: [", i)
		for j, v := range sample {
			if j > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%.1f", v)
		}
		fmt.Printf("] -> Class %d (", predClass)
		for j, p := range output {
			if j > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%.3f", p)
		}
		fmt.Printf(")\n")
	}
}

// ============================================================================
// Example 2: Convolutional Network Inference
// ============================================================================

func cnnNetworkInference() {
	fmt.Println("Architecture: Conv2D(1->4, 3x3) + ReLU -> Dense(16->1)")
	fmt.Println("Task: Binary Classification (2D images)")
	fmt.Println()

	// Create CNN: 8x8 images -> Conv2D -> Dense
	// After Conv2D with 3x3 kernel, stride=1, padding=1: 1x8x8 -> 4x8x8 = 256 values
	conv := layer.NewConv2D(1, 4, 3, 1, 1, activations.ReLU{})
	dense := layer.NewDense(256, 1, activations.Sigmoid{})

	network := net.New([]layer.Layer{conv, dense}, loss.MSE{}, opt.NewAdam(0.01))

	// Simulate training on 8x8 binary images
	// Class 0: low values, Class 1: high values in top-left quadrant
	trainX := make([][]float64, 20)
	trainY := make([][]float64, 20)

	for i := 0; i < 20; i++ {
		img := make([]float64, 64)
		class := i % 2

		for j := range img {
			row := j / 8
			col := j % 8
			base := (rand.Float64() - 0.5) * 0.2
			if class == 1 && row < 4 && col < 4 {
				base += 0.5
			}
			img[j] = base
		}
		trainX[i] = img
		trainY[i] = []float64{float64(class)}
	}

	for epoch := 0; epoch < 50; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// INFERENCE: Make predictions on new images
	fmt.Println("8x8 Image (top-left quadrant indicates class)")
	fmt.Println("Class 0: uniform noise, Class 1: brighter top-left")
	fmt.Println()

	for i := 0; i < 3; i++ {
		// Create test image
		img := make([]float64, 64)
		class := i % 2
		for j := range img {
			row := j / 8
			col := j % 8
			base := (rand.Float64() - 0.5) * 0.2
			if class == 1 && row < 4 && col < 4 {
				base += 0.5
			}
			img[j] = base
		}

		output := network.Forward(img)
		predClass := 0
		if output[0] > 0.5 {
			predClass = 1
		}

		fmt.Printf("Image %d: Predicted class %d (prob=%.4f)\n", i, predClass, output[0])
	}
}

// ============================================================================
// Example 3: LSTM Network Inference
// ============================================================================

func lstmNetworkInference() {
	fmt.Println("Architecture: LSTM(1->8) -> Dense(8->1)")
	fmt.Println("Task: Sequential Prediction")
	fmt.Println()

	// Create LSTM network for sequential data
	lstm := layer.NewLSTM(1, 8)
	dense := layer.NewDense(8, 1, activations.Sigmoid{})

	network := net.New([]layer.Layer{lstm, dense}, loss.MSE{}, opt.NewAdam(0.05))

	// Simulate training on sine wave
	trainX := make([][]float64, 10)
	trainY := make([][]float64, 10)

	for i := 0; i < 10; i++ {
		seqLen := 20
		xSeq := make([]float64, seqLen)
		ySeq := make([]float64, seqLen)

		for t := 0; t < seqLen; t++ {
			xSeq[t] = float64(t) / float64(seqLen-1) * 2 * math.Pi
			ySeq[t] = math.Sin(xSeq[t])/2 + 0.5
		}
		trainX[i] = xSeq
		trainY[i] = ySeq
	}

	// Train on sequences
	for epoch := 0; epoch < 100; epoch++ {
		totalLoss := 0.0
		count := 0

		for i := range trainX {
			lstm.Reset()
			for t := 0; t < len(trainX[i]); t++ {
				lstm.Forward([]float64{trainX[i][t]})
				if t > 0 {
					loss := network.Train([]float64{0}, []float64{trainY[i][t]})
					totalLoss += loss
					count++
				}
			}
		}

		if epoch%20 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(count))
		}
	}

	// INFERENCE: Generate predictions from a seed sequence
	fmt.Println("\nGenerating predictions from seed sequence...")
	fmt.Println("(Each input is a scalar, LSTM maintains state between steps)")
	fmt.Println()

	seedSeq := []float64{0, 0.5, 1, 1.5, 2}
	targetSeq := make([]float64, len(seedSeq))
	for i, x := range seedSeq {
		targetSeq[i] = math.Sin(x*2*math.Pi)/2 + 0.5
	}

	lstm.Reset()
	fmt.Println("Input (x) -> Predicted y (Target y)")
	fmt.Println("-----------------------------------")

	for i := 0; i < 10; i++ {
		if i < len(seedSeq) {
			// Use seed sequence
			out := lstm.Forward([]float64{seedSeq[i]})
			pred := dense.Forward(out)
			fmt.Printf("%.1f -> %.4f (%.4f)\n", seedSeq[i], pred[0], targetSeq[i])
		} else {
			// Continue prediction from last output
			lastOutput := targetSeq[len(targetSeq)-1]
			out := lstm.Forward([]float64{lastOutput})
			pred := dense.Forward(out)
			fmt.Printf("%.1f -> %.4f (extrapolated)\n", float64(len(seedSeq)+i-1), pred[0])
		}
	}
}

// ============================================================================
// Example 4: Hybrid Network (CNN + Dense)
// ============================================================================

func hybridNetworkInference() {
	fmt.Println("Architecture: Conv2D(1->4, 3x3) + ReLU -> Flatten -> Dense(144->16) + Tanh -> Dense(16->1)")
	fmt.Println("Task: Image Classification with Convolution + MLP")
	fmt.Println()

	// Create hybrid network
	conv := layer.NewConv2D(1, 4, 3, 1, 0, activations.ReLU{})
	// After conv2d 8x8 input, 3x3 kernel, no padding, stride=1: 4 channels of 6x6 = 144
	dense1 := layer.NewDense(144, 16, activations.Tanh{})
	dense2 := layer.NewDense(16, 1, activations.Sigmoid{})

	network := net.New([]layer.Layer{conv, dense1, dense2}, loss.MSE{}, opt.NewAdam(0.01))

	// Simulate training
	trainX := make([][]float64, 20)
	trainY := make([][]float64, 20)

	for i := 0; i < 20; i++ {
		img := make([]float64, 64)
		class := i % 2

		for j := range img {
			row := j / 8
			col := j % 8
			base := (rand.Float64() - 0.5) * 0.2
			if class == 1 && row < 4 && col < 4 {
				base += 0.5
			}
			img[j] = base
		}
		trainX[i] = img
		trainY[i] = []float64{float64(class)}
	}

	for epoch := 0; epoch < 50; epoch++ {
		for i := range trainX {
			network.Train(trainX[i], trainY[i])
		}
	}

	// INFERENCE
	fmt.Println("8x8 Image Classification")
	fmt.Println("------------------------")

	for i := 0; i < 3; i++ {
		img := make([]float64, 64)
		class := i % 2
		for j := range img {
			row := j / 8
			col := j % 8
			base := (rand.Float64() - 0.5) * 0.2
			if class == 1 && row < 4 && col < 4 {
				base += 0.5
			}
			img[j] = base
		}

		output := network.Forward(img)
		predClass := 0
		if output[0] > 0.5 {
			predClass = 1
		}

		fmt.Printf("Image %d: Class %d (prob=%.4f)\n", i, predClass, output[0])
	}
}

// ============================================================================
// Example 5: Network with Multiple Activation Functions
// ============================================================================

func multiActivationNetworkInference() {
	fmt.Println("Demonstrating different activation functions in inference")
	fmt.Println()

	// Sample input
	input := []float64{-2.0, -1.0, 0.0, 1.0, 2.0}

	// Test each activation
	activationTests := []struct {
		name string
		impl activations.Activation
	}{
		{"ReLU", activations.ReLU{}},
		{"Sigmoid", activations.Sigmoid{}},
		{"Tanh", activations.Tanh{}},
		{"LeakyReLU(α=0.01)", activations.NewLeakyReLU(0.01)},
	}

	fmt.Println("Activation Functions Comparison:")
	fmt.Println("Input values: [-2, -1, 0, 1, 2]")
	fmt.Println()

	for _, a := range activationTests {
		fmt.Printf("%s:\n", a.name)
		fmt.Print("  Output: [")
		for i, v := range input {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%.4f", a.impl.Activate(v))
		}
		fmt.Println("]")
	}

	// Use in a network
	fmt.Println("Network with mixed activations:")
	fmt.Println("  Input(5) -> Dense(5->3, ReLU) -> Dense(3->2, Sigmoid) -> Dense(2->1, Tanh)")
	fmt.Println()

	inputNet := []float64{1.0, -0.5, 2.0, 0.0, -1.0}

	l1 := layer.NewDense(5, 3, activations.ReLU{})
	l2 := layer.NewDense(3, 2, activations.Sigmoid{})
	l3 := layer.NewDense(2, 1, activations.Tanh{})

	// Forward through each layer manually
	h1 := l1.Forward(inputNet)
	h2 := l2.Forward(h1)
	output := l3.Forward(h2)

	fmt.Printf("Input:  %.4f, %.4f, %.4f, %.4f, %.4f\n", inputNet[0], inputNet[1], inputNet[2], inputNet[3], inputNet[4])
	fmt.Printf("After ReLU:     %.4f, %.4f, %.4f\n", h1[0], h1[1], h1[2])
	fmt.Printf("After Sigmoid:  %.4f, %.4f\n", h2[0], h2[1])
	fmt.Printf("Final Output:   %.4f\n", output[0])
}

// ============================================================================
// Example 6: Batch Inference
// ============================================================================

func batchInference() {
	fmt.Println("Performing inference on multiple samples efficiently")
	fmt.Println()

	// Create a simple network
	layers := []layer.Layer{
		layer.NewDense(4, 8, activations.Tanh{}),
		layer.NewDense(8, 3, activations.Sigmoid{}), // Using Sigmoid instead of Softmax
	}
	ce := loss.CrossEntropy{}
	network := net.New(layers, ce, opt.NewAdam(0.001))

	// Simulate training
	trainX := [][]float64{
		{5.1, 3.5, 1.4, 0.2},
		{6.2, 2.9, 4.3, 1.3},
		{7.2, 3.0, 5.8, 1.6},
	}
	trainY := [][]float64{
		{0.9, 0.1, 0.1},
		{0.1, 0.9, 0.1},
		{0.1, 0.1, 0.9},
	}

	for i := 0; i < 100; i++ {
		for j := range trainX {
			network.Train(trainX[j], trainY[j])
		}
	}

	// Single inference (slower for many samples)
	fmt.Println("Single sample inference (loop):")
	fmt.Println("-------------------------------")
	samples := [][]float64{
		{5.0, 3.4, 1.5, 0.2},
		{6.0, 2.8, 4.0, 1.2},
		{7.0, 3.1, 5.5, 1.5},
	}

	for i, sample := range samples {
		output := network.Forward(sample)
		predClass := 0
		maxProb := output[0]
		for j := 1; j < len(output); j++ {
			if output[j] > maxProb {
				maxProb = output[j]
				predClass = j
			}
		}
		fmt.Printf("Sample %d: Class %d (max prob=%.3f)\n", i, predClass, maxProb)
	}

	// Note: GoNeuron's TrainBatch uses parallel processing
	// For inference, you'd typically batch in your application code
	// or use the network's internal optimization for batch sizes >= 16
	fmt.Println()
	fmt.Println("Note: For efficient batch inference, consider:")
	fmt.Println("  1. Processing samples in your application with a batch size >= 16")
	fmt.Println("  2. Using Go's goroutines for concurrent inference on multiple batches")
	fmt.Println("  3. The network automatically parallelizes batch training (>= 16 samples)")
}
