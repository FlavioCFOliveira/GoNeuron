package main

import (
	"fmt"
	"math"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

func main() {
	fmt.Println("=== XOR Training Example ===")

	// Create a simple XOR network: 2 inputs -> 3 hidden -> 1 output
	// The XOR function cannot be solved by a single-layer perceptron
	// but can be solved by a multi-layer perceptron with hidden layers
	in := 2
	hidden := 3
	out := 1

	fmt.Printf("Network architecture: %d-%d-%d\n", in, hidden, out)
	fmt.Println("Activation functions: ReLU (hidden), Sigmoid (output)")
	fmt.Println("Loss function: MSE")
	fmt.Println("Optimizer: SGD with learning rate 0.1")

	// Create layers with deterministic initialization (seed=42)
	l1 := layer.NewDense(in, hidden, activations.ReLU{})
	l2 := layer.NewDense(hidden, out, activations.Sigmoid{})

	network := net.New(
		[]layer.Layer{l1, l2},
		loss.MSE{},
		opt.SGD{LearningRate: 0.1},
	)

	// XOR training data
	trainX := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	trainY := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Train for 5000 epochs
	for epoch := 0; epoch < 5000; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%500 == 0 {
			fmt.Printf("Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(len(trainX)))
		}
	}

	// Test the network
	fmt.Println("\nTesting trained network:")
	for i := range trainX {
		pred := network.Forward(trainX[i])
		fmt.Printf("Input: %v, Predicted: %.4f, Target: %v\n",
			trainX[i], pred[0], trainY[i][0])
	}

	// Save the trained network
	fmt.Println("\nSaving network to disk...")
	if err := network.Save("xor_network.bin"); err != nil {
		fmt.Printf("Error saving network: %v\n", err)
		return
	}
	fmt.Println("Network saved successfully!")

	// Load the network back
	fmt.Println("Loading network from disk...")
	loadedNetwork, _, err := net.Load("xor_network.bin")
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		return
	}
	fmt.Println("Network loaded successfully!")

	// Verify loaded network produces same predictions
	fmt.Println("\nVerifying loaded network:")
	allMatch := true
	for i := range trainX {
		originalPred := network.Forward(trainX[i])
		loadedPred := loadedNetwork.Forward(trainX[i])
		match := "OK"
		if math.Abs(originalPred[0]-loadedPred[0]) > 1e-6 {
			match = "MISMATCH"
			allMatch = false
		}
		fmt.Printf("Input: %v, Original: %.4f, Loaded: %.4f [%s]\n",
			trainX[i], originalPred[0], loadedPred[0], match)
	}

	if allMatch {
		fmt.Println("\nSUCCESS: All predictions match between original and loaded network!")
	} else {
		fmt.Println("\nFAILURE: Predictions differ between original and loaded network!")
	}
}
