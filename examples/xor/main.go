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

func main() {
	fmt.Println("=== XOR Neural Network Example ===")
	fmt.Println("Training a 2-4-1 network to solve the XOR problem.")

	// 1. Define XOR dataset
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// 2. Create the network architecture
	// Hidden layer: 2 inputs -> 4 outputs (Tanh)
	// Output layer: 4 inputs -> 1 output (Sigmoid)
	layers := []layer.Layer{
		layer.NewDense(2, 4, activations.Tanh{}),
		layer.NewDense(4, 1, activations.Sigmoid{}),
	}

	// 3. Initialize the network with MSE loss and Adam optimizer
	learningRate := 0.05
	optimizer := opt.NewAdam(learningRate)
	network := net.New(layers, loss.MSE{}, optimizer)

	// 4. Training loop
	epochs := 1000
	startTime := time.Now()

	fmt.Printf("\nTraining for %d epochs...\n", epochs)

	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := 0.0
		for i := range inputs {
			lossVal := network.Train(inputs[i], targets[i])
			totalLoss += lossVal
		}

		if epoch == 1 || epoch%100 == 0 || epoch == epochs {
			avgLoss := totalLoss / float64(len(inputs))
			fmt.Printf("Epoch %4d/%d - Avg Loss: %.8f\n", epoch, epochs, avgLoss)
		}
	}

	duration := time.Since(startTime)
	fmt.Printf("\nTraining completed in %v\n", duration)

	// 5. Evaluate the trained network
	fmt.Println("\n=== Final Predictions ===")
	fmt.Printf("%-10s | %-10s | %-10s | %-10s\n", "Input A", "Input B", "Target", "Prediction")
	fmt.Println("-----------|------------|------------|-----------")

	tolerance := 0.1
	success := true

	for i, input := range inputs {
		output := network.Forward(input)
		prediction := output[0]
		target := targets[i][0]

		fmt.Printf("%-10.1f | %-10.1f | %-10.1f | %-10.6f\n",
			input[0], input[1], target, prediction)

		// Simple check for correctness
		if math.Abs(prediction-target) > tolerance {
			success = false
		}
	}

	if success {
		fmt.Println("\nSuccess! The network learned the XOR function correctly.")
	} else {
		fmt.Println("\nThe network failed to learn the XOR function. Try increasing epochs or changing the learning rate.")
	}

	fmt.Println("\n=== Example Complete ===")
}
