package main

import (
	"fmt"
	"math"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

func main() {
	fmt.Println("=== XOR Neural Network Example (High-Level API) ===")
	fmt.Println("Training a 2-4-1 network to solve the XOR problem.")

	// 1. Define XOR dataset
	inputs := [][]float32{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	targets := [][]float32{
		{0}, {1}, {1}, {0},
	}

	// 2. Create the network architecture using Sequential API
	model := goneuron.NewSequential(
		goneuron.Dense(2, 4, goneuron.Tanh),
		goneuron.Dense(4, 1, goneuron.Sigmoid),
	)

	// 3. Compile the model
	model.Compile(goneuron.Adam(0.05), goneuron.MSE)

	// 4. Training
	fmt.Println("\nTraining for 1000 epochs...")
	startTime := time.Now()

	model.Fit(inputs, targets, 1000, 1, goneuron.Logger(100))

	duration := time.Since(startTime)
	fmt.Printf("\nTraining completed in %v\n", duration)

	// 5. Evaluate and Predict
	fmt.Println("\n=== Final Predictions ===")
	fmt.Printf("%-10s | %-10s | %-10s | %-10s\n", "Input A", "Input B", "Target", "Prediction")
	fmt.Println("-----------|------------|------------|-----------")

	success := true
	tolerance := float32(0.1)

	for i, input := range inputs {
		prediction := model.Predict(input)[0]
		target := targets[i][0]

		fmt.Printf("%-10.1f | %-10.1f | %-10.1f | %-10.6f\n",
			input[0], input[1], target, prediction)

		if float32(math.Abs(float64(prediction-target))) > tolerance {
			success = false
		}
	}

	if success {
		fmt.Println("\nSuccess! The network learned the XOR function correctly.")
	} else {
		fmt.Println("\nThe network failed to learn the XOR function.")
	}
}
