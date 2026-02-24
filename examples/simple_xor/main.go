package main

import (
	"fmt"
	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

func main() {
	fmt.Println("=== Simple XOR Example using High-Level API ===")

	// 1. Define XOR dataset
	inputs := [][]float32{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	targets := [][]float32{
		{0}, {1}, {1}, {0},
	}

	// 2. Create the network architecture
	model := goneuron.NewSequential(
		goneuron.Dense(4, goneuron.Tanh),
		goneuron.Dense(1, goneuron.Sigmoid),
	)

	// 3. Compile the model
	model.Compile(goneuron.Adam(0.05), goneuron.MSE)

	// 4. Print model summary
	model.Summary()

	// 5. Train the model
	fmt.Println("Training...")
	model.Fit(inputs, targets, 500, 1)

	// 6. Evaluate
	loss := model.Evaluate(inputs, targets)
	fmt.Printf("\nFinal Average Loss: %.6f\n", loss)

	// 7. Predictions
	fmt.Println("\nPredictions:")
	for _, in := range inputs {
		pred := model.Predict(in)
		fmt.Printf("Input: %v -> Prediction: %.4f\n", in, pred[0])
	}
}
