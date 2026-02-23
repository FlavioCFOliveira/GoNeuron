package main

import (
	"fmt"
	"math/rand"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

func main() {
	fmt.Println("=== Deep FNN for Synthetic Housing Regression ===")

	// 1. Generate synthetic data (8 features)
	numSamples := 1000
	dim := 8
	inputs := make([][]float32, numSamples)
	targets := make([][]float32, numSamples)

	// Simple linear combination with some non-linearity and noise
	for i := 0; i < numSamples; i++ {
		inputs[i] = make([]float32, dim)
		sum := float32(0.0)
		for j := 0; j < dim; j++ {
			inputs[i][j] = rand.Float32()*2 - 1
			sum += inputs[i][j] * float32(j+1)
		}
		// Target = sum + sum^2/10 + noise
		targets[i] = []float32{sum + (sum*sum)/10.0 + float32(rand.NormFloat64()*0.1)}
	}

	// 2. Create architecture (8 -> 64 -> 32 -> 1)
	model := goneuron.NewSequential(
		goneuron.Dense(8, 64, goneuron.ReLU),
		goneuron.Dense(64, 32, goneuron.ReLU),
		goneuron.Dense(32, 1, goneuron.Linear),
	)

	model.Compile(goneuron.Adam(0.005), goneuron.Huber(1.0))

	// 3. Train
	epochs := 200
	fmt.Printf("Training for %d epochs...\n", epochs)

	model.Fit(inputs, targets, epochs, 32, goneuron.Logger(20))

	// 4. Evaluate
	fmt.Println("\n=== Sample Predictions ===")
	for i := 0; i < 5; i++ {
		output := model.Forward(inputs[i])
		fmt.Printf("Target: %8.3f, Prediction: %8.3f, Diff: %8.3f\n",
			targets[i][0], output[0], output[0]-targets[i][0])
	}
}
