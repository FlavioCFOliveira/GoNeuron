package main

import (
	"fmt"
	"math/rand"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
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
	layers := []layer.Layer{
		layer.NewDense(8, 64, activations.ReLU{}),
		layer.NewDense(64, 32, activations.ReLU{}),
		layer.NewDense(32, 1, activations.Linear{}),
	}

	optimizer := opt.NewAdam(0.005)
	network := net.New(layers, loss.Huber{Delta: 1.0}, optimizer)

	// 3. Train
	epochs := 200
	fmt.Printf("Training for %d epochs...\n", epochs)

	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := float32(0.0)
		for i := range inputs {
			totalLoss += network.Train(inputs[i], targets[i])
		}
		if epoch%20 == 0 {
			fmt.Printf("Epoch %d - Avg Loss: %.6f\n", epoch, totalLoss/float32(numSamples))
		}
	}

	// 4. Evaluate
	fmt.Println("\n=== Sample Predictions ===")
	for i := 0; i < 5; i++ {
		output := network.Forward(inputs[i])
		fmt.Printf("Target: %8.3f, Prediction: %8.3f, Diff: %8.3f\n",
			targets[i][0], output[0], output[0]-targets[i][0])
	}
}
