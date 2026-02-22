package main

import (
	"fmt"
	"math"

	"github.com/FlavioCFOliveira/GoNeuron/examples/utils"
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

func main() {
	fmt.Println("=== RBF Network for Sine Wave Approximation ===")

	// 1. Generate data
	trainInputs, trainTargets := utils.GenerateSineWave(1000)
	testInputs, testTargets := utils.GenerateSineWave(100)

	// 2. Create architecture
	// RBF layer with 1 input, 50 centers, 50 outputs, gamma=1.0
	// Dense layer with 50 inputs, 1 output, Linear activation
	layers := []layer.Layer{
		layer.NewRBF(1, 50, 50, 1.0),
		layer.NewDense(50, 1, activations.Linear{}),
	}

	optimizer := opt.NewAdam(0.01)
	network := net.New(layers, loss.MSE{}, optimizer)

	// 3. Train
	epochs := 500
	fmt.Printf("Training for %d epochs...\n", epochs)

	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := 0.0
		for i := range trainInputs {
			totalLoss += network.Train(trainInputs[i], trainTargets[i])
		}
		if epoch%50 == 0 {
			fmt.Printf("Epoch %d - Avg Loss: %.6f\n", epoch, totalLoss/float64(len(trainInputs)))
		}
	}

	// 4. Evaluate
	fmt.Println("\n=== Sample Predictions ===")
	totalMSE := 0.0
	for i := range testInputs {
		output := network.Forward(testInputs[i])
		target := testTargets[i][0]
		diff := output[0] - target
		totalMSE += diff * diff
		if i < 10 {
			fmt.Printf("x: %6.3f, Target: %6.3f, Pred: %6.3f, Diff: %6.3f\n",
				testInputs[i][0], target, output[0], diff)
		}
	}

	fmt.Printf("\nTest MSE: %.6f\n", totalMSE/float64(len(testInputs)))
}
