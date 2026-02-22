package main

import (
	"fmt"

	"github.com/FlavioCFOliveira/GoNeuron/examples/utils"
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

func main() {
	fmt.Println("=== SLP for Iris Classification ===")

	// 1. Load and prepare data
	data, err := utils.LoadIris()
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		return
	}
	data.Normalize()
	data.Shuffle()

	// Split data (80% train, 20% test)
	trainSize := int(float32(len(data.Inputs)) * 0.8)
	trainInputs := data.Inputs[:trainSize]
	trainTargets := data.Targets[:trainSize]
	testInputs := data.Inputs[trainSize:]
	testTargets := data.Targets[trainSize:]

	// 2. Create architecture (4 inputs -> 3 outputs with Softmax)
	layers := []layer.Layer{
		layer.NewDense(4, 3, activations.Softmax{}),
	}

	optimizer := opt.NewAdam(0.01)
	network := net.New(layers, loss.CrossEntropy{}, optimizer)

	// 3. Train
	epochs := 200
	fmt.Printf("Training for %d epochs...\n", epochs)

	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := float32(0.0)
		for i := range trainInputs {
			totalLoss += network.Train(trainInputs[i], trainTargets[i])
		}
		if epoch%20 == 0 {
			fmt.Printf("Epoch %d - Avg Loss: %.6f\n", epoch, totalLoss/float32(trainSize))
		}
	}

	// 4. Evaluate
	correct := 0
	for i := range testInputs {
		output := network.Forward(testInputs[i])
		if argmax(output) == argmax(testTargets[i]) {
			correct++
		}
	}

	accuracy := float32(correct) / float32(len(testInputs)) * 100
	fmt.Printf("\nFinal Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(testInputs))
}

func argmax(v []float32) int {
	maxIdx := 0
	maxVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > maxVal {
			maxVal = v[i]
			maxIdx = i
		}
	}
	return maxIdx
}
