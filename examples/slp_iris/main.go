package main

import (
	"fmt"

	"github.com/FlavioCFOliveira/GoNeuron/examples/utils"
	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

func main() {
	fmt.Println("=== SLP for Iris Classification (High-Level API) ===")

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
	model := goneuron.NewSequential(
		goneuron.Dense(4, 3, goneuron.Softmax),
	)

	// 3. Compile
	model.Compile(goneuron.Adam(0.01), goneuron.CrossEntropy)

	// 4. Train
	fmt.Println("Training...")
	model.Fit(trainInputs, trainTargets, 200, 1, goneuron.Logger(20))

	// 5. Evaluate
	correct := 0
	for i := range testInputs {
		output := model.Predict(testInputs[i])
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
