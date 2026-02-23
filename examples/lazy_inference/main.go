package main

import (
	"fmt"
	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
)

func main() {
	fmt.Println("=== Testing Lazy Shape Inference ===")

	// 1. Create a Sequential model WITHOUT ANY input sizes
	model := goneuron.NewSequential(
		goneuron.Dense(64, goneuron.ReLU),    // Auto-infers 784 from input
		goneuron.Dense(32, goneuron.ReLU),    // Automatically infers 64
		goneuron.Dense(10, goneuron.Softmax), // Automatically infers 32
	)

	// 2. Compile (won't build yet as first layer size is unknown)
	model.Compile(goneuron.Adam(0.001), goneuron.CrossEntropy)

	// 3. First Predict/Forward triggers build
	fmt.Println("\nTriggering build with dummy input...")
	dummyInput := make([]float32, 784)
	model.Predict(dummyInput)

	// 4. Print summary to verify dimensions
	model.Summary()

	// 4. Test another architecture with CNN and Flatten
	fmt.Println("\n=== Testing CNN + Flatten Lazy Inference ===")
	conv := goneuron.Conv2D(16, 3, 1, 1, goneuron.ReLU, 1)
	// For CNNs, we usually need to specify input spatial dimensions if we want exact Flatten size inference
	conv.(*layer.Conv2D).SetInputDimensions(28, 28)

	model2 := goneuron.NewSequential(
		conv,
		goneuron.BatchNorm2D(), // Auto-infers 16 channels
		goneuron.MaxPool2D(2, 2, 0),
		goneuron.Flatten(),
		goneuron.Dense(128, goneuron.ReLU),
		goneuron.Dense(10, goneuron.Softmax),
	)
	model2.Compile(goneuron.Adam(0.001), goneuron.CrossEntropy)
	model2.Summary()

	// 5. Test RNN Lazy Inference
	fmt.Println("\n=== Testing RNN Lazy Inference ===")
	model3 := goneuron.NewSequential(
		goneuron.LSTM(64, 32), // 32 is input size
		goneuron.LSTM(32),     // Auto-infers 64
		goneuron.Dense(10, goneuron.Softmax),
	)
	model3.Compile(goneuron.Adam(0.001), goneuron.CrossEntropy)
	model3.Summary()

	// 6. Test MoE Lazy Inference
	fmt.Println("\n=== Testing MoE Lazy Inference ===")
	model4 := goneuron.NewSequential(
		goneuron.Dense(128, goneuron.ReLU, 64),
		goneuron.MoE(128, 4, 2), // 128 out, 4 experts, top-2. Inferred from 128.
		goneuron.Dense(10, goneuron.Softmax),
	)
	model4.Compile(goneuron.Adam(0.001), goneuron.CrossEntropy)
	model4.Summary()

	fmt.Println("\nLazy Inference successful!")
}
