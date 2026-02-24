package main

import (
	"fmt"
	"math"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
)

type ValidationResult struct {
	Backend      string
	Epochs       int
	FinalLoss    float32
	Accuracy     float32
	TotalTime    time.Duration
	TimePerEpoch time.Duration
}

func runXOR(device layer.Device, backendName string) ValidationResult {
	inputs := [][]float32{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	targets := [][]float32{
		{0}, {1}, {1}, {0},
	}

	model := goneuron.NewSequential(
		goneuron.Dense(4, goneuron.Tanh, 2),
		goneuron.Dense(1, goneuron.Sigmoid),
	)
	model.SetDevice(device)
	model.Compile(goneuron.Adam(0.05), goneuron.MSE)

	epochs := 500
	start := time.Now()
	model.Fit(inputs, targets, epochs, 1)
	duration := time.Since(start)

	// Calculate accuracy
	correct := 0
	lastLoss := float32(0)
	for i, input := range inputs {
		pred := model.Predict(input)
		lastLoss += float32(math.Pow(float64(pred[0]-targets[i][0]), 2))
		if (pred[0] > 0.5 && targets[i][0] == 1) || (pred[0] <= 0.5 && targets[i][0] == 0) {
			correct++
		}
	}
	avgLoss := lastLoss / 4

	return ValidationResult{
		Backend:      backendName,
		Epochs:       epochs,
		FinalLoss:    avgLoss,
		Accuracy:     float32(correct) / 4.0,
		TotalTime:    duration,
		TimePerEpoch: duration / time.Duration(epochs),
	}
}

func runLargeDense(device layer.Device, backendName string) ValidationResult {
	// 784 -> 512 -> 10
	inSize := 784
	hiddenSize := 512
	outSize := 10
	numSamples := 100

	// Synthetic data
	inputs := make([][]float32, numSamples)
	targets := make([][]float32, numSamples)
	rng := layer.NewRNG(42)
	for i := 0; i < numSamples; i++ {
		inputs[i] = make([]float32, inSize)
		for j := 0; j < inSize; j++ {
			inputs[i][j] = rng.RandFloat()
		}
		targets[i] = make([]float32, outSize)
		targets[i][i%outSize] = 1.0
	}

	model := goneuron.NewSequential(
		goneuron.Dense(hiddenSize, goneuron.ReLU, inSize),
		goneuron.Dense(outSize, goneuron.LogSoftmax),
	)
	model.SetDevice(device)
	model.Compile(goneuron.Adam(0.01), goneuron.NLLLoss)

	epochs := 10
	batchSize := 32
	start := time.Now()
	model.Fit(inputs, targets, epochs, batchSize)
	duration := time.Since(start)

	loss := model.Evaluate(inputs, targets)

	return ValidationResult{
		Backend:      backendName,
		Epochs:       epochs,
		FinalLoss:    loss,
		Accuracy:     0, // Not measuring accuracy for synthetic
		TotalTime:    duration,
		TimePerEpoch: duration / time.Duration(epochs),
	}
}

func main() {
	fmt.Println("=== GoNeuron Training & Performance Validation ===")

	cpuDevice := goneuron.CPUDevice()
	gpuDevice := goneuron.GetDefaultDevice()

	fmt.Println("\n--- XOR Validation ---")
	resCPU_XOR := runXOR(cpuDevice, "CPU")
	fmt.Printf("CPU - Loss: %.6f, Acc: %.2f, Time: %v\n", resCPU_XOR.FinalLoss, resCPU_XOR.Accuracy, resCPU_XOR.TotalTime)

	resGPU_XOR := runXOR(gpuDevice, "Metal")
	fmt.Printf("GPU - Loss: %.6f, Acc: %.2f, Time: %v\n", resGPU_XOR.FinalLoss, resGPU_XOR.Accuracy, resGPU_XOR.TotalTime)

	fmt.Println("\n--- Large Dense Validation (784 -> 512 -> 10) ---")
	resCPU_Large := runLargeDense(cpuDevice, "CPU")
	fmt.Printf("CPU - Loss: %.6f, Time: %v\n", resCPU_Large.FinalLoss, resCPU_Large.TotalTime)

	resGPU_Large := runLargeDense(gpuDevice, "Metal")
	fmt.Printf("GPU - Loss: %.6f, Time: %v\n", resGPU_Large.FinalLoss, resGPU_Large.TotalTime)

	fmt.Println("\n--- Summary Table ---")
	fmt.Printf("| Backend | Example | Epochs | Final Loss | Accuracy | Total Time | Time/Epoch |\n")
	fmt.Printf("|---------|---------|--------|------------|----------|------------|------------|\n")
	fmt.Printf("| %-7s | XOR     | %-6d | %-10.6f | %-8.2f | %-10v | %-10v |\n",
		resCPU_XOR.Backend, resCPU_XOR.Epochs, resCPU_XOR.FinalLoss, resCPU_XOR.Accuracy, resCPU_XOR.TotalTime, resCPU_XOR.TimePerEpoch)
	fmt.Printf("| %-7s | XOR     | %-6d | %-10.6f | %-8.2f | %-10v | %-10v |\n",
		resGPU_XOR.Backend, resGPU_XOR.Epochs, resGPU_XOR.FinalLoss, resGPU_XOR.Accuracy, resGPU_XOR.TotalTime, resGPU_XOR.TimePerEpoch)
	fmt.Printf("| %-7s | Large   | %-6d | %-10.6f | %-8s | %-10v | %-10v |\n",
		resCPU_Large.Backend, resCPU_Large.Epochs, resCPU_Large.FinalLoss, "N/A", resCPU_Large.TotalTime, resCPU_Large.TimePerEpoch)
	fmt.Printf("| %-7s | Large   | %-6d | %-10.6f | %-8s | %-10v | %-10v |\n",
		resGPU_Large.Backend, resGPU_Large.Epochs, resGPU_Large.FinalLoss, "N/A", resGPU_Large.TotalTime, resGPU_Large.TimePerEpoch)
}
