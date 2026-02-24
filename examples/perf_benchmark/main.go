package main

import (
	"fmt"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

func main() {
	fmt.Println("=== GPU vs CPU Batch Performance Benchmark ===")

	// 1. Setup a representative CNN architecture
	// (similar to MNIST but smaller to run faster)
	inChannels := 1
	outChannels1 := 8
	outChannels2 := 16
	imgSize := 28
	batchSize := 64
	epochs := 1
	numSamples := 128 // Small dataset for quick benchmark

	createModel := func() *goneuron.Model {
		m := goneuron.NewSequential(
			goneuron.Conv2D(outChannels1, 3, 1, 1, goneuron.ReLU),
			goneuron.MaxPool2D(2, 2, 0),
			goneuron.Conv2D(outChannels2, 3, 1, 1, goneuron.ReLU),
			goneuron.MaxPool2D(2, 2, 0),
			goneuron.Flatten(),
			goneuron.Dense(64, goneuron.ReLU),
			goneuron.Dense(10, goneuron.LogSoftmax),
		)
		// Set input dimensions for the first layer
		if l, ok := m.Layers()[0].(interface{ SetInputDimensions(int, int) }); ok {
			l.SetInputDimensions(imgSize, imgSize)
		}
		m.Build(inChannels)
		return m
	}

	// 2. Generate dummy data
	x := make([][]float32, numSamples)
	y := make([][]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		x[i] = make([]float32, inChannels*imgSize*imgSize)
		y[i] = make([]float32, 10)
		y[i][i%10] = 1.0
	}

	// Benchmark CPU
	fmt.Println("\nRunning CPU Benchmark...")
	cpuModel := createModel()
	cpuModel.SetDevice(goneuron.CPUDevice())
	cpuModel.Compile(goneuron.Adam(0.001), goneuron.NLLLoss)

	start := time.Now()
	cpuModel.Fit(x, y, epochs, batchSize)
	cpuDuration := time.Since(start)
	fmt.Printf("CPU Time: %v\n", cpuDuration)

	// Benchmark GPU
	fmt.Println("\nRunning GPU Benchmark...")
	gpuModel := createModel()
	// GPU is default on Darwin, but let's be explicit if we could
	// For this library, NewMetalDevice is internal/layer.
	// We'll trust default if on Darwin arm64.

	gpuModel.Compile(goneuron.Adam(0.001), goneuron.NLLLoss)

	start = time.Now()
	gpuModel.Fit(x, y, epochs, batchSize)
	gpuDuration := time.Since(start)
	fmt.Printf("GPU Time: %v\n", gpuDuration)

	// Results
	speedup := float64(cpuDuration) / float64(gpuDuration)
	fmt.Printf("\nSummary:\n")
	fmt.Printf("CPU: %v\n", cpuDuration)
	fmt.Printf("GPU: %v\n", gpuDuration)
	fmt.Printf("Speedup: %.2fx\n", speedup)

	if speedup < 1.1 {
		fmt.Println("Warning: GPU speedup is low. Small model/batch size might be dominated by overhead.")
	}
}
