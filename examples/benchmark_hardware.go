package main

import (
	"fmt"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

func runBenchmark(device layer.Device, name string, inSize, outSize, batchSize int) {
	fmt.Printf("\n--- Benchmark: %s (%v) ---\n", name, device.Type())
	if !device.IsAvailable() {
		fmt.Println("Device not available, skipping.")
		return
	}

	// Create a reasonably large network to stress the device
	layers := []layer.Layer{
		layer.NewDenseWithDevice(inSize, outSize, activations.Tanh{}, device),
		layer.NewDenseWithDevice(outSize, outSize, activations.Tanh{}, device),
		layer.NewDenseWithDevice(outSize, 10, activations.Sigmoid{}, device),
	}
	network := net.New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.01})

	// Generate random data
	x := make([][]float64, batchSize)
	y := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		x[i] = make([]float64, inSize)
		y[i] = []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}
		for j := 0; j < inSize; j++ {
			x[i][j] = 0.5
		}
	}

	// Warm-up
	network.TrainBatch(x[:1], y[:1])

	// Benchmark
	start := time.Now()
	iterations := 10
	for i := 0; i < iterations; i++ {
		network.TrainBatch(x, y)
	}
	duration := time.Since(start)

	fmt.Printf("Batch Size: %d | Input: %d | Hidden: %d\n", batchSize, inSize, outSize)
	fmt.Printf("Total Time (%d iterations): %v\n", iterations, duration)
	fmt.Printf("Avg Time per batch: %v\n", duration/time.Duration(iterations))
}

func runLSTMBenchmark(device layer.Device, name string, inSize, outSize, seqLen int) {
	fmt.Printf("\n--- Benchmark LSTM: %s (%v) ---\n", name, device.Type())
	if !device.IsAvailable() {
		fmt.Println("Device not available, skipping.")
		return
	}

	lstm := layer.NewLSTMWithDevice(inSize, outSize, device)
	unroller := layer.NewSequenceUnroller(lstm, seqLen, false)

	layers := []layer.Layer{
		unroller,
		layer.NewDenseWithDevice(outSize, 10, activations.Sigmoid{}, device),
	}
	network := net.New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.01})

	x := make([]float64, inSize * seqLen)
	y := []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}

	// Warm-up
	network.Train(x, y)

	// Benchmark
	start := time.Now()
	iterations := 50
	for i := 0; i < iterations; i++ {
		network.Train(x, y)
	}
	duration := time.Since(start)

	fmt.Printf("Sequence Length: %d | Input: %d | Hidden: %d\n", seqLen, inSize, outSize)
	fmt.Printf("Total Time (%d iterations): %v\n", iterations, duration)
	fmt.Printf("Avg Time per sample: %v\n", duration/time.Duration(iterations))
}

func main() {
	fmt.Println("========================================")
	fmt.Println(" GoNeuron Hardware Acceleration Benchmark")
	fmt.Println("========================================")

	// Test configurations
	inSize := 784
	outSize := 1024
	batchSize := 64

	// CPU Benchmark
	runBenchmark(&layer.CPUDevice{}, "Dense Network", inSize, outSize, batchSize)

	// GPU/Metal Benchmark
	gpu := layer.NewMetalDevice()
	if gpu.IsAvailable() {
		runBenchmark(gpu, "Dense Network", inSize, outSize, batchSize)
	} else {
		fmt.Println("\nMetal GPU not available on this system.")
	}

	// LSTM Benchmark
	seqLen := 100
	lstmHidden := 256

	runLSTMBenchmark(&layer.CPUDevice{}, "LSTM Sequence", inSize, lstmHidden, seqLen)
	if gpu.IsAvailable() {
		runLSTMBenchmark(gpu, "LSTM Sequence", inSize, lstmHidden, seqLen)
	}
}
