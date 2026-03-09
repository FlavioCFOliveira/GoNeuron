// Package net provides benchmarks for neural network training.
package net

import (
	"math/rand"
	"testing"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// fillRandom fills a slice with random values.
func fillRandom(slice []float32) {
	for i := range slice {
		slice[i] = rand.Float32()
	}
}

// mustDense creates a Dense layer or panics on error
func mustDense(in, out int, act activations.Activation) layer.Layer {
	l, err := layer.NewDense(in, out, act)
	if err != nil {
		panic(err)
	}
	return l
}

// mustDense creates a Dense layer or panics on error (for benchmarks only).
func mustDense(in, out int, act activations.Activation) layer.Layer {
	l, err := layer.NewDense(in, out, act)
	if err != nil {
		panic(err)
	}
	return l
}

// BenchmarkNetworkForward benchmarks a forward pass through a small network.
func BenchmarkNetworkForward(b *testing.B) {
	layers := []layer.Layer{
		mustDense(784, 256, activations.Tanh{}),
		mustDense(256, 128, activations.Tanh{}),
		mustDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := make([]float32, 784)
	fillRandom(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = network.Forward(input)
	}
}

// BenchmarkNetworkBackward benchmarks a backward pass through a small network.
func BenchmarkNetworkBackward(b *testing.B) {
	layers := []layer.Layer{
		mustDense(784, 256, activations.Tanh{}),
		mustDense(256, 128, activations.Tanh{}),
		mustDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := make([]float32, 784)
	grad := make([]float32, 10)
	fillRandom(input)
	fillRandom(grad)

	// Forward pass to set up state
	_, _ = network.Forward(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = network.Backward(grad)
	}
}

// BenchmarkNetworkTrain benchmarks training on a single sample.
func BenchmarkNetworkTrain(b *testing.B) {
	layers := []layer.Layer{
		mustDense(784, 256, activations.Tanh{}),
		mustDense(256, 128, activations.Tanh{}),
		mustDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := make([]float32, 784)
	target := make([]float32, 10)
	fillRandom(input)
	fillRandom(target)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.Train(input, target)
	}
}

// BenchmarkNetworkTrainBatchSequentialSmall benchmarks training on a small batch sequentially.
func BenchmarkNetworkTrainBatchSequentialSmall(b *testing.B) {
	layers := []layer.Layer{
		mustDense(784, 256, activations.Tanh{}),
		mustDense(256, 128, activations.Tanh{}),
		mustDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	batchSize := 8
	batchX := make([][]float32, batchSize)
	batchY := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = make([]float32, 784)
		batchY[i] = make([]float32, 10)
		fillRandom(batchX[i])
		fillRandom(batchY[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.TrainBatch(batchX, batchY)
	}
}

// BenchmarkNetworkTrainBatchSequential benchmarks training on a batch sequentially.
func BenchmarkNetworkTrainBatchSequential(b *testing.B) {
	layers := []layer.Layer{
		mustDense(784, 256, activations.Tanh{}),
		mustDense(256, 128, activations.Tanh{}),
		mustDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	batchSize := 64
	batchX := make([][]float32, batchSize)
	batchY := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = make([]float32, 784)
		batchY[i] = make([]float32, 10)
		fillRandom(batchX[i])
		fillRandom(batchY[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.TrainBatch(batchX, batchY)
	}
}

// BenchmarkNetworkTrainBatchParallel benchmarks training on a batch in parallel.
func BenchmarkNetworkTrainBatchParallel(b *testing.B) {
	layers := []layer.Layer{
		mustDense(784, 256, activations.Tanh{}),
		mustDense(256, 128, activations.Tanh{}),
		mustDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	batchSize := 64
	batchX := make([][]float32, batchSize)
	batchY := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = make([]float32, 784)
		batchY[i] = make([]float32, 10)
		fillRandom(batchX[i])
		fillRandom(batchY[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.TrainBatch(batchX, batchY)
	}
}

// BenchmarkNetworkTrainLargeBatch benchmarks training on a large batch.
func BenchmarkNetworkTrainLargeBatch(b *testing.B) {
	layers := []layer.Layer{
		mustDense(784, 256, activations.Tanh{}),
		mustDense(256, 128, activations.Tanh{}),
		mustDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	batchSize := 256
	batchX := make([][]float32, batchSize)
	batchY := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = make([]float32, 784)
		batchY[i] = make([]float32, 10)
		fillRandom(batchX[i])
		fillRandom(batchY[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.TrainBatch(batchX, batchY)
	}
}

// BenchmarkNetworkParams benchmarks getting all network parameters.
func BenchmarkNetworkParams(b *testing.B) {
	layers := []layer.Layer{
		mustDense(784, 256, activations.Tanh{}),
		mustDense(256, 128, activations.Tanh{}),
		mustDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = network.Params()
	}
}

// BenchmarkNetworkGradients benchmarks getting all network gradients.
func BenchmarkNetworkGradients(b *testing.B) {
	layers := []layer.Layer{
		mustDense(784, 256, activations.Tanh{}),
		mustDense(256, 128, activations.Tanh{}),
		mustDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := make([]float32, 784)
	target := make([]float32, 10)
	fillRandom(input)
	fillRandom(target)

	// Forward pass
	output, _ := network.Forward(input)
	grad, _ := loss.MSE{}.Backward(output, target)
	_, _ = network.Backward(grad)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = network.Gradients()
	}
}

// BenchmarkNetworkLSTMTrainBatch benchmarks training on a batch with LSTM layers.
func BenchmarkNetworkLSTMTrainBatch(b *testing.B) {
	layers := []layer.Layer{
		layer.NewLSTM(10, 32),
		mustDense(32, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	batchSize := 16
	seqLen := 10
	batchX := make([][]float32, batchSize)
	batchY := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = make([]float32, 10)
		batchY[i] = make([]float32, 1)
		fillRandom(batchX[i])
		fillRandom(batchY[i])
	}

	// Warm-up to grow arena
	for s := 0; s < seqLen; s++ {
		network.TrainBatch(batchX, batchY)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simulate sequence training
		for s := 0; s < seqLen; s++ {
			network.TrainBatch(batchX, batchY)
		}
	}
}

// BenchmarkNetworkGRUTrainBatch benchmarks training on a batch with GRU layers.
func BenchmarkNetworkGRUTrainBatch(b *testing.B) {
	layers := []layer.Layer{
		layer.NewGRU(10, 32),
		mustDense(32, 1, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	batchSize := 16
	seqLen := 10
	batchX := make([][]float32, batchSize)
	batchY := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		batchX[i] = make([]float32, 10)
		batchY[i] = make([]float32, 1)
		fillRandom(batchX[i])
		fillRandom(batchY[i])
	}

	// Warm-up to grow arena
	for s := 0; s < seqLen; s++ {
		network.TrainBatch(batchX, batchY)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simulate sequence training
		for s := 0; s < seqLen; s++ {
			network.TrainBatch(batchX, batchY)
		}
	}
}
