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

// BenchmarkNetworkForward benchmarks a forward pass through a small network.
func BenchmarkNetworkForward(b *testing.B) {
	layers := []layer.Layer{
		layer.NewDense(784, 256, activations.Tanh{}),
		layer.NewDense(256, 128, activations.Tanh{}),
		layer.NewDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := make([]float32, 784)
	fillRandom(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.Forward(input)
	}
}

// BenchmarkNetworkBackward benchmarks a backward pass through a small network.
func BenchmarkNetworkBackward(b *testing.B) {
	layers := []layer.Layer{
		layer.NewDense(784, 256, activations.Tanh{}),
		layer.NewDense(256, 128, activations.Tanh{}),
		layer.NewDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := make([]float32, 784)
	grad := make([]float32, 10)
	fillRandom(input)
	fillRandom(grad)

	// Forward pass to set up state
	network.Forward(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = network.Backward(grad)
	}
}

// BenchmarkNetworkTrain benchmarks training on a single sample.
func BenchmarkNetworkTrain(b *testing.B) {
	layers := []layer.Layer{
		layer.NewDense(784, 256, activations.Tanh{}),
		layer.NewDense(256, 128, activations.Tanh{}),
		layer.NewDense(128, 10, activations.Sigmoid{}),
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

// BenchmarkNetworkTrainBatchSequential benchmarks training on a batch sequentially.
func BenchmarkNetworkTrainBatchSequential(b *testing.B) {
	layers := []layer.Layer{
		layer.NewDense(784, 256, activations.Tanh{}),
		layer.NewDense(256, 128, activations.Tanh{}),
		layer.NewDense(128, 10, activations.Sigmoid{}),
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
		layer.NewDense(784, 256, activations.Tanh{}),
		layer.NewDense(256, 128, activations.Tanh{}),
		layer.NewDense(128, 10, activations.Sigmoid{}),
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
		layer.NewDense(784, 256, activations.Tanh{}),
		layer.NewDense(256, 128, activations.Tanh{}),
		layer.NewDense(128, 10, activations.Sigmoid{}),
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
		layer.NewDense(784, 256, activations.Tanh{}),
		layer.NewDense(256, 128, activations.Tanh{}),
		layer.NewDense(128, 10, activations.Sigmoid{}),
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
		layer.NewDense(784, 256, activations.Tanh{}),
		layer.NewDense(256, 128, activations.Tanh{}),
		layer.NewDense(128, 10, activations.Sigmoid{}),
	}
	network := New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.1})

	input := make([]float32, 784)
	target := make([]float32, 10)
	fillRandom(input)
	fillRandom(target)

	// Forward pass
	network.Forward(input)
	grad := loss.MSE{}.Backward(network.Forward(input), target)
	_ = network.Backward(grad)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = network.Gradients()
	}
}
