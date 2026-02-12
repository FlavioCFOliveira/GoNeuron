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

// CNN examples demonstrating convolutional neural networks
func main() {
	rand.Seed(42)

	fmt.Println("=== CNN Examples ===")

	// Example 1: Simple 1D convolution for sequence
	fmt.Println("Example 1: 1D Convolution for sequence features")
	exampleConv1D()

	// Example 2: 2D convolution for image-like data
	fmt.Println("\nExample 2: 2D Convolution for 8x8 images")
	exampleConv2D()

	// Example 3: Simple CNN for binary classification
	fmt.Println("\nExample 3: Simple CNN for binary classification")
	exampleSimpleCNN()
}

func exampleConv1D() {
	// 1D convolution on sequence data
	// Input: sequence of 16 values with 2 channels
	// Kernel size: 3, Output channels: 4

	inChannels := 2
	outChannels := 4
	kernelSize := 3
	stride := 1
	padding := 0

	conv := layer.NewConv2D(inChannels, outChannels, kernelSize, stride, padding, activations.ReLU{})

	// Input: [2 channels, 8 height, 8 width] = 128 values
	inputSize := inChannels * 8 * 8
	input := make([]float64, inputSize)
	for i := range input {
		input[i] = rand.Float64()
	}

	// Forward pass
	output := conv.Forward(input)
	fmt.Printf("  Input shape: [%d, 8, 8]\n", inChannels)
	fmt.Printf("  Output shape: [%d, %d, %d]\n", outChannels, 6, 6)
	fmt.Printf("  Output length: %d\n", len(output))
	fmt.Printf("  Sample output values: %.4f, %.4f, %.4f\n",
		output[0], output[1], output[2])
}

func exampleConv2D() {
	// 2D convolution on 8x8 image-like data
	// 1 input channel -> 4 output channels
	// Kernel: 3x3

	inChannels := 1
	outChannels := 4
	kernelSize := 3
	stride := 1
	padding := 1 // Keep same spatial size

	conv := layer.NewConv2D(inChannels, outChannels, kernelSize, stride, padding, activations.ReLU{})

	// Input: 1 channel, 8x8 = 64 values
	input := make([]float64, 64)
	for i := range input {
		input[i] = rand.Float64()
	}

	// Forward pass
	output := conv.Forward(input)
	outSize := computeConvOutputSize(8, kernelSize, stride, padding)

	fmt.Printf("  Input shape: [1, 8, 8]\n")
	fmt.Printf("  Output shape: [%d, %d, %d]\n", outChannels, outSize, outSize)
	fmt.Printf("  Sample output values: %.4f, %.4f, %.4f\n",
		output[0], output[1], output[2])
}

func exampleSimpleCNN() {
	// Simple CNN: Conv2D -> Dense
	// 1 input channel, 8x8 images

	inChannels := 1
	outChannels := 4
	kernelSize := 3
	stride := 1
	padding := 0

	conv := layer.NewConv2D(inChannels, outChannels, kernelSize, stride, padding, activations.ReLU{})
	// After conv: 4 channels * 6 * 6 = 144 outputs
	dense := layer.NewDense(outChannels*6*6, 1, activations.Sigmoid{})

	network := net.New(
		[]layer.Layer{conv, dense},
		loss.MSE{},
		opt.NewAdam(0.01),
	)

	// Generate simple training data
	trainX, trainY := generateBinaryImageLabels(20, 8)

	fmt.Println("  Training...")
	for epoch := 0; epoch < 100; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%20 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(trainX)))
		}
	}

	// Test
	fmt.Println("  Testing:")
	correct := 0
	for i := 0; i < 5; i++ {
		pred := network.Forward(trainX[i])
		predClass := 0
		if pred[0] > 0.5 {
			predClass = 1
		}
		trueClass := int(trainY[i][0])
		if predClass == trueClass {
			correct++
		}
		fmt.Printf("  Sample %d: Predicted=%.4f (%d), Target=%d\n",
			i, pred[0], predClass, trueClass)
	}
	fmt.Printf("  Accuracy: %.1f%%\n", float64(correct)/5*100)
}

func computeConvOutputSize(inputSize, kernelSize, stride, padding int) int {
	return (inputSize + 2*padding - kernelSize) / stride + 1
}

func generateBinaryImageLabels(nSamples, size int) ([][]float64, [][]float64) {
	X := make([][]float64, nSamples)
	y := make([][]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		// Create image with random shape
		img := make([]float64, size*size)
		class := i % 2

		for j := range img {
			if class == 0 {
				// Class 0: random noise
				img[j] = rand.Float64() * 0.2
			} else {
				// Class 1: some pattern
				row := j / size
				col := j % size
				base := (rand.Float64() - 0.5) * 0.5
				if row < size/2 && col < size/2 {
					base += 0.5
				}
				img[j] = base + rand.Float64()*0.1
			}
		}

		X[i] = img
		y[i] = []float64{float64(class)}
	}

	return X, y
}
