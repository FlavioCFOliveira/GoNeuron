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

// Combined network examples: combining different layer types
func main() {
	fmt.Println("=== Combined Network Examples ===")

	// Example 1: CNN + Dense for image classification
	fmt.Println("Example 1: CNN + Dense for image classification")
	exampleCNNtoDense()

	// Example 2: Deep CNN for classification
	fmt.Println("\nExample 2: Deep CNN with multiple conv layers")
	exampleDeepCNN()
}

func exampleCNNtoDense() {
	// CNN for feature extraction followed by Dense for classification
	// Input: 1 channel, 16x16 image

	inChannels := 1
	outChannels := 8
	kernelSize := 3
	stride := 1
	padding := 1

	// Conv layer extracts features (deterministic initialization with seed=42)
	conv := layer.NewConv2D(inChannels, outChannels, kernelSize, stride, padding, activations.Tanh{})

	// After conv with padding=1, size stays same: 16x16
	// Output: 8 channels * 16 * 16 = 2048 features
	convOutput := outChannels * 16 * 16

	// Dense layer for classification (deterministic initialization)
	dense1 := layer.NewDense(convOutput, 64, activations.Tanh{})
	dense2 := layer.NewDense(64, 2, activations.Sigmoid{}) // 2 classes

	network := net.New(
		[]layer.Layer{conv, dense1, dense2},
		loss.MSE{},
		opt.NewAdam(0.01),
	)

	// Generate training data: 16x16 images with 2 classes
	trainX, trainY := generateImageLabels(50, 16, 2)

	fmt.Println("  Training CNN+Dense...")
	for epoch := 0; epoch < 200; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%50 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(trainX)))
		}
	}

	// Test
	fmt.Println("  Testing:")
	correct := 0
	for i := 0; i < 10; i++ {
		pred := network.Forward(trainX[i])
		predClass := maxIndex(pred)
		trueClass := maxIndex(trainY[i])
		if predClass == trueClass {
			correct++
		}
		fmt.Printf("  Sample %d: Predicted=%d, Target=%d\n", i, predClass, trueClass)
	}
	fmt.Printf("  Accuracy: %.1f%%\n", float64(correct)/10*100)
}

func exampleDeepCNN() {
	// Deep CNN: multiple conv layers for feature hierarchy
	// Input: 1 channel, 16x16

	inChannels := 1

	// First conv layer: 1 -> 8 channels (deterministic initialization)
	conv1 := layer.NewConv2D(inChannels, 8, 3, 1, 1, activations.Tanh{})

	// Second conv layer: 8 -> 16 channels (deterministic initialization)
	conv2 := layer.NewConv2D(8, 16, 3, 1, 1, activations.Tanh{})

	// Output after two convs with padding=1: 16 channels * 16 * 16 = 4096
	dense1 := layer.NewDense(4096, 128, activations.ReLU{})
	dense2 := layer.NewDense(128, 3, activations.Sigmoid{}) // 3 classes

	network := net.New(
		[]layer.Layer{conv1, conv2, dense1, dense2},
		loss.MSE{},
		opt.NewAdam(0.01),
	)

	// Generate training data: 3 classes
	trainX, trainY := generateImageLabels(50, 16, 3)

	fmt.Println("  Training Deep CNN...")
	for epoch := 0; epoch < 150; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%30 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(trainX)))
		}
	}

	// Test
	fmt.Println("  Testing:")
	correct := 0
	for i := 0; i < 10; i++ {
		pred := network.Forward(trainX[i])
		predClass := maxIndex(pred)
		trueClass := maxIndex(trainY[i])
		if predClass == trueClass {
			correct++
		}
		fmt.Printf("  Sample %d: Predicted=%d, Target=%d\n", i, predClass, trueClass)
	}
	fmt.Printf("  Accuracy: %.1f%%\n", float64(correct)/10*100)
}

// Helper functions

type ImageSequence struct {
	Input  [][]float64 // sequence of images
	Output []float64   // labels for each step
}

func generateImageLabels(nSamples, size, nClasses int) ([][]float64, [][]float64) {
	X := make([][]float64, nSamples)
	y := make([][]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		img := make([]float64, size*size)
		class := i % nClasses

		for j := range img {
			row := j / size
			col := j % size
			base := rand.Float64() * 0.2

			// Create patterns based on class
			if class == 0 {
				if row < size/2 {
					base += 0.5
				}
			} else if class == 1 {
				if col < size/2 {
					base += 0.5
				}
			} else {
				if row < size/2 && col < size/2 {
					base += 0.5
				}
			}
			img[j] = base + rand.Float64()*0.1
		}

		X[i] = img
		label := make([]float64, nClasses)
		label[class] = 1.0
		y[i] = label
	}

	return X, y
}

func generateImageSequences(nSequences, seqLength, size, nClasses int) []ImageSequence {
	sequences := make([]ImageSequence, nSequences)

	for i := 0; i < nSequences; i++ {
		seqLen := seqLength
		inputs := make([][]float64, seqLen)
		outputs := make([]float64, seqLen)

		for t := 0; t < seqLen; t++ {
			img := make([]float64, size*size)
			class := (i + t) % nClasses

			for j := range img {
				row := j / size
				col := j % size
				base := rand.Float64() * 0.2

				if class == 0 && row < size/2 {
					base += 0.5
				} else if class == 1 && col < size/2 {
					base += 0.5
				} else if class == 2 && row < size/2 && col < size/2 {
					base += 0.5
				}

				img[j] = base + rand.Float64()*0.1
			}

			inputs[t] = img
			outputs[t] = float64(class)
		}

		sequences[i] = ImageSequence{
			Input:  inputs,
			Output: outputs,
		}
	}

	return sequences
}

func generateTestSequence(seqLength, size int) [][]float64 {
	seq := make([][]float64, seqLength)

	for t := 0; t < seqLength; t++ {
		img := make([]float64, size*size)
		for j := range img {
			img[j] = rand.Float64() * 0.3
		}
		seq[t] = img
	}

	return seq
}

func maxIndex(slice []float64) int {
	maxIdx := 0
	for i := 1; i < len(slice); i++ {
		if slice[i] > slice[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}
