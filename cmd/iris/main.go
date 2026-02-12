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

// Iris dataset: 3 classes (Setosa, Versicolor, Virginica)
// Each sample has 4 features (sepal length, sepal width, petal length, petal width)
func main() {
	fmt.Println("Training Iris classifier (4-8-6-3 network)...")

	// Iris dataset: 4 inputs -> 3 outputs (one-hot encoded)
	in := 4
	hidden1 := 8
	hidden2 := 6
	out := 3

	// Initialize layers with deterministic initialization (seed=42)
	l1 := layer.NewDense(in, hidden1, activations.ReLU{})
	l2 := layer.NewDense(hidden1, hidden2, activations.ReLU{})
	l3 := layer.NewDense(hidden2, out, activations.Sigmoid{})

	network := net.New(
		[]layer.Layer{l1, l2, l3},
		loss.MSE{}, // Use MSE for multi-class with Sigmoid
		opt.NewAdam(0.02),
	)

	// Iris dataset (simplified - using only the linearly separable parts)
	// Class 0: Setosa, Class 1: Versicolor, Class 2: Virginica
	trainX, trainY := generateIrisData()

	// Training loop
	for epoch := 0; epoch < 2000; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%200 == 0 {
			acc := evaluate(network, trainX, trainY)
			fmt.Printf("Epoch %d, Loss: %.4f, Accuracy: %.1f%%\n",
				epoch, totalLoss/float64(len(trainX)), acc*100)
		}
	}

	// Final evaluation
	acc := evaluate(network, trainX, trainY)
	fmt.Printf("\nFinal Accuracy: %.1f%%\n", acc*100)

	// Test on training data
	fmt.Println("\nSample predictions:")
	for i := 0; i < 10; i++ {
		pred := network.Forward(trainX[i])
		predClass := maxIndex(pred)
		trueClass := maxIndex(trainY[i])
		fmt.Printf("Sample %d: Predicted=%d, Actual=%d\n", i, predClass, trueClass)
	}
}

func generateIrisData() ([][]float64, [][]float64) {
	// Simplified Iris data - using mean values for each class
	// In a real scenario, you'd load the full dataset
	// Class 0: Setosa (sepal length 5.0, sepal width 3.4, petal length 1.5, petal width 0.2)
	// Class 1: Versicolor (sepal length 5.9, sepal width 2.8, petal length 4.3, petal width 1.3)
	// Class 2: Virginica (sepal length 6.6, sepal width 3.0, petal length 5.6, petal width 2.0)

	trainX := make([][]float64, 90)
	trainY := make([][]float64, 90)

	for i := 0; i < 30; i++ {
		// Setosa
		trainX[i] = addNoise([]float64{5.0, 3.4, 1.5, 0.2}, 0.2)
		trainY[i] = oneHot(0, 3)
	}

	for i := 30; i < 60; i++ {
		// Versicolor
		trainX[i] = addNoise([]float64{5.9, 2.8, 4.3, 1.3}, 0.25)
		trainY[i] = oneHot(1, 3)
	}

	for i := 60; i < 90; i++ {
		// Virginica
		trainX[i] = addNoise([]float64{6.6, 3.0, 5.6, 2.0}, 0.25)
		trainY[i] = oneHot(2, 3)
	}

	return trainX, trainY
}

func addNoise(sample []float64, noise float64) []float64 {
	result := make([]float64, len(sample))
	for i, v := range sample {
		result[i] = v + (rand.Float64()*2-1)*noise
	}
	return result
}

func oneHot(idx, size int) []float64 {
	result := make([]float64, size)
	result[idx] = 1.0
	return result
}

func evaluate(net *net.Network, X [][]float64, y [][]float64) float64 {
	correct := 0
	for i := range X {
		pred := net.Forward(X[i])
		if maxIndex(pred) == maxIndex(y[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(X))
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
