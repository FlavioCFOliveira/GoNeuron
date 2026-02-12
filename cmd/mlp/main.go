package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// MLP (Multi-Layer Perceptron) example
// Demonstrates training networks with different hidden layer configurations
func main() {
	rand.Seed(42)

	fmt.Println("=== MLP Examples ===")

	// Example 1: Small network (2-4-1) for XOR
	fmt.Println("Example 1: Small XOR network (2-4-1)")
	exampleXOR()

	// Example 2: Larger network for XOR
	fmt.Println("\nExample 2: Larger XOR network (2-8-8-1)")
	exampleXLOR()

	// Example 3: Binary classification with sigmoid activation
	fmt.Println("\nExample 3: Binary classification (4-8-4-1)")
	exampleBinaryClassification()
}

func exampleXOR() {
	// XOR: 2 inputs -> 4 hidden -> 1 output
	in, hidden, out := 2, 4, 1

	l1 := layer.NewDense(in, hidden, activations.ReLU{})
	l2 := layer.NewDense(hidden, out, activations.Sigmoid{})

	initDense(l1, in, hidden, true)
	initDense(l2, hidden, out, false)

	network := net.New(
		[]layer.Layer{l1, l2},
		loss.MSE{},
		opt.SGD{LearningRate: 0.1},
	)

	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {1}, {1}, {0}}

	// Train
	for epoch := 0; epoch < 5000; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%1000 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.4f\n", epoch, totalLoss/4)
		}
	}

	// Test
	fmt.Println("  Results:")
	for i := range trainX {
		pred := network.Forward(trainX[i])
		fmt.Printf("    %v -> %.4f (target: %.0f)\n", trainX[i], pred[0], trainY[i][0])
	}
}

func exampleXLOR() {
	// Extended XOR: 2 inputs -> 8 hidden -> 8 hidden -> 1 output
	in, h1, h2, out := 2, 8, 8, 1

	l1 := layer.NewDense(in, h1, activations.ReLU{})
	l2 := layer.NewDense(h1, h2, activations.ReLU{})
	l3 := layer.NewDense(h2, out, activations.Sigmoid{})

	initDense(l1, in, h1, true)
	initDense(l2, h1, h2, true)
	initDense(l3, h2, out, false)

	network := net.New(
		[]layer.Layer{l1, l2, l3},
		loss.MSE{},
		opt.NewAdam(0.01),
	)

	trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float64{{0}, {1}, {1}, {0}}

	// Train
	for epoch := 0; epoch < 3000; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%500 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.4f\n", epoch, totalLoss/4)
		}
	}

	fmt.Println("  Results:")
	for i := range trainX {
		pred := network.Forward(trainX[i])
		fmt.Printf("    %v -> %.4f (target: %.0f)\n", trainX[i], pred[0], trainY[i][0])
	}
}

func exampleBinaryClassification() {
	// Binary classification: 4 inputs -> 8 hidden -> 4 hidden -> 1 output
	in, h1, h2, out := 4, 8, 4, 1

	l1 := layer.NewDense(in, h1, activations.ReLU{})
	l2 := layer.NewDense(h1, h2, activations.ReLU{})
	l3 := layer.NewDense(h2, out, activations.Sigmoid{})

	initDense(l1, in, h1, true)
	initDense(l2, h1, h2, true)
	initDense(l3, h2, out, false)

	network := net.New(
		[]layer.Layer{l1, l2, l3},
		loss.MSE{},
		opt.NewAdam(0.05),
	)

	// Generate synthetic binary classification data
	trainX, trainY := generateBinaryData(100)

	// Train
	for epoch := 0; epoch < 1000; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%200 == 0 {
			acc := evaluate(network, trainX, trainY)
			fmt.Printf("  Epoch %d, Loss: %.4f, Accuracy: %.1f%%\n",
				epoch, totalLoss/float64(len(trainX)), acc*100)
		}
	}

	acc := evaluate(network, trainX, trainY)
	fmt.Printf("  Final Accuracy: %.1f%%\n", acc*100)
}

func generateBinaryData(n int) ([][]float64, [][]float64) {
	X := make([][]float64, n)
	y := make([][]float64, n)

	for i := 0; i < n; i++ {
		// Generate 4 features with different means for each class
		f1 := rand.Float64()*2 - 1
		f2 := rand.Float64()*2 - 1
		f3 := rand.Float64() + 0.5
		f4 := rand.Float64()*3 - 1.5

		X[i] = []float64{f1, f2, f3, f4}

		// Class 0 if f1 + f2 > 0, Class 1 otherwise
		if f1+f2 > 0 {
			y[i] = []float64{1}
		} else {
			y[i] = []float64{0}
		}
	}

	return X, y
}

func evaluate(net *net.Network, X [][]float64, y [][]float64) float64 {
	correct := 0
	for i := range X {
		pred := net.Forward(X[i])
		predClass := 0
		if pred[0] > 0.5 {
			predClass = 1
		}
		if predClass == int(y[i][0]) {
			correct++
		}
	}
	return float64(correct) / float64(len(X))
}

func initDense(d *layer.Dense, in, out int, positiveBiases bool) {
	scale := math.Sqrt(2.0 / float64(in+out))
	for i := 0; i < out; i++ {
		for j := 0; j < in; j++ {
			d.SetWeight(i, j, rand.Float64()*2*scale-scale)
		}
		if positiveBiases {
			d.SetBias(i, rand.Float64()*0.5+0.1)
		} else {
			d.SetBias(i, rand.Float64()*0.2-0.1)
		}
	}
}
