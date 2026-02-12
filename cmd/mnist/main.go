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

// MNIST-style digit classification example
// Demonstrates training on larger input dimensions
func main() {
	rand.Seed(42)

	fmt.Println("=== MNIST-style Digit Classification ===")

	// Simulated MNIST: 784 inputs (28x28 pixels) -> 128 -> 64 -> 10 outputs
	in := 784
	hidden1 := 128
	hidden2 := 64
	out := 10

	fmt.Printf("Architecture: %d-%d-%d-%d\n\n", in, hidden1, hidden2, out)

	// Initialize layers
	l1 := layer.NewDense(in, hidden1, activations.ReLU{})
	l2 := layer.NewDense(hidden1, hidden2, activations.ReLU{})
	l3 := layer.NewDense(hidden2, out, activations.Sigmoid{})

	initDense(l1, in, hidden1, false)
	initDense(l2, hidden1, hidden2, false)
	initDense(l3, hidden2, out, false)

	network := net.New(
		[]layer.Layer{l1, l2, l3},
		loss.MSE{}, // Use MSE for multi-class with Sigmoid
		opt.NewAdam(0.002),
	)

	// Generate synthetic MNIST-like data
	trainX, trainY := generateMNISTData(500)

	fmt.Println("Training...")
	for epoch := 0; epoch < 500; epoch++ {
		totalLoss := 0.0
		correct := 0

		// Mini-batch training (using full batch here for simplicity)
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss

			// Track accuracy
			pred := network.Forward(trainX[i])
			if maxIndex(pred) == maxIndex(trainY[i]) {
				correct++
			}
		}

		if epoch%50 == 0 {
			acc := float64(correct) / float64(len(trainX))
			fmt.Printf("  Epoch %3d: Loss=%.4f, Accuracy=%.1f%%\n",
				epoch, totalLoss/float64(len(trainX)), acc*100)
		}
	}

	// Final evaluation
	acc := evaluate(network, trainX, trainY)
	fmt.Printf("\nFinal Training Accuracy: %.1f%%\n", acc*100)

	// Show some predictions
	fmt.Println("\nSample Predictions:")
	for i := 0; i < 10; i++ {
		pred := network.Forward(trainX[i])
		predClass := maxIndex(pred)
		trueClass := maxIndex(trainY[i])
		probs := make([]string, len(pred))
		for j := range pred {
			probs[j] = fmt.Sprintf("%.2f", pred[j])
		}
		match := "✓"
		if predClass != trueClass {
			match = "✗"
		}
		fmt.Printf("  %s True=%d, Predicted=%d\n  %v\n",
			match, trueClass, predClass, probs)
	}
}

func generateMNISTData(n int) ([][]float64, [][]float64) {
	X := make([][]float64, n)
	y := make([][]float64, n)

	digitPatterns := map[int][]float64{
		0: {1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1}, // Simplified 0
		1: {0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0}, // Simplified 1
		2: {1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1}, // Simplified 2
		3: {1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1}, // Simplified 3
		4: {1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1}, // Simplified 4
		5: {1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1}, // Simplified 5
		6: {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1}, // Simplified 6
		7: {1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0}, // Simplified 7
		8: {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1}, // Simplified 8
		9: {1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1}, // Simplified 9
	}

	// Each sample is a 28x28 image = 784 pixels
	// We'll use a 15-pixel pattern scaled to 28x28
	patternWidth := 5
	patternHeight := 3

	for i := 0; i < n; i++ {
		digit := i % 10
		y[i] = oneHot(digit, 10)

		// Create a 784-dimensional input vector
		// Using a simplified pattern with noise
		X[i] = make([]float64, 784)

		// Create a simple pattern that scales to 28x28
		pattern := digitPatterns[digit]

		for py := 0; py < patternHeight; py++ {
			for px := 0; px < patternWidth; px++ {
				if px < patternWidth && py < patternHeight {
					val := pattern[py*patternWidth+px]
					// Scale to 28x28 grid
					for sy := 0; sy < 7; sy++ {
						for sx := 0; sx < 7; sx++ {
							pos := (py*7+sy)*28 + (px*7 + sx)
							noise := (rand.Float64() - 0.5) * 0.1
							if val > 0.5 {
								X[i][pos] = 1.0 + noise
							} else {
								X[i][pos] = 0.0 + noise
							}
						}
					}
				}
			}
		}
	}

	return X, y
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
