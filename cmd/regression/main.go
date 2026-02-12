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

// Regression examples: predicting continuous values
// Uses normalized data and appropriate activation functions
func main() {
	rand.Seed(42)

	fmt.Println("=== Regression Examples ===")

	// Example 1: Simple linear regression with normalized data
	fmt.Println("Example 1: Linear function with normalized data")
	exampleLinearRegression()

	// Example 2: Non-linear regression (y = x²)
	fmt.Println("\nExample 2: Non-linear function y = x²")
	exampleQuadraticRegression()

	// Example 3: Multi-input regression
	fmt.Println("\nExample 3: Multi-input function z = x + y")
	exampleMultiInputRegression()

	// Example 4: Using Huber loss for robust regression
	fmt.Println("\nExample 4: Robust regression with Huber loss")
	exampleHuberRegression()
}

func exampleLinearRegression() {
	// Network: 1 input -> 8 hidden -> 1 output
	// Use normalized data in range [0, 1] with Sigmoid output
	in, hidden, out := 1, 8, 1

	l1 := layer.NewDense(in, hidden, activations.Tanh{})
	l2 := layer.NewDense(hidden, out, activations.Sigmoid{})

	initDense(l1, in, hidden, false)
	initDense(l2, hidden, out, false)

	network := net.New(
		[]layer.Layer{l1, l2},
		loss.MSE{},
		opt.NewAdam(0.1),
	)

	// Generate training data: y = 0.5x + 0.3 (normalized to [0, 1] range)
	trainX, trainY := generateLinearDataNormalized(100, 0.5, 0.3)

	// Train
	for epoch := 0; epoch < 1000; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%200 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(len(trainX)))
		}
	}

	// Test
	fmt.Println("  Test predictions:")
	testValues := []float64{0, 0.25, 0.5, 0.75, 1.0}
	for _, x := range testValues {
		pred := network.Forward([]float64{x})
		expected := 0.5*x + 0.3
		err := math.Abs(pred[0] - expected)
		fmt.Printf("    x=%.2f: predicted=%.4f, expected=%.4f, error=%.6f\n",
			x, pred[0], expected, err)
	}
}

func exampleQuadraticRegression() {
	// Network: 1 input -> 16 hidden -> 8 hidden -> 1 output
	in, h1, h2, out := 1, 16, 8, 1

	l1 := layer.NewDense(in, h1, activations.ReLU{})
	l2 := layer.NewDense(h1, h2, activations.ReLU{})
	l3 := layer.NewDense(h2, out, activations.Sigmoid{})

	initDense(l1, in, h1, false)
	initDense(l2, h1, h2, false)
	initDense(l3, h2, out, false)

	network := net.New(
		[]layer.Layer{l1, l2, l3},
		loss.MSE{},
		opt.NewAdam(0.05),
	)

	// Generate training data: y = x² (normalized to [0, 1] range)
	trainX, trainY := generateQuadraticDataNormalized(200)

	// Train
	for epoch := 0; epoch < 1500; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%300 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(len(trainX)))
		}
	}

	// Test
	fmt.Println("  Test predictions:")
	testValues := []float64{0, 0.25, 0.5, 0.75, 1.0}
	for _, x := range testValues {
		pred := network.Forward([]float64{x})
		expected := x * x
		err := math.Abs(pred[0] - expected)
		fmt.Printf("    x=%.2f: predicted=%.4f, expected=%.4f, error=%.6f\n",
			x, pred[0], expected, err)
	}
}

func exampleMultiInputRegression() {
	// Network: 2 inputs -> 12 hidden -> 6 hidden -> 1 output
	in, h1, h2, out := 2, 12, 6, 1

	l1 := layer.NewDense(in, h1, activations.Tanh{})
	l2 := layer.NewDense(h1, h2, activations.Tanh{})
	l3 := layer.NewDense(h2, out, activations.Sigmoid{})

	initDense(l1, in, h1, false)
	initDense(l2, h1, h2, false)
	initDense(l3, h2, out, false)

	network := net.New(
		[]layer.Layer{l1, l2, l3},
		loss.MSE{},
		opt.NewAdam(0.05),
	)

	// Generate training data: z = (x + y) / 2 (normalized)
	trainX, trainY := generateMultiInputDataNormalized(150)

	// Train
	for epoch := 0; epoch < 800; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%200 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(len(trainX)))
		}
	}

	// Test
	fmt.Println("  Test predictions:")
	testCases := [][2]float64{
		{0, 0}, {0.5, 0.5}, {1, 0}, {0, 1}, {0.75, 0.25},
	}
	for _, tc := range testCases {
		pred := network.Forward([]float64{tc[0], tc[1]})
		expected := (tc[0] + tc[1]) / 2
		err := math.Abs(pred[0] - expected)
		fmt.Printf("    x=%.2f, y=%.2f: predicted=%.4f, expected=%.4f, error=%.6f\n",
			tc[0], tc[1], pred[0], expected, err)
	}
}

func exampleHuberRegression() {
	// Network: 1 input -> 10 hidden -> 1 output
	in, hidden, out := 1, 10, 1

	l1 := layer.NewDense(in, hidden, activations.Tanh{})
	l2 := layer.NewDense(hidden, out, activations.Sigmoid{})

	initDense(l1, in, hidden, false)
	initDense(l2, hidden, out, false)

	network := net.New(
		[]layer.Layer{l1, l2},
		loss.NewHuber(0.5), // Huber loss for robustness to outliers
		opt.NewAdam(0.1),
	)

	// Generate training data with outliers (normalized)
	trainX, trainY := generateDataWithOutliersNormalized(100)

	// Train
	for epoch := 0; epoch < 1000; epoch++ {
		totalLoss := 0.0
		for i := range trainX {
			loss := network.Train(trainX[i], trainY[i])
			totalLoss += loss
		}
		if epoch%200 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(len(trainX)))
		}
	}

	// Test
	fmt.Println("  Test predictions:")
	testValues := []float64{0, 0.25, 0.5, 0.75, 1.0}
	for _, x := range testValues {
		pred := network.Forward([]float64{x})
		expected := x * x
		err := math.Abs(pred[0] - expected)
		fmt.Printf("    x=%.2f: predicted=%.4f, expected=%.4f, error=%.6f\n",
			x, pred[0], expected, err)
	}
}

func generateLinearDataNormalized(n int, slope, intercept float64) ([][]float64, [][]float64) {
	X := make([][]float64, n)
	y := make([][]float64, n)

	for i := 0; i < n; i++ {
		x := rand.Float64() // Range: 0 to 1
		noise := (rand.Float64() - 0.5) * 0.05
		yVal := slope*x + intercept + noise
		X[i] = []float64{x}
		y[i] = []float64{yVal}
	}
	return X, y
}

func generateQuadraticDataNormalized(n int) ([][]float64, [][]float64) {
	X := make([][]float64, n)
	y := make([][]float64, n)

	for i := 0; i < n; i++ {
		x := rand.Float64()
		noise := (rand.Float64() - 0.5) * 0.05
		yVal := x*x + noise
		X[i] = []float64{x}
		y[i] = []float64{yVal}
	}
	return X, y
}

func generateMultiInputDataNormalized(n int) ([][]float64, [][]float64) {
	X := make([][]float64, n)
	y := make([][]float64, n)

	for i := 0; i < n; i++ {
		x1 := rand.Float64()
		x2 := rand.Float64()
		noise := (rand.Float64() - 0.5) * 0.05
		z := (x1 + x2) / 2 + noise // Average in [0, 1]
		X[i] = []float64{x1, x2}
		y[i] = []float64{z}
	}
	return X, y
}

func generateDataWithOutliersNormalized(n int) ([][]float64, [][]float64) {
	X := make([][]float64, n)
	y := make([][]float64, n)

	for i := 0; i < n; i++ {
		x := rand.Float64()
		noise := (rand.Float64() - 0.5) * 0.05
		yVal := x*x + noise

		// Add occasional outliers (10% of data)
		if rand.Float64() < 0.1 {
			yVal += (rand.Float64() - 0.5) * 0.2
		}

		X[i] = []float64{x}
		y[i] = []float64{yVal}
	}
	return X, y
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
