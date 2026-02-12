// Package activations provides benchmarks for activation functions.
package activations

import (
	"math/rand"
	"testing"
)

// fillRandom fills a slice with random values.
func fillRandom(slice []float64) {
	for i := range slice {
		slice[i] = rand.Float64()
	}
}

// BenchmarkReLUActivate benchmarks the ReLU activation function.
func BenchmarkReLUActivate(b *testing.B) {
	relu := ReLU{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			relu.Activate(x)
		}
	}
}

// BenchmarkReLUDerivative benchmarks the ReLU derivative function.
func BenchmarkReLUDerivative(b *testing.B) {
	relu := ReLU{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			relu.Derivative(x)
		}
	}
}

// BenchmarkReLUFull benchmarks both Activate and Derivative.
func BenchmarkReLUFull(b *testing.B) {
	relu := ReLU{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			_ = relu.Activate(x)
			_ = relu.Derivative(x)
		}
	}
}

// BenchmarkSigmoidActivate benchmarks the Sigmoid activation function.
func BenchmarkSigmoidActivate(b *testing.B) {
	sigmoid := Sigmoid{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			sigmoid.Activate(x)
		}
	}
}

// BenchmarkSigmoidDerivative benchmarks the Sigmoid derivative function.
func BenchmarkSigmoidDerivative(b *testing.B) {
	sigmoid := Sigmoid{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			sigmoid.Derivative(x)
		}
	}
}

// BenchmarkSigmoidFull benchmarks both Activate and Derivative.
func BenchmarkSigmoidFull(b *testing.B) {
	sigmoid := Sigmoid{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			_ = sigmoid.Activate(x)
			_ = sigmoid.Derivative(x)
		}
	}
}

// BenchmarkTanhActivate benchmarks the Tanh activation function.
func BenchmarkTanhActivate(b *testing.B) {
	tanh := Tanh{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			tanh.Activate(x)
		}
	}
}

// BenchmarkTanhDerivative benchmarks the Tanh derivative function.
func BenchmarkTanhDerivative(b *testing.B) {
	tanh := Tanh{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			tanh.Derivative(x)
		}
	}
}

// BenchmarkTanhFull benchmarks both Activate and Derivative.
func BenchmarkTanhFull(b *testing.B) {
	tanh := Tanh{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			_ = tanh.Activate(x)
			_ = tanh.Derivative(x)
		}
	}
}

// BenchmarkLeakyReLUActivate benchmarks the LeakyReLU activation function.
func BenchmarkLeakyReLUActivate(b *testing.B) {
	leakyReLU := NewLeakyReLU(0.01)
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			leakyReLU.Activate(x)
		}
	}
}

// BenchmarkLeakyReLUDerivative benchmarks the LeakyReLU derivative function.
func BenchmarkLeakyReLUDerivative(b *testing.B) {
	leakyReLU := NewLeakyReLU(0.01)
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			leakyReLU.Derivative(x)
		}
	}
}

// BenchmarkLeakyReLUFull benchmarks both Activate and Derivative.
func BenchmarkLeakyReLUFull(b *testing.B) {
	leakyReLU := NewLeakyReLU(0.01)
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			_ = leakyReLU.Activate(x)
			_ = leakyReLU.Derivative(x)
		}
	}
}

// BenchmarkSoftmaxActivateBatch benchmarks the Softmax activation batch function.
func BenchmarkSoftmaxActivateBatch(b *testing.B) {
	softmax := Softmax{}
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = softmax.ActivateBatch(inputs)
	}
}

// BenchmarkActivationsComparison benchmarks all activations with same input.
func BenchmarkActivationsComparison(b *testing.B) {
	inputs := make([]float64, 1000)
	fillRandom(inputs)

	relu := ReLU{}
	sigmoid := Sigmoid{}
	tanh := Tanh{}
	leakyReLU := NewLeakyReLU(0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, x := range inputs {
			_ = relu.Activate(x)
			_ = sigmoid.Activate(x)
			_ = tanh.Activate(x)
			_ = leakyReLU.Activate(x)
		}
	}
}
