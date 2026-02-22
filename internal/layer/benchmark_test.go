// Package layer provides benchmarks for neural network layer implementations.
package layer

import (
	"math/rand"
	"testing"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
)

// fillRandom fills a slice with random values.
func fillRandom(slice []float32) {
	for i := range slice {
		slice[i] = rand.Float32()
	}
}

// BenchmarkDenseForward benchmarks the forward pass of a dense layer.
func BenchmarkDenseForward(b *testing.B) {
	layer := NewDense(784, 256, activations.Tanh{})
	input := make([]float32, 784)
	fillRandom(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

// BenchmarkDenseBackward benchmarks the backward pass of a dense layer.
func BenchmarkDenseBackward(b *testing.B) {
	layer := NewDense(784, 256, activations.Tanh{})
	input := make([]float32, 784)
	grad := make([]float32, 256)
	fillRandom(input)
	fillRandom(grad)

	// Forward pass to set up state
	layer.Forward(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Backward(grad)
	}
}

// BenchmarkDenseFull benchmarks a complete forward and backward pass.
func BenchmarkDenseFull(b *testing.B) {
	layer := NewDense(784, 256, activations.Tanh{})
	input := make([]float32, 784)
	grad := make([]float32, 256)
	fillRandom(input)
	fillRandom(grad)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
		layer.Backward(grad)
	}
}

// BenchmarkDenseLarge benchmarks a large dense layer (typical MLP hidden layer).
func BenchmarkDenseLarge(b *testing.B) {
	layer := NewDense(1024, 1024, activations.Tanh{})
	input := make([]float32, 1024)
	fillRandom(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

// BenchmarkConv2DForward benchmarks the forward pass of a convolutional layer.
func BenchmarkConv2DForward(b *testing.B) {
	// Typical CNN configuration: 3 input channels, 64 output channels, 3x3 kernel
	// Input: 32x32x3 = 3072 elements
	layer := NewConv2D(3, 64, 3, 1, 1, activations.Tanh{})
	input := make([]float32, 3*32*32)
	fillRandom(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

// BenchmarkConv2DBackward benchmarks the backward pass of a convolutional layer.
func BenchmarkConv2DBackward(b *testing.B) {
	// Typical CNN configuration
	layer := NewConv2D(3, 64, 3, 1, 1, activations.Tanh{})
	input := make([]float32, 3*32*32)
	grad := make([]float32, 64*32*32) // Output: 64 channels, 32x32
	fillRandom(input)
	fillRandom(grad)

	// Forward pass to set up state
	layer.Forward(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Backward(grad)
	}
}

// BenchmarkConv2DFull benchmarks a complete forward and backward pass.
func BenchmarkConv2DFull(b *testing.B) {
	layer := NewConv2D(3, 64, 3, 1, 1, activations.Tanh{})
	input := make([]float32, 3*32*32)
	grad := make([]float32, 64*32*32)
	fillRandom(input)
	fillRandom(grad)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
		layer.Backward(grad)
	}
}

// BenchmarkConv2DLarge benchmarks a larger convolutional layer.
func BenchmarkConv2DLarge(b *testing.B) {
	// Larger CNN: 64 input channels, 128 output channels, 3x3 kernel
	// Input: 64x64x64 = 262144 elements
	layer := NewConv2D(64, 128, 3, 1, 1, activations.Tanh{})
	input := make([]float32, 64*64*64)
	fillRandom(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

// BenchmarkLSTMForward benchmarks the forward pass of an LSTM layer.
func BenchmarkLSTMForward(b *testing.B) {
	layer := NewLSTM(256, 512) // Typical LSTM config
	input := make([]float32, 256)
	fillRandom(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

// BenchmarkLSTMBackward benchmarks the backward pass of an LSTM layer.
func BenchmarkLSTMBackward(b *testing.B) {
	layer := NewLSTM(256, 512)
	input := make([]float32, 256)
	grad := make([]float32, 512)
	fillRandom(input)
	fillRandom(grad)

	// Forward pass to set up state
	layer.Forward(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Backward(grad)
	}
}

// BenchmarkLSTMFull benchmarks a complete forward and backward pass.
func BenchmarkLSTMFull(b *testing.B) {
	layer := NewLSTM(256, 512)
	input := make([]float32, 256)
	grad := make([]float32, 512)
	fillRandom(input)
	fillRandom(grad)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
		layer.Backward(grad)
	}
}

// BenchmarkLSTMSequence benchmarks processing a sequence of 10 timesteps.
func BenchmarkLSTMSequence(b *testing.B) {
	layer := NewLSTM(256, 512)
	input := make([]float32, 256)
	grad := make([]float32, 512)
	fillRandom(input)
	fillRandom(grad)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Process 10 timesteps
		for t := 0; t < 10; t++ {
			layer.Forward(input)
		}
		// Backward through all timesteps
		for t := 0; t < 10; t++ {
			layer.Backward(grad)
		}
		layer.Reset()
	}
}

// BenchmarkDenseParams benchmarks getting/setting parameters.
func BenchmarkDenseParams(b *testing.B) {
	layer := NewDense(784, 256, activations.Tanh{})
	params := layer.Params()

	b.Run("Params", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = layer.Params()
		}
	})

	b.Run("SetParams", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			layer.SetParams(params)
		}
	})
}

// BenchmarkConv2DParams benchmarks getting/setting parameters.
func BenchmarkConv2DParams(b *testing.B) {
	layer := NewConv2D(3, 64, 3, 1, 1, activations.Tanh{})
	params := layer.Params()

	b.Run("Params", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = layer.Params()
		}
	})

	b.Run("SetParams", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			layer.SetParams(params)
		}
	})
}

// BenchmarkLSTMParams benchmarks getting/setting parameters.
func BenchmarkLSTMParams(b *testing.B) {
	layer := NewLSTM(256, 512)
	params := layer.Params()

	b.Run("Params", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = layer.Params()
		}
	})

	b.Run("SetParams", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			layer.SetParams(params)
		}
	})
}
