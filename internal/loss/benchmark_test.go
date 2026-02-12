// Package loss provides benchmarks for loss functions.
package loss

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

// BenchmarkMSEForward benchmarks MSE loss forward pass.
func BenchmarkMSEForward(b *testing.B) {
	mse := MSE{}
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mse.Forward(yPred, yTrue)
	}
}

// BenchmarkMSEBackward benchmarks MSE loss backward pass.
func BenchmarkMSEBackward(b *testing.B) {
	mse := MSE{}
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mse.Backward(yPred, yTrue)
	}
}

// BenchmarkMSEBackwardInPlace benchmarks MSE in-place gradient computation.
func BenchmarkMSEBackwardInPlace(b *testing.B) {
	mse := MSE{}
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	grad := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mse.BackwardInPlace(yPred, yTrue, grad)
	}
}

// BenchmarkMSEFull benchmarks complete forward and backward pass.
func BenchmarkMSEFull(b *testing.B) {
	mse := MSE{}
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mse.Forward(yPred, yTrue)
		_ = mse.Backward(yPred, yTrue)
	}
}

// BenchmarkCrossEntropyForward benchmarks CrossEntropy loss forward pass.
func BenchmarkCrossEntropyForward(b *testing.B) {
	ce := CrossEntropy{}
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	// Normalize to make valid probabilities
	for i := range yPred {
		yPred[i] = yPred[i] / 10.0 // Scale down
		yTrue[i] = yTrue[i] / 10.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ce.Forward(yPred, yTrue)
	}
}

// BenchmarkCrossEntropyBackward benchmarks CrossEntropy loss backward pass.
func BenchmarkCrossEntropyBackward(b *testing.B) {
	ce := CrossEntropy{}
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ce.Backward(yPred, yTrue)
	}
}

// BenchmarkCrossEntropyFull benchmarks complete forward and backward pass.
func BenchmarkCrossEntropyFull(b *testing.B) {
	ce := CrossEntropy{}
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ce.Forward(yPred, yTrue)
		_ = ce.Backward(yPred, yTrue)
	}
}

// BenchmarkHuberForward benchmarks Huber loss forward pass with default delta.
func BenchmarkHuberForward(b *testing.B) {
	huber := NewHuber(1.0)
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = huber.Forward(yPred, yTrue)
	}
}

// BenchmarkHuberBackward benchmarks Huber loss backward pass.
func BenchmarkHuberBackward(b *testing.B) {
	huber := NewHuber(1.0)
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = huber.Backward(yPred, yTrue)
	}
}

// BenchmarkHuberFull benchmarks complete forward and backward pass.
func BenchmarkHuberFull(b *testing.B) {
	huber := NewHuber(1.0)
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = huber.Forward(yPred, yTrue)
		_ = huber.Backward(yPred, yTrue)
	}
}

// BenchmarkHuberSmallDelta benchmarks Huber with small delta.
func BenchmarkHuberSmallDelta(b *testing.B) {
	huber := NewHuber(0.1)
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = huber.Forward(yPred, yTrue)
		_ = huber.Backward(yPred, yTrue)
	}
}

// BenchmarkHuberLargeDelta benchmarks Huber with large delta.
func BenchmarkHuberLargeDelta(b *testing.B) {
	huber := NewHuber(5.0)
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = huber.Forward(yPred, yTrue)
		_ = huber.Backward(yPred, yTrue)
	}
}

// BenchmarkLossComparison benchmarks all loss functions with same data.
func BenchmarkLossComparison(b *testing.B) {
	yPred := make([]float64, 1000)
	yTrue := make([]float64, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	mse := MSE{}
	ce := CrossEntropy{}
	huber := NewHuber(1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mse.Forward(yPred, yTrue)
		_ = ce.Forward(yPred, yTrue)
		_ = huber.Forward(yPred, yTrue)
	}
}
