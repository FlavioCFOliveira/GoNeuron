// Package loss provides benchmarks for loss functions.
package loss

import (
	"math/rand"
	"testing"
)

// fillRandom fills a slice with random values.
func fillRandom(slice []float32) {
	for i := range slice {
		slice[i] = rand.Float32()
	}
}

// BenchmarkMSEForward benchmarks MSE loss forward pass.
func BenchmarkMSEForward(b *testing.B) {
	mse := MSE{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	grad := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
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

// BenchmarkL1LossForward benchmarks L1Loss forward pass.
func BenchmarkL1LossForward(b *testing.B) {
	l1 := L1Loss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l1.Forward(yPred, yTrue)
	}
}

// BenchmarkL1LossBackward benchmarks L1Loss backward pass.
func BenchmarkL1LossBackward(b *testing.B) {
	l1 := L1Loss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l1.Backward(yPred, yTrue)
	}
}

// BenchmarkL1LossFull benchmarks complete L1Loss forward and backward pass.
func BenchmarkL1LossFull(b *testing.B) {
	l1 := L1Loss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l1.Forward(yPred, yTrue)
		_ = l1.Backward(yPred, yTrue)
	}
}

// BenchmarkBCELossForward benchmarks BCELoss forward pass.
func BenchmarkBCELossForward(b *testing.B) {
	bce := BCELoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	// Clamp predictions to valid range for BCE
	for i := range yPred {
		if yPred[i] < 0.01 {
			yPred[i] = 0.01
		}
		if yPred[i] > 0.99 {
			yPred[i] = 0.99
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bce.Forward(yPred, yTrue)
	}
}

// BenchmarkBCELossBackward benchmarks BCELoss backward pass.
func BenchmarkBCELossBackward(b *testing.B) {
	bce := BCELoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	for i := range yPred {
		if yPred[i] < 0.01 {
			yPred[i] = 0.01
		}
		if yPred[i] > 0.99 {
			yPred[i] = 0.99
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bce.Backward(yPred, yTrue)
	}
}

// BenchmarkBCELossFull benchmarks complete BCELoss forward and backward pass.
func BenchmarkBCELossFull(b *testing.B) {
	bce := BCELoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	for i := range yPred {
		if yPred[i] < 0.01 {
			yPred[i] = 0.01
		}
		if yPred[i] > 0.99 {
			yPred[i] = 0.99
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bce.Forward(yPred, yTrue)
		_ = bce.Backward(yPred, yTrue)
	}
}

// BenchmarkBCEWithLogitsLossForward benchmarks BCEWithLogitsLoss forward pass.
func BenchmarkBCEWithLogitsLossForward(b *testing.B) {
	bce := BCEWithLogitsLoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bce.Forward(yPred, yTrue)
	}
}

// BenchmarkBCEWithLogitsLossBackward benchmarks BCEWithLogitsLoss backward pass.
func BenchmarkBCEWithLogitsLossBackward(b *testing.B) {
	bce := BCEWithLogitsLoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bce.Backward(yPred, yTrue)
	}
}

// BenchmarkBCEWithLogitsLossFull benchmarks complete BCEWithLogitsLoss forward and backward pass.
func BenchmarkBCEWithLogitsLossFull(b *testing.B) {
	bce := BCEWithLogitsLoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bce.Forward(yPred, yTrue)
		_ = bce.Backward(yPred, yTrue)
	}
}

// BenchmarkNLLLossForward benchmarks NLLLoss forward pass.
func BenchmarkNLLLossForward(b *testing.B) {
	nll := NLLLoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	// Normalize to make valid probabilities (100 groups of 10)
	for i := 0; i < 100; i++ {
		sum := float32(0.0)
		for j := i * 10; j < (i+1)*10; j++ {
			sum += yPred[j]
		}
		for j := i * 10; j < (i+1)*10; j++ {
			yPred[j] /= sum
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nll.Forward(yPred, yTrue)
	}
}

// BenchmarkNLLLossBackward benchmarks NLLLoss backward pass.
func BenchmarkNLLLossBackward(b *testing.B) {
	nll := NLLLoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	// Normalize to make valid probabilities (100 groups of 10)
	for i := 0; i < 100; i++ {
		sum := float32(0.0)
		for j := i * 10; j < (i+1)*10; j++ {
			sum += yPred[j]
		}
		for j := i * 10; j < (i+1)*10; j++ {
			yPred[j] /= sum
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nll.Backward(yPred, yTrue)
	}
}

// BenchmarkNLLLossFull benchmarks complete NLLLoss forward and backward pass.
func BenchmarkNLLLossFull(b *testing.B) {
	nll := NLLLoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	// Normalize to make valid probabilities (100 groups of 10)
	for i := 0; i < 100; i++ {
		sum := float32(0.0)
		for j := i * 10; j < (i+1)*10; j++ {
			sum += yPred[j]
		}
		for j := i * 10; j < (i+1)*10; j++ {
			yPred[j] /= sum
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nll.Forward(yPred, yTrue)
		_ = nll.Backward(yPred, yTrue)
	}
}

// BenchmarkCosineEmbeddingLossForward benchmarks CosineEmbeddingLoss forward pass.
func BenchmarkCosineEmbeddingLossForward(b *testing.B) {
	l := NewCosineEmbeddingLoss(0.0)
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l.Forward(yPred, yTrue)
	}
}

// BenchmarkCosineEmbeddingLossBackward benchmarks CosineEmbeddingLoss backward pass.
func BenchmarkCosineEmbeddingLossBackward(b *testing.B) {
	l := NewCosineEmbeddingLoss(0.0)
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l.Backward(yPred, yTrue)
	}
}

// BenchmarkCosineEmbeddingLossFull benchmarks complete CosineEmbeddingLoss forward and backward pass.
func BenchmarkCosineEmbeddingLossFull(b *testing.B) {
	l := NewCosineEmbeddingLoss(0.0)
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l.Forward(yPred, yTrue)
		_ = l.Backward(yPred, yTrue)
	}
}

// BenchmarkMarginRankingLossForward benchmarks MarginRankingLoss forward pass.
func BenchmarkMarginRankingLossForward(b *testing.B) {
	l := NewMarginRankingLoss(1.0)
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 500)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l.Forward(yPred, yTrue)
	}
}

// BenchmarkMarginRankingLossBackward benchmarks MarginRankingLoss backward pass.
func BenchmarkMarginRankingLossBackward(b *testing.B) {
	l := NewMarginRankingLoss(1.0)
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 500)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l.Backward(yPred, yTrue)
	}
}

// BenchmarkMarginRankingLossFull benchmarks complete MarginRankingLoss forward and backward pass.
func BenchmarkMarginRankingLossFull(b *testing.B) {
	l := NewMarginRankingLoss(1.0)
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 500)
	fillRandom(yPred)
	fillRandom(yTrue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l.Forward(yPred, yTrue)
		_ = l.Backward(yPred, yTrue)
	}
}

// BenchmarkKLDivLossForward benchmarks KLDivLoss forward pass.
func BenchmarkKLDivLossForward(b *testing.B) {
	kld := KLDivLoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	// Normalize to make valid probabilities (100 groups of 10)
	for i := 0; i < 100; i++ {
		sum := float32(0.0)
		for j := i * 10; j < (i+1)*10; j++ {
			sum += yPred[j]
		}
		for j := i * 10; j < (i+1)*10; j++ {
			yPred[j] /= sum
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = kld.Forward(yPred, yTrue)
	}
}

// BenchmarkKLDivLossBackward benchmarks KLDivLoss backward pass.
func BenchmarkKLDivLossBackward(b *testing.B) {
	kld := KLDivLoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	// Normalize to make valid probabilities (100 groups of 10)
	for i := 0; i < 100; i++ {
		sum := float32(0.0)
		for j := i * 10; j < (i+1)*10; j++ {
			sum += yPred[j]
		}
		for j := i * 10; j < (i+1)*10; j++ {
			yPred[j] /= sum
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = kld.Backward(yPred, yTrue)
	}
}

// BenchmarkKLDivLossFull benchmarks complete KLDivLoss forward and backward pass.
func BenchmarkKLDivLossFull(b *testing.B) {
	kld := KLDivLoss{}
	yPred := make([]float32, 1000)
	yTrue := make([]float32, 1000)
	fillRandom(yPred)
	fillRandom(yTrue)

	// Normalize to make valid probabilities (100 groups of 10)
	for i := 0; i < 100; i++ {
		sum := float32(0.0)
		for j := i * 10; j < (i+1)*10; j++ {
			sum += yPred[j]
		}
		for j := i * 10; j < (i+1)*10; j++ {
			yPred[j] /= sum
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = kld.Forward(yPred, yTrue)
		_ = kld.Backward(yPred, yTrue)
	}
}
