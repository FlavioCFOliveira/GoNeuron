// Package opt provides benchmarks for optimizers.
package opt

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

// BenchmarkSGDStep benchmarks SGD Step method.
func BenchmarkSGDStep(b *testing.B) {
	sgd := SGD{LearningRate: 0.01}
	params := make([]float64, 1000)
	gradients := make([]float64, 1000)
	fillRandom(params)
	fillRandom(gradients)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sgd.Step(params, gradients)
	}
}

// BenchmarkSGDStepInPlace benchmarks SGD StepInPlace method.
func BenchmarkSGDStepInPlace(b *testing.B) {
	sgd := SGD{LearningRate: 0.01}
	params := make([]float64, 1000)
	gradients := make([]float64, 1000)
	fillRandom(params)
	fillRandom(gradients)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sgd.StepInPlace(params, gradients)
	}
}

// BenchmarkSGDStepSmallLR benchmarks SGD with small learning rate.
func BenchmarkSGDStepSmallLR(b *testing.B) {
	sgd := SGD{LearningRate: 0.001}
	params := make([]float64, 1000)
	gradients := make([]float64, 1000)
	fillRandom(params)
	fillRandom(gradients)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sgd.Step(params, gradients)
	}
}

// BenchmarkSGDStepLargeLR benchmarks SGD with large learning rate.
func BenchmarkSGDStepLargeLR(b *testing.B) {
	sgd := SGD{LearningRate: 0.1}
	params := make([]float64, 1000)
	gradients := make([]float64, 1000)
	fillRandom(params)
	fillRandom(gradients)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sgd.Step(params, gradients)
	}
}

// BenchmarkAdamStep benchmarks Adam Step method.
func BenchmarkAdamStep(b *testing.B) {
	adam := NewAdam(0.001)
	params := make([]float64, 1000)
	gradients := make([]float64, 1000)
	fillRandom(params)
	fillRandom(gradients)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = adam.Step(params, gradients)
	}
}

// BenchmarkAdamStepInPlace benchmarks Adam StepInPlace method.
func BenchmarkAdamStepInPlace(b *testing.B) {
	adam := NewAdam(0.001)
	params := make([]float64, 1000)
	gradients := make([]float64, 1000)
	fillRandom(params)
	fillRandom(gradients)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adam.StepInPlace(params, gradients)
	}
}

// BenchmarkSGDStepSmall benchmarks SGD with small parameter count.
func BenchmarkSGDStepSmall(b *testing.B) {
	sgd := SGD{LearningRate: 0.01}
	params := make([]float64, 10)
	gradients := make([]float64, 10)
	fillRandom(params)
	fillRandom(gradients)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sgd.Step(params, gradients)
	}
}

// BenchmarkSGDStepLarge benchmarks SGD with large parameter count.
func BenchmarkSGDStepLarge(b *testing.B) {
	sgd := SGD{LearningRate: 0.01}
	params := make([]float64, 10000)
	gradients := make([]float64, 10000)
	fillRandom(params)
	fillRandom(gradients)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sgd.Step(params, gradients)
	}
}

// BenchmarkOptimizerComparison benchmarks all optimizers with same data.
func BenchmarkOptimizerComparison(b *testing.B) {
	params := make([]float64, 1000)
	gradients := make([]float64, 1000)
	fillRandom(params)
	fillRandom(gradients)

	sgd := SGD{LearningRate: 0.01}
	adam := NewAdam(0.001)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sgd.Step(params, gradients)
		_ = adam.Step(params, gradients)
	}
}

// BenchmarkOptimizerStepInPlaceComparison benchmarks in-place updates.
func BenchmarkOptimizerStepInPlaceComparison(b *testing.B) {
	params := make([]float64, 1000)
	gradients := make([]float64, 1000)
	fillRandom(params)
	fillRandom(gradients)

	sgd := SGD{LearningRate: 0.01}
	adam := NewAdam(0.001)

	b.Run("SGD", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sgd.StepInPlace(params, gradients)
		}
	})

	b.Run("Adam", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			adam.StepInPlace(params, gradients)
		}
	})
}
