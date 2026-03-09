package layer

import (
	"math/rand"
	"testing"
)

// Benchmark para comparar computeGradW original vs blocked
func BenchmarkComputeGradW(b *testing.B) {
	sizes := []struct {
		name      string
		batchSize int
		inSize    int
		outSize   int
	}{
		{"Small_32x256x128", 32, 256, 128},
		{"Medium_64x512x256", 64, 512, 256},
		{"Large_128x1024x512", 128, 1024, 512},
		{"XLarge_256x2048x1024", 256, 2048, 1024},
	}

	for _, sz := range sizes {
		b.Run(sz.name+"_Original", func(b *testing.B) {
			input := make([]float32, sz.batchSize*sz.inSize)
			gradOutput := make([]float32, sz.batchSize*sz.outSize)
			gradW := make([]float32, sz.outSize*sz.inSize)

			// Initialize with random data
			for i := range input {
				input[i] = rand.Float32()
			}
			for i := range gradOutput {
				gradOutput[i] = rand.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Clear gradW
				for j := range gradW {
					gradW[j] = 0
				}
				computeGradWOriginal(input, gradOutput, gradW, sz.batchSize, sz.inSize, sz.outSize)
			}
		})

		b.Run(sz.name+"_Blocked64", func(b *testing.B) {
			input := make([]float32, sz.batchSize*sz.inSize)
			gradOutput := make([]float32, sz.batchSize*sz.outSize)
			gradW := make([]float32, sz.outSize*sz.inSize)

			for i := range input {
				input[i] = rand.Float32()
			}
			for i := range gradOutput {
				gradOutput[i] = rand.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := range gradW {
					gradW[j] = 0
				}
				computeGradWBlocked(input, gradOutput, gradW, sz.batchSize, sz.inSize, sz.outSize, 64)
			}
		})

		b.Run(sz.name+"_Blocked128", func(b *testing.B) {
			input := make([]float32, sz.batchSize*sz.inSize)
			gradOutput := make([]float32, sz.batchSize*sz.outSize)
			gradW := make([]float32, sz.outSize*sz.inSize)

			for i := range input {
				input[i] = rand.Float32()
			}
			for i := range gradOutput {
				gradOutput[i] = rand.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := range gradW {
					gradW[j] = 0
				}
				computeGradWBlocked(input, gradOutput, gradW, sz.batchSize, sz.inSize, sz.outSize, 128)
			}
		})

		b.Run(sz.name+"_Blocked32", func(b *testing.B) {
			input := make([]float32, sz.batchSize*sz.inSize)
			gradOutput := make([]float32, sz.batchSize*sz.outSize)
			gradW := make([]float32, sz.outSize*sz.inSize)

			for i := range input {
				input[i] = rand.Float32()
			}
			for i := range gradOutput {
				gradOutput[i] = rand.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := range gradW {
					gradW[j] = 0
				}
				computeGradWBlocked(input, gradOutput, gradW, sz.batchSize, sz.inSize, sz.outSize, 32)
			}
		})

		b.Run(sz.name+"_BlockedBatchFirst", func(b *testing.B) {
			input := make([]float32, sz.batchSize*sz.inSize)
			gradOutput := make([]float32, sz.batchSize*sz.outSize)
			gradW := make([]float32, sz.outSize*sz.inSize)

			for i := range input {
				input[i] = rand.Float32()
			}
			for i := range gradOutput {
				gradOutput[i] = rand.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := range gradW {
					gradW[j] = 0
				}
				computeGradWBlockedBatchFirst(input, gradOutput, gradW, sz.batchSize, sz.inSize, sz.outSize)
			}
		})
	}
}

// computeGradWOriginal mantém a implementação atual sem blocking
func computeGradWOriginal(input, gradOutput, gradW []float32, batchSize, inSize, outSize int) {
	for b := 0; b < batchSize; b++ {
		inOffset := b * inSize
		gradOutOffset := b * outSize

		for o := 0; o < outSize; o++ {
			gradOut := gradOutput[gradOutOffset+o]
			wOffset := o * inSize

			i := 0
			for ; i < inSize-3; i += 4 {
				gradW[wOffset+i] += gradOut * input[inOffset+i]
				gradW[wOffset+i+1] += gradOut * input[inOffset+i+1]
				gradW[wOffset+i+2] += gradOut * input[inOffset+i+2]
				gradW[wOffset+i+3] += gradOut * input[inOffset+i+3]
			}
			for ; i < inSize; i++ {
				gradW[wOffset+i] += gradOut * input[inOffset+i]
			}
		}
	}
}

// computeGradWBlocked implementa blocking configurável
func computeGradWBlocked(input, gradOutput, gradW []float32, batchSize, inSize, outSize, blockSize int) {
	for b := 0; b < batchSize; b++ {
		inOffset := b * inSize
		gradOutOffset := b * outSize

		for oo := 0; oo < outSize; oo += blockSize {
			endO := oo + blockSize
			if endO > outSize {
				endO = outSize
			}

			for ii := 0; ii < inSize; ii += blockSize {
				endI := ii + blockSize
				if endI > inSize {
					endI = inSize
				}

				for o := oo; o < endO; o++ {
					gradOut := gradOutput[gradOutOffset+o]
					wOffset := o * inSize

					i := ii
					for ; i < endI-3; i += 4 {
						gradW[wOffset+i] += gradOut * input[inOffset+i]
						gradW[wOffset+i+1] += gradOut * input[inOffset+i+1]
						gradW[wOffset+i+2] += gradOut * input[inOffset+i+2]
						gradW[wOffset+i+3] += gradOut * input[inOffset+i+3]
					}
					for ; i < endI; i++ {
						gradW[wOffset+i] += gradOut * input[inOffset+i]
					}
				}
			}
		}
	}
}

// computeGradWBlockedBatchFirst muda a ordem dos loops para melhorar cache
// Processa todos os batches para cada bloco antes de ir para o próximo bloco
func computeGradWBlockedBatchFirst(input, gradOutput, gradW []float32, batchSize, inSize, outSize int) {
	const blockSize = 64

	for oo := 0; oo < outSize; oo += blockSize {
		endO := oo + blockSize
		if endO > outSize {
			endO = outSize
		}

		for ii := 0; ii < inSize; ii += blockSize {
			endI := ii + blockSize
			if endI > inSize {
				endI = inSize
			}

			for b := 0; b < batchSize; b++ {
				inOffset := b * inSize
				gradOutOffset := b * outSize

				for o := oo; o < endO; o++ {
					gradOut := gradOutput[gradOutOffset+o]
					wOffset := o * inSize

					i := ii
					for ; i < endI-3; i += 4 {
						gradW[wOffset+i] += gradOut * input[inOffset+i]
						gradW[wOffset+i+1] += gradOut * input[inOffset+i+1]
						gradW[wOffset+i+2] += gradOut * input[inOffset+i+2]
						gradW[wOffset+i+3] += gradOut * input[inOffset+i+3]
					}
					for ; i < endI; i++ {
						gradW[wOffset+i] += gradOut * input[inOffset+i]
					}
				}
			}
		}
	}
}
