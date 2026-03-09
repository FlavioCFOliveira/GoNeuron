// Package layer provides optimized CPU operations for LSTM gates.
// These implementations use SIMD-like techniques without requiring assembly:
// - Loop unrolling for better instruction pipelining
// - Multiple accumulators to hide latency
// - Cache-friendly access patterns
package layer

// lstmGateUnrolled4 computes one gate's pre-activation using 4x unrolling.
// Computes: output[i] += sum_j(weights[i*inSize+j] * input[j])
// This is used for both input weights (x) and recurrent weights (h_prev).
//
// Parameters:
//   - weights: flattened weight matrix (gateOutSize x inSize)
//   - input: input vector (length inSize)
//   - output: output buffer (length gateOutSize) - ACCUMULATES to existing values
//   - gateOutSize: output dimension (typically outSize for LSTM gates)
//   - inSize: input dimension
//   - weightOffset: offset into weights for this specific gate
func lstmGateUnrolled4(weights, input, output []float32, gateOutSize, inSize, weightOffset int) {
	for i := 0; i < gateOutSize; i++ {
		wBase := weightOffset + i*inSize

		// Multiple accumulators for better ILP (Instruction Level Parallelism)
		var sum0, sum1, sum2, sum3 float32

		// 4x unrolled inner loop
		j := 0
		for ; j <= inSize-4; j += 4 {
			sum0 += weights[wBase+j] * input[j]
			sum1 += weights[wBase+j+1] * input[j+1]
			sum2 += weights[wBase+j+2] * input[j+2]
			sum3 += weights[wBase+j+3] * input[j+3]
		}

		// Handle remaining elements
		for ; j < inSize; j++ {
			sum0 += weights[wBase+j] * input[j]
		}

		// Accumulate to output (not overwrite, as this adds to bias)
		output[i] += sum0 + sum1 + sum2 + sum3
	}
}

// lstmGateUnrolled8 computes one gate's pre-activation using 8x unrolling.
// Better for larger matrices where we can utilize more accumulators.
//
// Parameters:
//   - weights: flattened weight matrix (gateOutSize x inSize)
//   - input: input vector (length inSize)
//   - output: output buffer (length gateOutSize) - ACCUMULATES to existing values
//   - gateOutSize: output dimension
//   - inSize: input dimension
//   - weightOffset: offset into weights for this specific gate
func lstmGateUnrolled8(weights, input, output []float32, gateOutSize, inSize, weightOffset int) {
	for i := 0; i < gateOutSize; i++ {
		wBase := weightOffset + i*inSize

		// 8 accumulators for maximum ILP
		var sum0, sum1, sum2, sum3 float32
		var sum4, sum5, sum6, sum7 float32

		// 8x unrolled inner loop
		j := 0
		for ; j <= inSize-8; j += 8 {
			sum0 += weights[wBase+j] * input[j]
			sum1 += weights[wBase+j+1] * input[j+1]
			sum2 += weights[wBase+j+2] * input[j+2]
			sum3 += weights[wBase+j+3] * input[j+3]
			sum4 += weights[wBase+j+4] * input[j+4]
			sum5 += weights[wBase+j+5] * input[j+5]
			sum6 += weights[wBase+j+6] * input[j+6]
			sum7 += weights[wBase+j+7] * input[j+7]
		}

		// Handle remaining elements with 4x unrolling if possible
		for ; j <= inSize-4; j += 4 {
			sum0 += weights[wBase+j] * input[j]
			sum1 += weights[wBase+j+1] * input[j+1]
			sum2 += weights[wBase+j+2] * input[j+2]
			sum3 += weights[wBase+j+3] * input[j+3]
		}

		// Handle final elements
		for ; j < inSize; j++ {
			sum0 += weights[wBase+j] * input[j]
		}

		// Accumulate to output
		output[i] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7
	}
}

// lstmGateSimple is the baseline implementation without unrolling.
// Used for very small matrices where unrolling overhead isn't worth it.
func lstmGateSimple(weights, input, output []float32, gateOutSize, inSize, weightOffset int) {
	for i := 0; i < gateOutSize; i++ {
		wBase := weightOffset + i*inSize
		var sum float32
		for j := 0; j < inSize; j++ {
			sum += weights[wBase+j] * input[j]
		}
		output[i] += sum
	}
}

// computeLSTMGatesOptimized computes all 4 LSTM gate pre-activations.
// This replaces the nested loops in ForwardWithArena (lines 304-328).
//
// The function computes:
//   preAct[gate*outSize:(gate+1)*outSize] += biases + W_input[gate] @ x + W_recurrent[gate] @ h_prev
//
// Parameters:
//   - inputWeights: all input weights (4*outSize x inSize, flattened)
//   - recurrentWeights: all recurrent weights (4*outSize x outSize, flattened)
//   - biases: all biases (4*outSize)
//   - x: input vector (inSize)
//   - hPrev: previous hidden state (outSize)
//   - preAct: output buffer for pre-activations (4*outSize) - initialized with biases
//   - inSize: input dimension
//   - outSize: hidden state dimension
func computeLSTMGatesOptimized(inputWeights, recurrentWeights, biases, x, hPrev, preAct []float32, inSize, outSize int) {
	// Strategy selection based on problem size
	// Thresholds tuned empirically for modern CPUs
	useUnroll8 := inSize >= 128 || outSize >= 128
	useUnroll4 := inSize >= 32 || outSize >= 32

	// Compute contributions for each of the 4 gates
	// Gate order: input, forget, cell_candidate, output
	for g := 0; g < 4; g++ {
		baseG := g * outSize
		baseW := baseG * inSize    // offset into inputWeights
		baseWR := baseG * outSize  // offset into recurrentWeights
		dst := preAct[baseG : baseG+outSize]

		// Input contribution: W_x[gate] @ x
		if useUnroll8 {
			lstmGateUnrolled8(inputWeights, x, dst, outSize, inSize, baseW)
		} else if useUnroll4 {
			lstmGateUnrolled4(inputWeights, x, dst, outSize, inSize, baseW)
		} else {
			lstmGateSimple(inputWeights, x, dst, outSize, inSize, baseW)
		}

		// Recurrent contribution: W_h[gate] @ h_prev
		if useUnroll8 {
			lstmGateUnrolled8(recurrentWeights, hPrev, dst, outSize, outSize, baseWR)
		} else if useUnroll4 {
			lstmGateUnrolled4(recurrentWeights, hPrev, dst, outSize, outSize, baseWR)
		} else {
			lstmGateSimple(recurrentWeights, hPrev, dst, outSize, outSize, baseWR)
		}
	}
}

// lstmBackwardUnrolled4 computes gradient w.r.t. input with 4x unrolling.
// Used in Backward pass for computing dx and dhPrev.
//
// Computes: output[i] = sum_j(weights[j*inSize+i] * grad[j])
// Note: weights are transposed implicitly (column-major access pattern)
func lstmBackwardUnrolled4(weights, grad, output []float32, inSize, outSize int, weightOffset int, clear bool) {
	if clear {
		// Clear output first
		for i := 0; i < inSize; i++ {
			output[i] = 0
		}
	}

	for j := 0; j < outSize; j++ {
		gj := grad[j]
		wBase := weightOffset + j*inSize

		// 4x unrolled accumulation
		i := 0
		for ; i <= inSize-4; i += 4 {
			output[i] += weights[wBase+i] * gj
			output[i+1] += weights[wBase+i+1] * gj
			output[i+2] += weights[wBase+i+2] * gj
			output[i+3] += weights[wBase+i+3] * gj
		}

		// Handle remaining elements
		for ; i < inSize; i++ {
			output[i] += weights[wBase+i] * gj
		}
	}
}

// lstmBackwardUnrolled8 computes gradient w.r.t. input with 8x unrolling.
func lstmBackwardUnrolled8(weights, grad, output []float32, inSize, outSize int, weightOffset int, clear bool) {
	if clear {
		// Clear output first
		for i := 0; i < inSize; i++ {
			output[i] = 0
		}
	}

	for j := 0; j < outSize; j++ {
		gj := grad[j]
		wBase := weightOffset + j*inSize

		// 8x unrolled accumulation
		i := 0
		for ; i <= inSize-8; i += 8 {
			output[i] += weights[wBase+i] * gj
			output[i+1] += weights[wBase+i+1] * gj
			output[i+2] += weights[wBase+i+2] * gj
			output[i+3] += weights[wBase+i+3] * gj
			output[i+4] += weights[wBase+i+4] * gj
			output[i+5] += weights[wBase+i+5] * gj
			output[i+6] += weights[wBase+i+6] * gj
			output[i+7] += weights[wBase+i+7] * gj
		}

		// Handle remaining elements with 4x unrolling
		for ; i <= inSize-4; i += 4 {
			output[i] += weights[wBase+i] * gj
			output[i+1] += weights[wBase+i+1] * gj
			output[i+2] += weights[wBase+i+2] * gj
			output[i+3] += weights[wBase+i+3] * gj
		}

		// Handle final elements
		for ; i < inSize; i++ {
			output[i] += weights[wBase+i] * gj
		}
	}
}

// computeLSTMBackwardInputOptimized computes dx = sum_gates(W_x[gate]^T @ d_gate).
// This replaces the triple-nested loop in Backward (lines 610-628).
func computeLSTMBackwardInputOptimized(inputWeights, dInput, dForget, dCell, dOutput, dx []float32, inSize, outSize int) {
	// Strategy selection
	useUnroll8 := inSize >= 128 || outSize >= 128
	useUnroll4 := inSize >= 32 || outSize >= 32

	dGates := [][]float32{dInput, dForget, dCell, dOutput}

	// Clear dx first
	for i := 0; i < inSize; i++ {
		dx[i] = 0
	}

	// Accumulate contributions from all 4 gates
	for g := 0; g < 4; g++ {
		baseW := g * outSize * inSize

		if useUnroll8 {
			lstmBackwardUnrolled8(inputWeights, dGates[g], dx, inSize, outSize, baseW, false)
		} else if useUnroll4 {
			lstmBackwardUnrolled4(inputWeights, dGates[g], dx, inSize, outSize, baseW, false)
		} else {
			// Simple implementation
			for j := 0; j < outSize; j++ {
				gj := dGates[g][j]
				wBase := baseW + j*inSize
				for i := 0; i < inSize; i++ {
					dx[i] += inputWeights[wBase+i] * gj
				}
			}
		}
	}
}

// computeLSTMBackwardHiddenOptimized computes dhPrev = sum_gates(W_h[gate]^T @ d_gate).
// This replaces the triple-nested loop in Backward (lines 633-651).
func computeLSTMBackwardHiddenOptimized(recurrentWeights, dInput, dForget, dCell, dOutput, dhPrev []float32, outSize int) {
	// Strategy selection
	useUnroll8 := outSize >= 128
	useUnroll4 := outSize >= 32

	dGates := [][]float32{dInput, dForget, dCell, dOutput}

	// Clear dhPrev first
	for i := 0; i < outSize; i++ {
		dhPrev[i] = 0
	}

	// Accumulate contributions from all 4 gates
	for g := 0; g < 4; g++ {
		baseW := g * outSize * outSize

		if useUnroll8 {
			lstmBackwardUnrolled8(recurrentWeights, dGates[g], dhPrev, outSize, outSize, baseW, false)
		} else if useUnroll4 {
			lstmBackwardUnrolled4(recurrentWeights, dGates[g], dhPrev, outSize, outSize, baseW, false)
		} else {
			// Simple implementation
			for j := 0; j < outSize; j++ {
				gj := dGates[g][j]
				wBase := baseW + j*outSize
				for i := 0; i < outSize; i++ {
					dhPrev[i] += recurrentWeights[wBase+i] * gj
				}
			}
		}
	}
}

// lstmWeightGradUnrolled4 computes gradient w.r.t. weights with 4x unrolling.
// gradW[i*inSize+j] += dGate[i] * input[j]
func lstmWeightGradUnrolled4(dGate, input, gradW []float32, outSize, inSize int, gradOffset int) {
	for i := 0; i < outSize; i++ {
		di := dGate[i]
		wBase := gradOffset + i*inSize

		// 4x unrolled accumulation
		j := 0
		for ; j <= inSize-4; j += 4 {
			gradW[wBase+j] += di * input[j]
			gradW[wBase+j+1] += di * input[j+1]
			gradW[wBase+j+2] += di * input[j+2]
			gradW[wBase+j+3] += di * input[j+3]
		}

		// Handle remaining elements
		for ; j < inSize; j++ {
			gradW[wBase+j] += di * input[j]
		}
	}
}

// lstmWeightGradUnrolled8 computes gradient w.r.t. weights with 8x unrolling.
func lstmWeightGradUnrolled8(dGate, input, gradW []float32, outSize, inSize int, gradOffset int) {
	for i := 0; i < outSize; i++ {
		di := dGate[i]
		wBase := gradOffset + i*inSize

		// 8x unrolled accumulation
		j := 0
		for ; j <= inSize-8; j += 8 {
			gradW[wBase+j] += di * input[j]
			gradW[wBase+j+1] += di * input[j+1]
			gradW[wBase+j+2] += di * input[j+2]
			gradW[wBase+j+3] += di * input[j+3]
			gradW[wBase+j+4] += di * input[j+4]
			gradW[wBase+j+5] += di * input[j+5]
			gradW[wBase+j+6] += di * input[j+6]
			gradW[wBase+j+7] += di * input[j+7]
		}

		// Handle remaining elements with 4x unrolling
		for ; j <= inSize-4; j += 4 {
			gradW[wBase+j] += di * input[j]
			gradW[wBase+j+1] += di * input[j+1]
			gradW[wBase+j+2] += di * input[j+2]
			gradW[wBase+j+3] += di * input[j+3]
		}

		// Handle final elements
		for ; j < inSize; j++ {
			gradW[wBase+j] += di * input[j]
		}
	}
}

// computeLSTMWeightGradsOptimized computes gradients for all LSTM weights.
// This optimizes the loops in Backward (lines 557-605).
func computeLSTMWeightGradsOptimized(inputBuf, hPrev []float32,
	dInput, dForget, dCell, dOutput []float32,
	gradInputWeights, gradRecurrentWeights, gradBiases []float32,
	inSize, outSize int) {

	// Strategy selection
	useUnroll8 := inSize >= 128 || outSize >= 128
	useUnroll4 := inSize >= 32 || outSize >= 32

	dGates := [][]float32{dInput, dForget, dCell, dOutput}

	// Input weight gradients for all 4 gates
	for g := 0; g < 4; g++ {
		baseG := g * outSize
		baseW := baseG * inSize
		dg := dGates[g]

		// Accumulate gradW += d_gate @ x^T
		if useUnroll8 {
			lstmWeightGradUnrolled8(dg, inputBuf, gradInputWeights, outSize, inSize, baseW)
		} else if useUnroll4 {
			lstmWeightGradUnrolled4(dg, inputBuf, gradInputWeights, outSize, inSize, baseW)
		} else {
			// Simple implementation
			for i := 0; i < outSize; i++ {
				di := dg[i]
				wBase := baseW + i*inSize
				for j := 0; j < inSize; j++ {
					gradInputWeights[wBase+j] += di * inputBuf[j]
				}
			}
		}

		// Accumulate bias gradients
		for i := 0; i < outSize; i++ {
			gradBiases[baseG+i] += dg[i]
		}
	}

	// Recurrent weight gradients for all 4 gates
	for g := 0; g < 4; g++ {
		baseG := g * outSize
		baseW := baseG * outSize
		dg := dGates[g]

		// Accumulate gradW += d_gate @ h_prev^T
		if useUnroll8 {
			lstmWeightGradUnrolled8(dg, hPrev, gradRecurrentWeights, outSize, outSize, baseW)
		} else if useUnroll4 {
			lstmWeightGradUnrolled4(dg, hPrev, gradRecurrentWeights, outSize, outSize, baseW)
		} else {
			// Simple implementation
			for i := 0; i < outSize; i++ {
				di := dg[i]
				wBase := baseW + i*outSize
				for j := 0; j < outSize; j++ {
					gradRecurrentWeights[wBase+j] += di * hPrev[j]
				}
			}
		}
	}
}

// SelectLSTMImplementation chooses the best implementation based on problem size.
// Returns: "simple", "unroll4", or "unroll8"
func SelectLSTMImplementation(inSize, outSize int) string {
	if inSize >= 128 || outSize >= 128 {
		return "unroll8"
	} else if inSize >= 32 || outSize >= 32 {
		return "unroll4"
	}
	return "simple"
}

// GetLSTMPerformanceMetrics returns performance characteristics for LSTM dimensions.
func GetLSTMPerformanceMetrics(inSize, outSize int) map[string]interface{} {
	metrics := make(map[string]interface{})

	// Problem characteristics
	// Each gate computation: outSize * inSize multiplications + (outSize * (inSize-1)) additions
	// Total for 4 gates: 4 * outSize * (2*inSize - 1) FLOPs per timestep
	flopsPerGate := outSize * (2*inSize - 1)
	totalFLOPs := 4 * flopsPerGate

	// Memory access pattern
	// Input: inSize (reused for 4 gates - cache friendly)
	// Weights: 4 * outSize * inSize (read sequentially)
	// Output: 4 * outSize (write sequentially)
	memoryBytes := (inSize + 4*outSize*inSize + 4*outSize) * 4

	metrics["inSize"] = inSize
	metrics["outSize"] = outSize
	metrics["flops_per_timestep"] = totalFLOPs
	metrics["memory_bytes"] = memoryBytes
	metrics["arithmetic_intensity"] = float64(totalFLOPs) / float64(memoryBytes)
	metrics["recommended_implementation"] = SelectLSTMImplementation(inSize, outSize)

	return metrics
}
