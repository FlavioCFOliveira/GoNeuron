package layer

import (
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"math"
)

// Attention implements a scaled dot-product attention mechanism.
// It computes Attention(Q, K, V) = Softmax(Q*K^T / sqrt(d_k)) * V
type Attention struct {
	inSize  int
	outSize int

	// Parameters (Query, Key, Value weights)
	// We implement self-attention where Q, K, V are derived from the same input.
	// Weights are [outSize * inSize]
	queryWeights []float64
	keyWeights   []float64
	valueWeights []float64

	// Buffers for forward pass
	inputBuf   []float64
	qBuf       []float64 // [outSize]
	kBuf       []float64 // [outSize]
	vBuf       []float64 // [outSize]
	scoresBuf  []float64 // [current_sequence_length]
	outputBuf  []float64 // [outSize]

	// Sequence state
	storedInputs [][]float64
	storedQueries [][]float64
	storedKeys    [][]float64
	storedValues  [][]float64
	storedScores  [][]float64 // Softmax weights

	// Gradients
	gradQueryW []float64
	gradKeyW   []float64
	gradValueW []float64
	gradInBuf  []float64
}

// NewAttention creates a new Attention layer.
func NewAttention(inSize, outSize int) *Attention {
	rng := NewRNG(uint64(inSize*100 + outSize + 42))
	scale := math.Sqrt(1.0 / float64(inSize))

	queryWeights := make([]float64, outSize*inSize)
	keyWeights := make([]float64, outSize*inSize)
	valueWeights := make([]float64, outSize*inSize)

	for i := range queryWeights {
		queryWeights[i] = rng.RandFloat()*2*scale - scale
		keyWeights[i] = rng.RandFloat()*2*scale - scale
		valueWeights[i] = rng.RandFloat()*2*scale - scale
	}

	return &Attention{
		inSize:       inSize,
		outSize:      outSize,
		queryWeights: queryWeights,
		keyWeights:   keyWeights,
		valueWeights: valueWeights,
		inputBuf:     make([]float64, inSize),
		qBuf:         make([]float64, outSize),
		kBuf:         make([]float64, outSize),
		vBuf:         make([]float64, outSize),
		outputBuf:    make([]float64, outSize),
		gradQueryW:   make([]float64, outSize*inSize),
		gradKeyW:     make([]float64, outSize*inSize),
		gradValueW:   make([]float64, outSize*inSize),
		gradInBuf:    make([]float64, inSize),
		storedInputs: make([][]float64, 0),
		storedQueries: make([][]float64, 0),
		storedKeys:    make([][]float64, 0),
		storedValues:  make([][]float64, 0),
		storedScores:  make([][]float64, 0),
	}
}

// Forward performs a forward pass for one time step.
func (a *Attention) Forward(x []float64) []float64 {
	// 1. Transform input to Q, K, V
	q := make([]float64, a.outSize)
	k := make([]float64, a.outSize)
	v := make([]float64, a.outSize)

	for o := 0; o < a.outSize; o++ {
		sumQ, sumK, sumV := 0.0, 0.0, 0.0
		base := o * a.inSize
		for i := 0; i < a.inSize; i++ {
			val := x[i]
			sumQ += a.queryWeights[base+i] * val
			sumK += a.keyWeights[base+i] * val
			sumV += a.valueWeights[base+i] * val
		}
		q[o] = sumQ
		k[o] = sumK
		v[o] = sumV
	}

	// 2. Store states
	xCopy := make([]float64, len(x))
	copy(xCopy, x)
	a.storedInputs = append(a.storedInputs, xCopy)
	a.storedQueries = append(a.storedQueries, q)
	a.storedKeys = append(a.storedKeys, k)
	a.storedValues = append(a.storedValues, v)

	// 3. Compute Attention Scores: Softmax(Q_t * K_prev^T / sqrt(d_k))
	seqLen := len(a.storedKeys)
	scores := make([]float64, seqLen)
	scale := 1.0 / math.Sqrt(float64(a.outSize))

	for i := 0; i < seqLen; i++ {
		dot := 0.0
		ki := a.storedKeys[i]
		for d := 0; d < a.outSize; d++ {
			dot += q[d] * ki[d]
		}
		scores[i] = dot * scale
	}

	// Apply Softmax
	softmax := activations.Softmax{}
	scores = softmax.ActivateBatch(scores)
	a.storedScores = append(a.storedScores, scores)

	// 4. Compute context vector: sum(scores * V)
	context := make([]float64, a.outSize)
	for i := 0; i < seqLen; i++ {
		vi := a.storedValues[i]
		s := scores[i]
		for d := 0; d < a.outSize; d++ {
			context[d] += s * vi[d]
		}
	}

	copy(a.outputBuf, context)
	return a.outputBuf
}

// Backward performs a backward pass.
func (a *Attention) Backward(grad []float64) []float64 {
	// Simplified backward pass for the most recent timestep
	t := len(a.storedInputs) - 1
	if t < 0 {
		return nil
	}

	q := a.storedQueries[t]
	scores := a.storedScores[t]
	gradIn := make([]float64, a.inSize)
	scale := 1.0 / math.Sqrt(float64(a.outSize))

	// Gradients w.r.t context vector components
	// dL/dcontext is grad

	// 1. Gradients w.r.t Values and Scores
	gradV_t_all := make([][]float64, t+1)
	gradScores := make([]float64, t+1)
	for i := 0; i <= t; i++ {
		gradV_t_all[i] = make([]float64, a.outSize)
		s := scores[i]
		vi := a.storedValues[i]
		for d := 0; d < a.outSize; d++ {
			// dL/dvi = dL/dcontext * dcontext/dvi = grad * s
			gradV_t_all[i][d] = grad[d] * s
			// dL/ds = sum(dL/dcontext * dcontext/ds) = sum(grad * vi)
			gradScores[i] += grad[d] * vi[d]
		}
	}

	// 2. Gradients through Softmax
	// Simplified: assuming Softmax derivative for one-hot like behavior
	// In reality, we'd need full Jacobian.
	gradPreSoftmax := make([]float64, t+1)
	for i := 0; i <= t; i++ {
		sum := 0.0
		for j := 0; j <= t; j++ {
			delta := 0.0
			if i == j {
				delta = 1.0
			}
			sum += gradScores[j] * scores[j] * (delta - scores[i])
		}
		gradPreSoftmax[i] = sum
	}

	// 3. Gradients w.r.t Q and K
	gradQ := make([]float64, a.outSize)
	gradK_all := make([][]float64, t+1)
	for i := 0; i <= t; i++ {
		gradK_all[i] = make([]float64, a.outSize)
		ki := a.storedKeys[i]
		gps := gradPreSoftmax[i] * scale
		for d := 0; d < a.outSize; d++ {
			// dL/dQ = sum_i(dL/dScore_i * dScore_i/dQ) = sum_i(gps * ki)
			gradQ[d] += gps * ki[d]
			// dL/dKi = dL/dScore_i * dScore_i/dKi = gps * Q
			gradK_all[i][d] = gps * q[d]
		}
	}

	// 4. Gradients w.r.t Weights and Input
	input := a.storedInputs[t]
	for o := 0; o < a.outSize; o++ {
		gq := gradQ[o]
		gv := gradV_t_all[t][o] // Only current timestep V contributes to weights here for simplicity
		gk := gradK_all[t][o]
		base := o * a.inSize
		for i := 0; i < a.inSize; i++ {
			val := input[i]
			a.gradQueryW[base+i] += gq * val
			a.gradKeyW[base+i] += gk * val
			a.gradValueW[base+i] += gv * val

			gradIn[i] += a.queryWeights[base+i] * gq
			gradIn[i] += a.keyWeights[base+i] * gk
			gradIn[i] += a.valueWeights[base+i] * gv
		}
	}

	copy(a.gradInBuf, gradIn)
	return a.gradInBuf
}

// AccumulateBackward (stub)
func (a *Attention) AccumulateBackward(grad []float64) []float64 {
	return a.Backward(grad)
}

// Reset resets the attention state.
func (a *Attention) Reset() {
	a.storedInputs = a.storedInputs[:0]
	a.storedQueries = a.storedQueries[:0]
	a.storedKeys = a.storedKeys[:0]
	a.storedValues = a.storedValues[:0]
	a.storedScores = a.storedScores[:0]
}

// Params returns all parameters.
func (a *Attention) Params() []float64 {
	total := len(a.queryWeights) + len(a.keyWeights) + len(a.valueWeights)
	params := make([]float64, total)
	n := copy(params, a.queryWeights)
	n += copy(params[n:], a.keyWeights)
	copy(params[n:], a.valueWeights)
	return params
}

// SetParams sets parameters.
func (a *Attention) SetParams(params []float64) {
	n := copy(a.queryWeights, params)
	n += copy(a.keyWeights, params[n:])
	copy(a.valueWeights, params[n:])
}

// Gradients returns all gradients.
func (a *Attention) Gradients() []float64 {
	total := len(a.gradQueryW) + len(a.gradKeyW) + len(a.gradValueW)
	grads := make([]float64, total)
	n := copy(grads, a.gradQueryW)
	n += copy(grads[n:], a.gradKeyW)
	copy(grads[n:], a.gradValueW)
	return grads
}

// SetGradients sets gradients.
func (a *Attention) SetGradients(grads []float64) {
	n := copy(a.gradQueryW, grads)
	n += copy(a.gradKeyW, grads[n:])
	copy(a.gradValueW, grads[n:])
}

// ClearGradients clears gradients.
func (a *Attention) ClearGradients() {
	for i := range a.gradQueryW {
		a.gradQueryW[i] = 0
		a.gradKeyW[i] = 0
		a.gradValueW[i] = 0
	}
}

// InSize returns input size.
func (a *Attention) InSize() int {
	return a.inSize
}

// OutSize returns output size.
func (a *Attention) OutSize() int {
	return a.outSize
}

// Clone creates a deep copy.
func (a *Attention) Clone() Layer {
	newA := NewAttention(a.inSize, a.outSize)
	copy(newA.queryWeights, a.queryWeights)
	copy(newA.keyWeights, a.keyWeights)
	copy(newA.valueWeights, a.valueWeights)
	return newA
}
