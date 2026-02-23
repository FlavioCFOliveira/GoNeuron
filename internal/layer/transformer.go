// Package layer provides neural network layer implementations.
package layer

import (
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"math"
)

// PositionalEncoding adds positional information to embeddings.
// This implementation uses learned positional embeddings.
type PositionalEncoding struct {
	seqLen int
	dim    int

	// Parameters: [seqLen, dim]
	weights []float32

	// Buffers
	outputBuf []float32
	gradBuf   []float32

	device Device
}

func NewPositionalEncoding(seqLen, dim int) *PositionalEncoding {
	weights := make([]float32, seqLen*dim)
	rng := NewRNG(uint64(seqLen + dim + 42))
	scale := float32(math.Sqrt(1.0 / float64(dim)))
	for i := range weights {
		weights[i] = rng.RandFloat()*2*scale - scale
	}

	return &PositionalEncoding{
		seqLen:    seqLen,
		dim:       dim,
		weights:   weights,
		outputBuf: make([]float32, seqLen*dim),
		gradBuf:   make([]float32, seqLen*dim),
		device:    &CPUDevice{},
	}
}

func (p *PositionalEncoding) Forward(x []float32) []float32 {
	// x is [seqLen * dim]
	if len(p.outputBuf) != len(x) {
		p.outputBuf = make([]float32, len(x))
	}
	for i := range x {
		p.outputBuf[i] = x[i] + p.weights[i]
	}
	return p.outputBuf
}

func (p *PositionalEncoding) Backward(grad []float32) []float32 {
	// Gradient w.r.t. parameters is same as gradient w.r.t. output
	// Gradient w.r.t. input is also same as gradient w.r.t. output
	for i := range grad {
		p.gradBuf[i] += grad[i]
	}
	return grad
}

func (p *PositionalEncoding) Params() []float32         { return p.weights }
func (p *PositionalEncoding) SetParams(params []float32) { copy(p.weights, params) }
func (p *PositionalEncoding) Gradients() []float32      { return p.gradBuf }
func (p *PositionalEncoding) SetGradients(g []float32)   { copy(p.gradBuf, g) }
func (p *PositionalEncoding) SetDevice(d Device)         { p.device = d }
func (p *PositionalEncoding) InSize() int                { return p.seqLen * p.dim }
func (p *PositionalEncoding) OutSize() int               { return p.seqLen * p.dim }
func (p *PositionalEncoding) Reset()                    {}
func (p *PositionalEncoding) ClearGradients()           { for i := range p.gradBuf { p.gradBuf[i] = 0 } }
func (p *PositionalEncoding) Clone() Layer {
	newP := NewPositionalEncoding(p.seqLen, p.dim)
	copy(newP.weights, p.weights)
	return newP
}
func (p *PositionalEncoding) AccumulateBackward(grad []float32) []float32 { return p.Backward(grad) }

// MultiHeadAttention implements scaled dot-product multi-head attention.
type MultiHeadAttention struct {
	dim      int
	numHeads int
	headDim  int
	seqLen   int
	Causal   bool

	// Query, Key, Value projections
	wQ, wK, wV *Dense
	wOut       *Dense

	// Buffers
	outputBuf []float32
	gradInBuf []float32

	// Pre-allocated forward buffers
	qBuf        []float32
	kBuf        []float32
	vBuf        []float32
	scoresBuf   []float32
	finalOutBuf []float32

	device Device
}

func NewMultiHeadAttention(dim, numHeads, seqLen int, causal bool) *MultiHeadAttention {
	headDim := dim / numHeads
	if headDim*numHeads != dim {
		panic("dim must be divisible by numHeads")
	}

	return &MultiHeadAttention{
		dim:         dim,
		numHeads:    numHeads,
		headDim:     headDim,
		seqLen:      seqLen,
		Causal:      causal,
		wQ:          NewDense(dim, dim, activations.Linear{}),
		wK:          NewDense(dim, dim, activations.Linear{}),
		wV:          NewDense(dim, dim, activations.Linear{}),
		wOut:        NewDense(dim, dim, activations.Linear{}),
		outputBuf:   make([]float32, seqLen*dim),
		gradInBuf:   make([]float32, seqLen*dim),
		qBuf:        make([]float32, seqLen*dim),
		kBuf:        make([]float32, seqLen*dim),
		vBuf:        make([]float32, seqLen*dim),
		scoresBuf:   make([]float32, seqLen*seqLen),
		finalOutBuf: make([]float32, seqLen*dim),
		device:      &CPUDevice{},
	}
}

// Forward performs self-attention.
func (m *MultiHeadAttention) Forward(x []float32) []float32 {
	// x is [seqLen * dim]
	seqLen := m.seqLen
	dim := m.dim

	// 1. Projections
	m.wQ.Reset()
	m.wK.Reset()
	m.wV.Reset()

	for t := 0; t < seqLen; t++ {
		start := t * dim
		end := start + dim
		copy(m.qBuf[start:end], m.wQ.Forward(x[start:end]))
		copy(m.kBuf[start:end], m.wK.Forward(x[start:end]))
		copy(m.vBuf[start:end], m.wV.Forward(x[start:end]))
	}

	// 2. Scaled Dot-Product Attention for each head
	headDim := m.headDim
	numHeads := m.numHeads
	scale := float32(math.Sqrt(float64(headDim)))

	softmax := activations.Softmax{}

	for h := 0; h < numHeads; h++ {
		// Compute Attention Scores: [seqLen, seqLen]
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				if m.Causal && j > i {
					m.scoresBuf[i*seqLen+j] = -1e9 // Causal mask
					continue
				}

				dot := float32(0.0)
				for d := 0; d < headDim; d++ {
					dot += m.qBuf[i*dim+h*headDim+d] * m.kBuf[j*dim+h*headDim+d]
				}
				m.scoresBuf[i*seqLen+j] = dot / scale
			}
		}

		// Softmax along rows
		for i := 0; i < seqLen; i++ {
			row := m.scoresBuf[i*seqLen : (i+1)*seqLen]
			softmax.ActivateBatch(row)
		}

		// Weighted Sum: [seqLen, headDim]
		for i := 0; i < seqLen; i++ {
			for d := 0; d < headDim; d++ {
				sum := float32(0.0)
				for j := 0; j < seqLen; j++ {
					sum += m.scoresBuf[i*seqLen+j] * m.vBuf[j*dim+h*headDim+d]
				}
				m.finalOutBuf[i*dim+h*headDim+d] = sum
			}
		}
	}

	// 3. Final projection
	m.wOut.Reset()
	for t := 0; t < seqLen; t++ {
		start := t * dim
		end := start + dim
		copy(m.outputBuf[start:end], m.wOut.Forward(m.finalOutBuf[start:end]))
	}

	return m.outputBuf
}

func (m *MultiHeadAttention) Backward(grad []float32) []float32 {
	// grad is [seqLen * dim]
	seqLen := m.seqLen
	dim := m.dim
	headDim := m.headDim
	numHeads := m.numHeads
	scale := float32(math.Sqrt(float64(headDim)))

	// 1. Backprop through wOut
	gradFinalOut := make([]float32, seqLen*dim)
	for t := 0; t < seqLen; t++ {
		start := t * dim
		end := start + dim
		copy(gradFinalOut[start:end], m.wOut.Backward(grad[start:end]))
	}

	// 2. Backprop through Multi-Head Attention
	gradQ := make([]float32, seqLen*dim)
	gradK := make([]float32, seqLen*dim)
	gradV := make([]float32, seqLen*dim)
	gradIn := make([]float32, seqLen*dim)

	softmax := activations.Softmax{}

	for h := 0; h < numHeads; h++ {
		for i := 0; i < seqLen; i++ {
			// dL/dScores
			gradRow := make([]float32, seqLen)
			for j := 0; j < seqLen; j++ {
				// Causal mask: gradient is 0 for masked positions
				if m.Causal && j > i {
					continue
				}

				// dL/dScores[i,j] = sum_d (gradFinalOut[i,h,d] * vBuf[j,h,d])
				dot := float32(0.0)
				for d := 0; d < headDim; d++ {
					dot += gradFinalOut[i*dim+h*headDim+d] * m.vBuf[j*dim+h*headDim+d]
				}
				gradRow[j] = dot
			}

			// Softmax backward
			row := make([]float32, seqLen)
			for j := 0; j < seqLen; j++ {
				if m.Causal && j > i {
					row[j] = -1e9
				} else {
					dot := float32(0.0)
					for d := 0; d < headDim; d++ {
						dot += m.qBuf[i*dim+h*headDim+d] * m.kBuf[j*dim+h*headDim+d]
					}
					row[j] = dot / scale
				}
			}
			probs := softmax.ActivateBatch(row)

			// dL/dPreSoftmax
			dPreSoftmax := make([]float32, seqLen)
			for j := 0; j < seqLen; j++ {
				sum := float32(0.0)
				for k := 0; k < seqLen; k++ {
					delta := float32(0.0)
					if j == k {
						delta = 1.0
					}
					sum += gradRow[k] * probs[k] * (delta - probs[j])
				}
				dPreSoftmax[j] = sum
			}

			// dL/dQ and dL/dK
			for j := 0; j < seqLen; j++ {
				if m.Causal && j > i {
					continue
				}
				gradScore := dPreSoftmax[j] / scale
				for d := 0; d < headDim; d++ {
					gradQ[i*dim+h*headDim+d] += gradScore * m.kBuf[j*dim+h*headDim+d]
					gradK[j*dim+h*headDim+d] += gradScore * m.qBuf[i*dim+h*headDim+d]
				}
			}

			// dL/dV
			for j := 0; j < seqLen; j++ {
				score := probs[j]
				for d := 0; d < headDim; d++ {
					gradV[j*dim+h*headDim+d] += score * gradFinalOut[i*dim+h*headDim+d]
				}
			}
		}
	}

	// 3. Backprop through wQ, wK, wV
	for t := 0; t < seqLen; t++ {
		start := t * dim
		end := start + dim
		dq := m.wQ.Backward(gradQ[start:end])
		dk := m.wK.Backward(gradK[start:end])
		dv := m.wV.Backward(gradV[start:end])

		for i := 0; i < dim; i++ {
			gradIn[start+i] = dq[i] + dk[i] + dv[i]
		}
	}

	return gradIn
}

func (m *MultiHeadAttention) Params() []float32 {
	res := append(m.wQ.Params(), m.wK.Params()...)
	res = append(res, m.wV.Params()...)
	res = append(res, m.wOut.Params()...)
	return res
}

func (m *MultiHeadAttention) SetParams(p []float32) {
	size := len(m.wQ.Params())
	m.wQ.SetParams(p[0:size])
	m.wK.SetParams(p[size : 2*size])
	m.wV.SetParams(p[2*size : 3*size])
	m.wOut.SetParams(p[3*size : 4*size])
}

func (m *MultiHeadAttention) Gradients() []float32 {
	res := append(m.wQ.Gradients(), m.wK.Gradients()...)
	res = append(res, m.wV.Gradients()...)
	res = append(res, m.wOut.Gradients()...)
	return res
}

func (m *MultiHeadAttention) SetGradients(g []float32) {
	size := len(m.wQ.Gradients())
	m.wQ.SetGradients(g[0:size])
	m.wK.SetGradients(g[size : 2*size])
	m.wV.SetGradients(g[2*size : 3*size])
	m.wOut.SetGradients(g[3*size : 4*size])
}

func (m *MultiHeadAttention) SetDevice(d Device) {
	m.device = d
	m.wQ.SetDevice(d)
	m.wK.SetDevice(d)
	m.wV.SetDevice(d)
	m.wOut.SetDevice(d)
}

func (m *MultiHeadAttention) InSize() int  { return m.seqLen * m.dim }
func (m *MultiHeadAttention) OutSize() int { return m.seqLen * m.dim }
func (m *MultiHeadAttention) Reset() {
	m.wQ.Reset()
	m.wK.Reset()
	m.wV.Reset()
	m.wOut.Reset()
}
func (m *MultiHeadAttention) ClearGradients() {
	m.wQ.ClearGradients()
	m.wK.ClearGradients()
	m.wV.ClearGradients()
	m.wOut.ClearGradients()
}
func (m *MultiHeadAttention) Clone() Layer {
	return NewMultiHeadAttention(m.dim, m.numHeads, m.seqLen, m.Causal)
}
func (m *MultiHeadAttention) AccumulateBackward(grad []float32) []float32 { return m.Backward(grad) }

// TransformerBlock combines Self-Attention, LayerNorm, and Feed-Forward.
type TransformerBlock struct {
	mha   *MultiHeadAttention
	norm1 *LayerNorm
	norm2 *LayerNorm
	ff1   *Dense
	ff2   *Dense

	dim    int
	seqLen int

	outputBuf   []float32
	res1Buf     []float32
	norm1OutBuf []float32
	ffOutBuf    []float32
	res2Buf     []float32
}

func NewTransformerBlock(dim, numHeads, seqLen, ffDim int, causal bool) *TransformerBlock {
	return &TransformerBlock{
		mha:         NewMultiHeadAttention(dim, numHeads, seqLen, causal),
		norm1:       NewLayerNorm(dim, 1e-5, true),
		norm2:       NewLayerNorm(dim, 1e-5, true),
		ff1:         NewDense(dim, ffDim, activations.ReLU{}),
		ff2:         NewDense(ffDim, dim, activations.Linear{}),
		dim:         dim,
		seqLen:      seqLen,
		outputBuf:   make([]float32, seqLen*dim),
		res1Buf:     make([]float32, seqLen*dim),
		norm1OutBuf: make([]float32, seqLen*dim),
		ffOutBuf:    make([]float32, seqLen*dim),
		res2Buf:     make([]float32, seqLen*dim),
	}
}

func (t *TransformerBlock) Forward(x []float32) []float32 {
	// 1. Multi-Head Attention + Residual Connection + LayerNorm
	attOut := t.mha.Forward(x)

	// Add & Norm 1 (Residual without new slice)
	for i := range x {
		t.res1Buf[i] = x[i] + attOut[i]
	}

	// LayerNorm 1
	copy(t.norm1OutBuf, t.norm1.Forward(t.res1Buf))

	// 2. Feed-Forward + Residual Connection + LayerNorm
	for s := 0; s < t.seqLen; s++ {
		start := s * t.dim
		end := start + t.dim
		mid := t.ff1.Forward(t.norm1OutBuf[start:end])
		copy(t.ffOutBuf[start:end], t.ff2.Forward(mid))
	}

	// Add & Norm 2 (Residual without new slice)
	for i := range x {
		t.res2Buf[i] = t.norm1OutBuf[i] + t.ffOutBuf[i]
	}

	// LayerNorm 2
	copy(t.outputBuf, t.norm2.Forward(t.res2Buf))


	return t.outputBuf
}

func (t *TransformerBlock) Backward(grad []float32) []float32 {
	// grad is [seqLen * dim]
	seqLen := t.seqLen
	dim := t.dim

	// Backprop through LayerNorm 2
	gradRes2 := make([]float32, seqLen*dim)
	copy(gradRes2, t.norm2.Backward(grad))

	// Residual: gradNorm1Out = gradRes2, gradFFOut = gradRes2
	gradFFOut := gradRes2

	// Backprop through Feed-Forward
	gradNorm1Out_FF := make([]float32, seqLen*dim)
	for s := 0; s < seqLen; s++ {
		start := s * dim
		end := start + dim
		midGrad := t.ff2.Backward(gradFFOut[start:end])
		copy(gradNorm1Out_FF[start:end], t.ff1.Backward(midGrad))
	}

	// Backprop through LayerNorm 1 and Add & Norm 1
	gradRes1 := make([]float32, seqLen*dim)
	// Combined gradient from residual and FF path
	combinedGrad := make([]float32, seqLen*dim)
	for i := 0; i < seqLen*dim; i++ {
		combinedGrad[i] = gradRes2[i] + gradNorm1Out_FF[i]
	}
	copy(gradRes1, t.norm1.Backward(combinedGrad))

	// Backprop through Multi-Head Attention
	gradAttOut := gradRes1
	gradIn_Att := t.mha.Backward(gradAttOut)

	// Final residual grad
	gradIn := make([]float32, seqLen*dim)
	for i := range gradIn {
		gradIn[i] = gradRes1[i] + gradIn_Att[i]
	}

	return gradIn
}

func (t *TransformerBlock) Params() []float32 {
	res := t.mha.Params()
	res = append(res, t.norm1.Params()...)
	res = append(res, t.norm2.Params()...)
	res = append(res, t.ff1.Params()...)
	res = append(res, t.ff2.Params()...)
	return res
}

func (t *TransformerBlock) SetParams(p []float32) {
	offset := 0
	size := len(t.mha.Params())
	t.mha.SetParams(p[offset : offset+size])
	offset += size

	size = len(t.norm1.Params())
	t.norm1.SetParams(p[offset : offset+size])
	offset += size

	size = len(t.norm2.Params())
	t.norm2.SetParams(p[offset : offset+size])
	offset += size

	size = len(t.ff1.Params())
	t.ff1.SetParams(p[offset : offset+size])
	offset += size

	size = len(t.ff2.Params())
	t.ff2.SetParams(p[offset : offset+size])
}

func (t *TransformerBlock) Gradients() []float32 {
	res := t.mha.Gradients()
	res = append(res, t.norm1.Gradients()...)
	res = append(res, t.norm2.Gradients()...)
	res = append(res, t.ff1.Gradients()...)
	res = append(res, t.ff2.Gradients()...)
	return res
}

func (t *TransformerBlock) SetGradients(g []float32) {
	offset := 0
	size := len(t.mha.Gradients())
	t.mha.SetGradients(g[offset : offset+size])
	offset += size

	size = len(t.norm1.Gradients())
	t.norm1.SetGradients(g[offset : offset+size])
	offset += size

	size = len(t.norm2.Gradients())
	t.norm2.SetGradients(g[offset : offset+size])
	offset += size

	size = len(t.ff1.Gradients())
	t.ff1.SetGradients(g[offset : offset+size])
	offset += size

	size = len(t.ff2.Gradients())
	t.ff2.SetGradients(g[offset : offset+size])
}
func (t *TransformerBlock) SetDevice(d Device)         { t.mha.SetDevice(d) }
func (t *TransformerBlock) InSize() int                { return t.seqLen * t.dim }
func (t *TransformerBlock) OutSize() int               { return t.seqLen * t.dim }
func (t *TransformerBlock) Reset() {
	t.mha.Reset()
	t.norm1.Reset()
	t.norm2.Reset()
	t.ff1.Reset()
	t.ff2.Reset()
}
func (t *TransformerBlock) ClearGradients() {
	t.mha.ClearGradients()
	t.norm1.ClearGradients()
	t.norm2.ClearGradients()
	t.ff1.ClearGradients()
	t.ff2.ClearGradients()
}
func (t *TransformerBlock) Clone() Layer {
	return NewTransformerBlock(t.dim, t.mha.numHeads, t.seqLen, t.ff1.OutSize(), t.mha.Causal)
}
func (t *TransformerBlock) AccumulateBackward(grad []float32) []float32 { return t.Backward(grad) }

// GlobalAveragePooling1D pools across the sequence dimension.
type GlobalAveragePooling1D struct {
	seqLen int
	dim    int
}

func NewGlobalAveragePooling1D(seqLen, dim int) *GlobalAveragePooling1D {
	return &GlobalAveragePooling1D{seqLen: seqLen, dim: dim}
}

func (g *GlobalAveragePooling1D) Forward(x []float32) []float32 {
	out := make([]float32, g.dim)
	for i := 0; i < g.dim; i++ {
		sum := float32(0.0)
		for s := 0; s < g.seqLen; s++ {
			sum += x[s*g.dim+i]
		}
		out[i] = sum / float32(g.seqLen)
	}
	return out
}

func (g *GlobalAveragePooling1D) Backward(grad []float32) []float32 {
	dx := make([]float32, g.seqLen*g.dim)
	for s := 0; s < g.seqLen; s++ {
		for i := 0; i < g.dim; i++ {
			dx[s*g.dim+i] = grad[i] / float32(g.seqLen)
		}
	}
	return dx
}

func (g *GlobalAveragePooling1D) Params() []float32         { return nil }
func (g *GlobalAveragePooling1D) SetParams(p []float32)     {}
func (g *GlobalAveragePooling1D) Gradients() []float32      { return nil }
func (g *GlobalAveragePooling1D) SetGradients(g_ []float32) {}
func (g *GlobalAveragePooling1D) SetDevice(d Device)        {}
func (g *GlobalAveragePooling1D) InSize() int               { return g.seqLen * g.dim }
func (g *GlobalAveragePooling1D) OutSize() int              { return g.dim }
func (g *GlobalAveragePooling1D) Reset()                   {}
func (g *GlobalAveragePooling1D) ClearGradients()          {}
func (g *GlobalAveragePooling1D) Clone() Layer              { return &GlobalAveragePooling1D{g.seqLen, g.dim} }
func (g *GlobalAveragePooling1D) AccumulateBackward(grad []float32) []float32 { return g.Backward(grad) }

// CLSPooling extracts the first vector (CLS token) from a sequence.
type CLSPooling struct {
	seqLen int
	dim    int

	outputBuf []float32
	gradBuf   []float32
}

func NewCLSPooling(seqLen, dim int) *CLSPooling {
	return &CLSPooling{
		seqLen:    seqLen,
		dim:       dim,
		outputBuf: make([]float32, dim),
		gradBuf:   make([]float32, seqLen*dim),
	}
}

func (c *CLSPooling) Forward(x []float32) []float32 {
	// x is [seqLen * dim]
	copy(c.outputBuf, x[0:c.dim])
	return c.outputBuf
}

func (c *CLSPooling) Backward(grad []float32) []float32 {
	// grad is [dim]
	for i := range c.gradBuf {
		c.gradBuf[i] = 0
	}
	copy(c.gradBuf[0:c.dim], grad)
	return c.gradBuf
}

func (c *CLSPooling) Params() []float32         { return nil }
func (c *CLSPooling) SetParams(p []float32)     {}
func (c *CLSPooling) Gradients() []float32      { return nil }
func (c *CLSPooling) SetGradients(g []float32) {}
func (c *CLSPooling) SetDevice(d Device)        {}
func (c *CLSPooling) InSize() int               { return c.seqLen * c.dim }
func (c *CLSPooling) OutSize() int              { return c.dim }
func (c *CLSPooling) Reset()                   {}
func (c *CLSPooling) ClearGradients()          {}
func (c *CLSPooling) Clone() Layer              { return &CLSPooling{c.seqLen, c.dim, make([]float32, c.dim), make([]float32, c.seqLen*c.dim)} }
func (c *CLSPooling) AccumulateBackward(grad []float32) []float32 { return c.Backward(grad) }
