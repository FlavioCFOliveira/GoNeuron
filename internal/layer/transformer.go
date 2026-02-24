// Package layer provides neural network layer implementations.
package layer

import (
	"fmt"
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

	training bool

	device Device
}

func NewPositionalEncoding(seqLen, dim int) *PositionalEncoding {
	p := &PositionalEncoding{
		seqLen: seqLen,
		dim:    dim,
		device: &CPUDevice{},
	}

	if dim != -1 {
		p.Build(seqLen * dim)
	}

	return p
}

func (p *PositionalEncoding) Build(inSize int) {
	if p.dim == -1 {
		p.dim = inSize / p.seqLen
	}
	dim := p.dim
	seqLen := p.seqLen

	p.weights = make([]float32, seqLen*dim)
	rng := NewRNG(uint64(seqLen + dim + 42))
	scale := float32(math.Sqrt(1.0 / float64(dim)))
	for i := range p.weights {
		p.weights[i] = rng.RandFloat()*2*scale - scale
	}

	p.outputBuf = make([]float32, seqLen*dim)
	p.gradBuf = make([]float32, seqLen*dim)
}

// SetTraining sets whether the layer is in training mode.
func (p *PositionalEncoding) SetTraining(training bool) {
	p.training = training
}

func (p *PositionalEncoding) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	// x is [seqLen * dim]
	if len(p.outputBuf) < len(x) {
		p.outputBuf = make([]float32, len(x))
	}
	for i := range x {
		p.outputBuf[i] = x[i] + p.weights[i]
	}
	return p.outputBuf[:len(x)]
}

func (p *PositionalEncoding) Forward(x []float32) []float32 {
	return p.ForwardWithArena(x, nil, nil)
}

func (p *PositionalEncoding) ForwardBatch(x []float32, batchSize int) []float32 {
	return p.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (p *PositionalEncoding) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return p.ForwardWithArena(x, arena, offset)
	}
	inSize := p.InSize()
	outSize := p.OutSize()
	if len(p.outputBuf) < batchSize*outSize {
		p.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := p.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(p.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return p.outputBuf[:batchSize*outSize]
}

func (p *PositionalEncoding) Backward(grad []float32) []float32 {
	// Gradient w.r.t. parameters is same as gradient w.r.t. output
	// Gradient w.r.t. input is also same as gradient w.r.t. output
	for i := range grad {
		p.gradBuf[i] += grad[i]
	}
	return grad
}

func (p *PositionalEncoding) BackwardBatch(grad []float32, batchSize int) []float32 {
	return p.Backward(grad)
}

func (p *PositionalEncoding) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return p.Backward(grad)
}

func (p *PositionalEncoding) Params() []float32 {
	return p.weights
}

func (p *PositionalEncoding) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &p.weights[0] != &params[0] {
		p.weights = params
	}
}

func (p *PositionalEncoding) Gradients() []float32 {
	return p.gradBuf
}

func (p *PositionalEncoding) SetGradients(g []float32) {
	if len(g) == 0 {
		return
	}
	if &p.gradBuf[0] != &g[0] {
		p.gradBuf = g
	}
}
func (p *PositionalEncoding) SetDevice(d Device)         { p.device = d }
func (p *PositionalEncoding) InSize() int                { return p.seqLen * p.dim }
func (p *PositionalEncoding) OutSize() int               { return p.seqLen * p.dim }

func (p *PositionalEncoding) NamedParams() []NamedParam {
	return []NamedParam{
		{
			Name:  "weights",
			Shape: []int{p.seqLen, p.dim},
			Data:  p.weights,
		},
	}
}

func (p *PositionalEncoding) Reset()                    {}
func (p *PositionalEncoding) ClearGradients()           { for i := range p.gradBuf { p.gradBuf[i] = 0 } }
func (p *PositionalEncoding) Clone() Layer {
	newP := NewPositionalEncoding(p.seqLen, p.dim)
	copy(newP.weights, p.weights)
	return newP
}

func (p *PositionalEncoding) LightweightClone(params []float32, grads []float32) Layer {
	newP := &PositionalEncoding{
		seqLen:    p.seqLen,
		dim:       p.dim,
		weights:   params,
		outputBuf: make([]float32, p.seqLen*p.dim),
		gradBuf:   grads,
		training:  p.training,
		device:    p.device,
	}
	return newP
}
func (p *PositionalEncoding) AccumulateBackward(grad []float32) []float32 { return p.Backward(grad) }

type ActivationType int

const (
	ActReLU ActivationType = iota
	ActSwiGLU
	ActGELU
)

type NormType int

const (
	NormLN NormType = iota
	NormRMS
)

// applyRoPE applies Rotary Positional Embeddings in-place.
func applyRoPE(x []float32, seqLen, numHeads, headDim int, conjugate bool) {
	for t := 0; t < seqLen; t++ {
		for h := 0; h < numHeads; h++ {
			for i := 0; i < headDim/2; i++ {
				theta := float32(math.Pow(10000, -float64(2*i)/float64(headDim)))
				mTheta := float32(t) * theta
				c := float32(math.Cos(float64(mTheta)))
				s := float32(math.Sin(float64(mTheta)))
				if conjugate {
					s = -s
				}

				idx1 := t*numHeads*headDim + h*headDim + i
				idx2 := t*numHeads*headDim + h*headDim + i + headDim/2

				v1 := x[idx1]
				v2 := x[idx2]

				x[idx1] = v1*c - v2*s
				x[idx2] = v1*s + v2*c
			}
		}
	}
}

// MultiHeadAttention implements scaled dot-product multi-head attention.
type MultiHeadAttention struct {
	dim      int
	numHeads int
	headDim  int
	seqLen   int
	Causal   bool
	UseRoPE  bool

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

	// Pre-allocated backward buffers
	gradFinalOutBuf []float32
	gradQBuf        []float32
	gradKBuf        []float32
	gradVBuf        []float32

	// Contiguous parameter and gradient buffers
	params []float32
	grads  []float32

	training bool

	device Device
}

func NewMultiHeadAttention(dim, numHeads, seqLen int, causal bool) *MultiHeadAttention {
	headDim := -1
	if dim != -1 {
		headDim = dim / numHeads
		if headDim*numHeads != dim {
			panic("dim must be divisible by numHeads")
		}
	}

	m := &MultiHeadAttention{
		dim:      dim,
		numHeads: numHeads,
		headDim:  headDim,
		seqLen:   seqLen,
		Causal:   causal,
		device:   &CPUDevice{},
	}

	if dim != -1 {
		m.Build(seqLen * dim)
	}

	return m
}

func (m *MultiHeadAttention) Build(inSize int) {
	if m.dim == -1 {
		if inSize%m.seqLen == 0 && inSize > m.seqLen {
			m.dim = inSize / m.seqLen
		} else {
			m.dim = inSize
		}
		m.headDim = m.dim / m.numHeads
		if m.headDim*m.numHeads != m.dim {
			panic(fmt.Sprintf("inferred dim %d must be divisible by numHeads %d (inSize %d, seqLen %d)", m.dim, m.numHeads, inSize, m.seqLen))
		}
	}

	dim := m.dim
	seqLen := m.seqLen

	m.wQ = NewDense(dim, dim, activations.Linear{})
	m.wK = NewDense(dim, dim, activations.Linear{})
	m.wV = NewDense(dim, dim, activations.Linear{})
	m.wOut = NewDense(dim, dim, activations.Linear{})

	// Calculate total parameters
	paramSize := len(m.wQ.Params())
	totalSize := paramSize * 4
	m.params = make([]float32, totalSize)
	m.grads = make([]float32, totalSize)

	// Link sub-layers to these contiguous buffers
	m.wQ.SetParams(m.params[0:paramSize])
	m.wK.SetParams(m.params[paramSize : 2*paramSize])
	m.wV.SetParams(m.params[2*paramSize : 3*paramSize])
	m.wOut.SetParams(m.params[3*paramSize : 4*paramSize])

	m.wQ.SetGradients(m.grads[0:paramSize])
	m.wK.SetGradients(m.grads[paramSize : 2*paramSize])
	m.wV.SetGradients(m.grads[2*paramSize : 3*paramSize])
	m.wOut.SetGradients(m.grads[3*paramSize : 4*paramSize])

	m.outputBuf = make([]float32, seqLen*dim)
	m.gradInBuf = make([]float32, seqLen*dim)
	m.qBuf = make([]float32, seqLen*dim)
	m.kBuf = make([]float32, seqLen*dim)
	m.vBuf = make([]float32, seqLen*dim)
	m.scoresBuf = make([]float32, seqLen*seqLen)
	m.finalOutBuf = make([]float32, seqLen*dim)

	m.gradFinalOutBuf = make([]float32, seqLen*dim)
	m.gradQBuf = make([]float32, seqLen*dim)
	m.gradKBuf = make([]float32, seqLen*dim)
	m.gradVBuf = make([]float32, seqLen*dim)
}

// SetTraining sets whether the layer is in training mode.
func (m *MultiHeadAttention) SetTraining(training bool) {
	m.training = training
	m.wQ.SetTraining(training)
	m.wK.SetTraining(training)
	m.wV.SetTraining(training)
	m.wOut.SetTraining(training)
}

func (m *MultiHeadAttention) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
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
		var q, k, v []float32
		if arena != nil && offset != nil {
			q = m.wQ.ForwardWithArena(x[start:end], arena, offset)
			k = m.wK.ForwardWithArena(x[start:end], arena, offset)
			v = m.wV.ForwardWithArena(x[start:end], arena, offset)
		} else {
			q = m.wQ.Forward(x[start:end])
			k = m.wK.Forward(x[start:end])
			v = m.wV.Forward(x[start:end])
		}
		copy(m.qBuf[start:end], q)
		copy(m.kBuf[start:end], k)
		copy(m.vBuf[start:end], v)
	}

	// 1.5 Apply RoPE if enabled
	if m.UseRoPE {
		applyRoPE(m.qBuf, seqLen, m.numHeads, m.headDim, false)
		applyRoPE(m.kBuf, seqLen, m.numHeads, m.headDim, false)
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
		var out []float32
		if arena != nil && offset != nil {
			out = m.wOut.ForwardWithArena(m.finalOutBuf[start:end], arena, offset)
		} else {
			out = m.wOut.Forward(m.finalOutBuf[start:end])
		}
		copy(m.outputBuf[start:end], out)
	}

	return m.outputBuf
}

// Forward performs self-attention.
func (m *MultiHeadAttention) Forward(x []float32) []float32 {
	return m.ForwardWithArena(x, nil, nil)
}

func (m *MultiHeadAttention) ForwardBatch(x []float32, batchSize int) []float32 {
	return m.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (m *MultiHeadAttention) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return m.ForwardWithArena(x, arena, offset)
	}
	inSize := m.InSize()
	outSize := m.OutSize()
	if len(m.outputBuf) < batchSize*outSize {
		m.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := m.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(m.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return m.outputBuf[:batchSize*outSize]
}

func (m *MultiHeadAttention) Backward(grad []float32) []float32 {
	// grad is [seqLen * dim]
	seqLen := m.seqLen
	dim := m.dim
	headDim := m.headDim
	numHeads := m.numHeads
	scale := float32(math.Sqrt(float64(headDim)))

	// 1. Backprop through wOut
	gradFinalOut := m.gradFinalOutBuf
	for t := 0; t < seqLen; t++ {
		start := t * dim
		end := start + dim
		copy(gradFinalOut[start:end], m.wOut.Backward(grad[start:end]))
	}

	// 2. Backprop through Multi-Head Attention
	gradQ := m.gradQBuf
	gradK := m.gradKBuf
	gradV := m.gradVBuf
	for i := range gradQ {
		gradQ[i] = 0
		gradK[i] = 0
		gradV[i] = 0
	}

	softmax := activations.Softmax{}

	for h := 0; h < numHeads; h++ {
		for i := 0; i < seqLen; i++ {
			// dL/dScores
			// Reuse scoresBuf for a moment to store gradRow if possible, but scoresBuf is [seqLen, seqLen]
			// Let's use a local small buffer for row-based ops
			gradRow := make([]float32, seqLen) // Small enough for stack
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

	// 2.5 Apply RoPE backward if enabled
	if m.UseRoPE {
		applyRoPE(gradQ, seqLen, m.numHeads, m.headDim, true)
		applyRoPE(gradK, seqLen, m.numHeads, m.headDim, true)
	}

	// 3. Backprop through wQ, wK, wV
	gradIn := m.gradInBuf
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

func (m *MultiHeadAttention) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return m.Backward(grad)
	}
	inSize := m.InSize()
	outSize := m.OutSize()
	if len(m.gradInBuf) < batchSize*inSize {
		m.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		out := m.Backward(grad[i*outSize : (i+1)*outSize])
		copy(m.gradInBuf[i*inSize:(i+1)*inSize], out)
	}
	return m.gradInBuf[:batchSize*inSize]
}

func (m *MultiHeadAttention) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return m.BackwardBatch(grad, batchSize)
}

func (m *MultiHeadAttention) Params() []float32 {
	return m.params
}

func (m *MultiHeadAttention) SetParams(p []float32) {
	if len(p) == 0 {
		return
	}
	if &m.params[0] != &p[0] {
		if len(p) == len(m.params) {
			m.params = p
			m.updateViews()
		} else {
			copy(m.params, p)
		}
	}
}

func (m *MultiHeadAttention) updateViews() {
	paramSize := len(m.wQ.Params())
	m.wQ.SetParams(m.params[0:paramSize])
	m.wK.SetParams(m.params[paramSize : 2*paramSize])
	m.wV.SetParams(m.params[2*paramSize : 3*paramSize])
	m.wOut.SetParams(m.params[3*paramSize : 4*paramSize])
}

func (m *MultiHeadAttention) Gradients() []float32 {
	return m.grads
}

func (m *MultiHeadAttention) SetGradients(g []float32) {
	if len(g) == 0 {
		return
	}
	if &m.grads[0] != &g[0] {
		if len(g) == len(m.grads) {
			m.grads = g
			m.updateGradViews()
		} else {
			copy(m.grads, g)
		}
	}
}

func (m *MultiHeadAttention) updateGradViews() {
	paramSize := len(m.wQ.Gradients())
	m.wQ.SetGradients(m.grads[0:paramSize])
	m.wK.SetGradients(m.grads[paramSize : 2*paramSize])
	m.wV.SetGradients(m.grads[2*paramSize : 3*paramSize])
	m.wOut.SetGradients(m.grads[3*paramSize : 4*paramSize])
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

func (m *MultiHeadAttention) NamedParams() []NamedParam {
	var params []NamedParam
	for _, p := range m.wQ.NamedParams() {
		params = append(params, NamedParam{Name: "wQ_" + p.Name, Shape: p.Shape, Data: p.Data})
	}
	for _, p := range m.wK.NamedParams() {
		params = append(params, NamedParam{Name: "wK_" + p.Name, Shape: p.Shape, Data: p.Data})
	}
	for _, p := range m.wV.NamedParams() {
		params = append(params, NamedParam{Name: "wV_" + p.Name, Shape: p.Shape, Data: p.Data})
	}
	for _, p := range m.wOut.NamedParams() {
		params = append(params, NamedParam{Name: "wOut_" + p.Name, Shape: p.Shape, Data: p.Data})
	}
	return params
}

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

func (m *MultiHeadAttention) LightweightClone(params []float32, grads []float32) Layer {
	paramSize := len(m.wQ.Params())
	gradSize := len(m.wQ.Gradients())

	newM := &MultiHeadAttention{
		dim:             m.dim,
		numHeads:        m.numHeads,
		headDim:         m.headDim,
		seqLen:          m.seqLen,
		Causal:          m.Causal,
		params:          params,
		grads:           grads,
		outputBuf:       make([]float32, m.seqLen*m.dim),
		gradInBuf:       make([]float32, m.seqLen*m.dim),
		qBuf:            make([]float32, m.seqLen*m.dim),
		kBuf:            make([]float32, m.seqLen*m.dim),
		vBuf:            make([]float32, m.seqLen*m.dim),
		scoresBuf:       make([]float32, m.seqLen*m.seqLen),
		finalOutBuf:     make([]float32, m.seqLen*m.dim),
		gradFinalOutBuf: make([]float32, m.seqLen*m.dim),
		gradQBuf:        make([]float32, m.seqLen*m.dim),
		gradKBuf:        make([]float32, m.seqLen*m.dim),
		gradVBuf:        make([]float32, m.seqLen*m.dim),
		training:        m.training,
		device:          m.device,
	}

	newM.wQ = m.wQ.LightweightClone(params[0:paramSize], grads[0:gradSize]).(*Dense)
	newM.wK = m.wK.LightweightClone(params[paramSize:2*paramSize], grads[gradSize:2*gradSize]).(*Dense)
	newM.wV = m.wV.LightweightClone(params[2*paramSize:3*paramSize], grads[2*gradSize:3*gradSize]).(*Dense)
	newM.wOut = m.wOut.LightweightClone(params[3*paramSize:4*paramSize], grads[3*gradSize:4*gradSize]).(*Dense)

	return newM
}
func (m *MultiHeadAttention) AccumulateBackward(grad []float32) []float32 { return m.Backward(grad) }

type TransformerBlock struct {
	mha   *MultiHeadAttention
	norm1 Layer
	norm2 Layer
	ff1   Layer
	ff2   *Dense

	dim    int
	seqLen int
	ffDim  int

	outputBuf   []float32
	res1Buf     []float32
	norm1OutBuf []float32
	ffOutBuf    []float32
	res2Buf     []float32

	// Pre-allocated backward buffers
	gradRes2Buf       []float32
	gradNorm1OutFFBuf []float32
	combinedGradBuf   []float32
	gradRes1Buf       []float32
	gradInBuf         []float32

	// Contiguous parameter and gradient buffers
	params []float32
	grads  []float32

	actType  ActivationType
	normType NormType
	useRoPE  bool
	device   Device
	training bool
}

func NewTransformerBlock(dim, numHeads, seqLen, ffDim int, causal bool) *TransformerBlock {
	return NewTransformerBlockExt(dim, numHeads, seqLen, ffDim, causal, ActReLU, NormLN, false)
}

func NewTransformerBlockExt(dim, numHeads, seqLen, ffDim int, causal bool, actType ActivationType, normType NormType, useRoPE bool) *TransformerBlock {
	mha := NewMultiHeadAttention(dim, numHeads, seqLen, causal)
	mha.UseRoPE = useRoPE

	var norm1, norm2 Layer
	if normType == NormRMS {
		norm1 = NewRMSNorm(dim, 1e-5)
		norm2 = NewRMSNorm(dim, 1e-5)
	} else {
		norm1 = NewLayerNorm(dim, 1e-5, true)
		norm2 = NewLayerNorm(dim, 1e-5, true)
	}

	var ff1 Layer
	if actType == ActSwiGLU {
		ff1 = NewSwiGLU(dim, ffDim)
	} else if actType == ActGELU {
		ff1 = NewDense(dim, ffDim, activations.GELU{})
	} else {
		ff1 = NewDense(dim, ffDim, activations.ReLU{})
	}
	ff2 := NewDense(ffDim, dim, activations.Linear{})

	t := &TransformerBlock{
		mha:      mha,
		norm1:    norm1,
		norm2:    norm2,
		ff1:      ff1,
		ff2:      ff2,
		dim:      dim,
		seqLen:   seqLen,
		ffDim:    ffDim,
		actType:  actType,
		normType: normType,
		useRoPE:  useRoPE,
	}

	if dim != -1 {
		t.Build(seqLen * dim)
	}

	return t
}

func (t *TransformerBlock) Build(inSize int) {
	if t.dim == -1 {
		if inSize%t.seqLen == 0 && inSize > t.seqLen {
			t.dim = inSize / t.seqLen
		} else {
			t.dim = inSize
		}
	}
	dim := t.dim
	seqLen := t.seqLen

	// Build sub-layers if they are deferred
	if len(t.mha.Params()) == 0 {
		t.mha.Build(inSize)
	}
	if b, ok := t.norm1.(DeferredLayer); ok && len(b.Params()) == 0 {
		b.Build(dim)
	}
	if b, ok := t.norm2.(DeferredLayer); ok && len(b.Params()) == 0 {
		b.Build(dim)
	}
	if b, ok := t.ff1.(DeferredLayer); ok && len(b.Params()) == 0 {
		b.Build(dim)
	}
	if len(t.ff2.Params()) == 0 {
		if t.ff2.outSize <= 0 {
			t.ff2.outSize = dim
		}
		t.ff2.Build(t.ffDim)
	}

	// Aggregate parameters
	sizeMHA := len(t.mha.Params())
	sizeNorm1 := len(t.norm1.Params())
	sizeNorm2 := len(t.norm2.Params())
	sizeFF1 := len(t.ff1.Params())
	sizeFF2 := len(t.ff2.Params())

	totalParams := sizeMHA + sizeNorm1 + sizeNorm2 + sizeFF1 + sizeFF2
	t.params = make([]float32, totalParams)
	t.grads = make([]float32, totalParams)

	offset := 0
	t.mha.SetParams(t.params[offset : offset+sizeMHA])
	t.mha.SetGradients(t.grads[offset : offset+sizeMHA])
	offset += sizeMHA

	t.norm1.SetParams(t.params[offset : offset+sizeNorm1])
	t.norm1.SetGradients(t.grads[offset : offset+sizeNorm1])
	offset += sizeNorm1

	t.norm2.SetParams(t.params[offset : offset+sizeNorm2])
	t.norm2.SetGradients(t.grads[offset : offset+sizeNorm2])
	offset += sizeNorm2

	t.ff1.SetParams(t.params[offset : offset+sizeFF1])
	t.ff1.SetGradients(t.grads[offset : offset+sizeFF1])
	offset += sizeFF1

	t.ff2.SetParams(t.params[offset : offset+sizeFF2])
	t.ff2.SetGradients(t.grads[offset : offset+sizeFF2])

	t.outputBuf = make([]float32, seqLen*dim)
	t.res1Buf = make([]float32, seqLen*dim)
	t.norm1OutBuf = make([]float32, seqLen*dim)
	t.ffOutBuf = make([]float32, seqLen*dim)
	t.res2Buf = make([]float32, seqLen*dim)
	t.gradRes2Buf = make([]float32, seqLen*dim)
	t.gradNorm1OutFFBuf = make([]float32, seqLen*dim)
	t.combinedGradBuf = make([]float32, seqLen*dim)
	t.gradRes1Buf = make([]float32, seqLen*dim)
	t.gradInBuf = make([]float32, seqLen*dim)
}

// SetTraining sets whether the layer is in training mode.
func (t *TransformerBlock) SetTraining(training bool) {
	t.training = training
	t.mha.SetTraining(training)
	t.norm1.SetTraining(training)
	t.norm2.SetTraining(training)
	t.ff1.SetTraining(training)
	t.ff2.SetTraining(training)
}

func (t *TransformerBlock) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	// 1. Multi-Head Attention + Residual Connection + LayerNorm
	var attOut []float32
	if arena != nil && offset != nil {
		attOut = t.mha.ForwardWithArena(x, arena, offset)
	} else {
		attOut = t.mha.Forward(x)
	}

	// Add & Norm 1
	for i := range x {
		t.res1Buf[i] = x[i] + attOut[i]
	}

	// LayerNorm 1
	var norm1Out []float32
	if arena != nil && offset != nil {
		if al, ok := t.norm1.(ArenaLayer); ok {
			norm1Out = al.ForwardWithArena(t.res1Buf, arena, offset)
		} else {
			norm1Out = t.norm1.Forward(t.res1Buf)
		}
	} else {
		norm1Out = t.norm1.Forward(t.res1Buf)
	}
	copy(t.norm1OutBuf, norm1Out)

	// 2. Feed-Forward + Residual Connection + LayerNorm
	for s := 0; s < t.seqLen; s++ {
		start := s * t.dim
		end := start + t.dim
		var mid []float32
		if arena != nil && offset != nil {
			if al, ok := t.ff1.(ArenaLayer); ok {
				mid = al.ForwardWithArena(t.norm1OutBuf[start:end], arena, offset)
			} else {
				mid = t.ff1.Forward(t.norm1OutBuf[start:end])
			}
		} else {
			mid = t.ff1.Forward(t.norm1OutBuf[start:end])
		}

		var ffOut []float32
		if arena != nil && offset != nil {
			ffOut = t.ff2.ForwardWithArena(mid, arena, offset)
		} else {
			ffOut = t.ff2.Forward(mid)
		}
		copy(t.ffOutBuf[start:end], ffOut)
	}

	// Add & Norm 2
	for i := range x {
		t.res2Buf[i] = t.norm1OutBuf[i] + t.ffOutBuf[i]
	}

	// LayerNorm 2
	var output []float32
	if arena != nil && offset != nil {
		if al, ok := t.norm2.(ArenaLayer); ok {
			output = al.ForwardWithArena(t.res2Buf, arena, offset)
		} else {
			output = t.norm2.Forward(t.res2Buf)
		}
	} else {
		output = t.norm2.Forward(t.res2Buf)
	}
	copy(t.outputBuf, output)

	return t.outputBuf
}

func (t *TransformerBlock) Forward(x []float32) []float32 {
	return t.ForwardWithArena(x, nil, nil)
}

func (t *TransformerBlock) ForwardBatch(x []float32, batchSize int) []float32 {
	return t.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (t *TransformerBlock) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return t.ForwardWithArena(x, arena, offset)
	}
	inSize := t.InSize()
	outSize := t.OutSize()
	if len(t.outputBuf) < batchSize*outSize {
		t.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := t.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(t.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return t.outputBuf[:batchSize*outSize]
}

func (t *TransformerBlock) Backward(grad []float32) []float32 {
	// grad is [seqLen * dim]
	seqLen := t.seqLen
	dim := t.dim

	// Backprop through LayerNorm 2
	gradRes2 := t.gradRes2Buf
	copy(gradRes2, t.norm2.Backward(grad))

	// Residual: gradNorm1Out = gradRes2, gradFFOut = gradRes2
	gradFFOut := gradRes2

	// Backprop through Feed-Forward
	gradNorm1Out_FF := t.gradNorm1OutFFBuf
	for s := 0; s < seqLen; s++ {
		start := s * dim
		end := start + dim
		midGrad := t.ff2.Backward(gradFFOut[start:end])
		copy(gradNorm1Out_FF[start:end], t.ff1.Backward(midGrad))
	}

	// Backprop through LayerNorm 1 and Add & Norm 1
	gradRes1 := t.gradRes1Buf
	// Combined gradient from residual and FF path
	combinedGrad := t.combinedGradBuf
	for i := 0; i < seqLen*dim; i++ {
		combinedGrad[i] = gradRes2[i] + gradNorm1Out_FF[i]
	}
	copy(gradRes1, t.norm1.Backward(combinedGrad))

	// Backprop through Multi-Head Attention
	gradAttOut := gradRes1
	gradIn_Att := t.mha.Backward(gradAttOut)

	// Final residual grad
	gradIn := t.gradInBuf
	for i := range gradIn {
		gradIn[i] = gradRes1[i] + gradIn_Att[i]
	}

	return gradIn
}

func (t *TransformerBlock) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return t.Backward(grad)
	}
	inSize := t.InSize()
	outSize := t.OutSize()
	if len(t.gradInBuf) < batchSize*inSize {
		t.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		out := t.Backward(grad[i*outSize : (i+1)*outSize])
		copy(t.gradInBuf[i*inSize:(i+1)*inSize], out)
	}
	return t.gradInBuf[:batchSize*inSize]
}

func (t *TransformerBlock) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return t.BackwardBatch(grad, batchSize)
}

func (t *TransformerBlock) Params() []float32 {
	return t.params
}

func (t *TransformerBlock) SetParams(p []float32) {
	if len(p) == 0 {
		return
	}
	if &t.params[0] != &p[0] {
		if len(p) == len(t.params) {
			t.params = p
			t.updateViews()
		} else {
			copy(t.params, p)
		}
	}
}

func (t *TransformerBlock) updateViews() {
	sizeMHA := len(t.mha.Params())
	sizeNorm1 := len(t.norm1.Params())
	sizeNorm2 := len(t.norm2.Params())
	sizeFF1 := len(t.ff1.Params())
	sizeFF2 := len(t.ff2.Params())

	offset := 0
	t.mha.SetParams(t.params[offset : offset+sizeMHA])
	offset += sizeMHA

	t.norm1.SetParams(t.params[offset : offset+sizeNorm1])
	offset += sizeNorm1

	t.norm2.SetParams(t.params[offset : offset+sizeNorm2])
	offset += sizeNorm2

	t.ff1.SetParams(t.params[offset : offset+sizeFF1])
	offset += sizeFF1

	t.ff2.SetParams(t.params[offset : offset+sizeFF2])
}

func (t *TransformerBlock) Gradients() []float32 {
	return t.grads
}

func (t *TransformerBlock) SetGradients(g []float32) {
	if len(g) == 0 {
		return
	}
	if &t.grads[0] != &g[0] {
		if len(g) == len(t.grads) {
			t.grads = g
			t.updateGradViews()
		} else {
			copy(t.grads, g)
		}
	}
}

func (t *TransformerBlock) updateGradViews() {
	sizeMHA := len(t.mha.Gradients())
	sizeNorm1 := len(t.norm1.Gradients())
	sizeNorm2 := len(t.norm2.Gradients())
	sizeFF1 := len(t.ff1.Gradients())
	sizeFF2 := len(t.ff2.Gradients())

	offset := 0
	t.mha.SetGradients(t.grads[offset : offset+sizeMHA])
	offset += sizeMHA

	t.norm1.SetGradients(t.grads[offset : offset+sizeNorm1])
	offset += sizeNorm1

	t.norm2.SetGradients(t.grads[offset : offset+sizeNorm2])
	offset += sizeNorm2

	t.ff1.SetGradients(t.grads[offset : offset+sizeFF1])
	offset += sizeFF1

	t.ff2.SetGradients(t.grads[offset : offset+sizeFF2])
}
func (t *TransformerBlock) SetDevice(d Device) {
	t.mha.SetDevice(d)
	t.norm1.SetDevice(d)
	t.norm2.SetDevice(d)
	t.ff1.SetDevice(d)
	t.ff2.SetDevice(d)
}
func (t *TransformerBlock) InSize() int                { return t.seqLen * t.dim }
func (t *TransformerBlock) OutSize() int               { return t.seqLen * t.dim }

func (t *TransformerBlock) NamedParams() []NamedParam {
	var params []NamedParam
	for _, p := range t.mha.NamedParams() {
		params = append(params, NamedParam{Name: "mha_" + p.Name, Shape: p.Shape, Data: p.Data})
	}
	if np, ok := t.norm1.(NamedParamProvider); ok {
		for _, p := range np.NamedParams() {
			params = append(params, NamedParam{Name: "norm1_" + p.Name, Shape: p.Shape, Data: p.Data})
		}
	}
	if np, ok := t.norm2.(NamedParamProvider); ok {
		for _, p := range np.NamedParams() {
			params = append(params, NamedParam{Name: "norm2_" + p.Name, Shape: p.Shape, Data: p.Data})
		}
	}
	if np, ok := t.ff1.(NamedParamProvider); ok {
		for _, p := range np.NamedParams() {
			params = append(params, NamedParam{Name: "ff1_" + p.Name, Shape: p.Shape, Data: p.Data})
		}
	}
	for _, p := range t.ff2.NamedParams() {
		params = append(params, NamedParam{Name: "ff2_" + p.Name, Shape: p.Shape, Data: p.Data})
	}
	return params
}

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
	return NewTransformerBlockExt(t.dim, t.mha.numHeads, t.seqLen, t.ffDim, t.mha.Causal, t.actType, t.normType, t.useRoPE)
}

func (t *TransformerBlock) LightweightClone(params []float32, grads []float32) Layer {
	sizeMHA := len(t.mha.Params())
	sizeNorm1 := len(t.norm1.Params())
	sizeNorm2 := len(t.norm2.Params())
	sizeFF1 := len(t.ff1.Params())
	sizeFF2 := len(t.ff2.Params())

	gradSizeMHA := len(t.mha.Gradients())
	gradSizeNorm1 := len(t.norm1.Gradients())
	gradSizeNorm2 := len(t.norm2.Gradients())
	gradSizeFF1 := len(t.ff1.Gradients())
	gradSizeFF2 := len(t.ff2.Gradients())

	newT := &TransformerBlock{
		dim:               t.dim,
		seqLen:            t.seqLen,
		ffDim:             t.ffDim,
		actType:           t.actType,
		normType:          t.normType,
		useRoPE:           t.useRoPE,
		outputBuf:         make([]float32, t.seqLen*t.dim),
		res1Buf:           make([]float32, t.seqLen*t.dim),
		norm1OutBuf:       make([]float32, t.seqLen*t.dim),
		ffOutBuf:          make([]float32, t.seqLen*t.dim),
		res2Buf:           make([]float32, t.seqLen*t.dim),
		gradRes2Buf:       make([]float32, t.seqLen*t.dim),
		gradNorm1OutFFBuf: make([]float32, t.seqLen*t.dim),
		combinedGradBuf:   make([]float32, t.seqLen*t.dim),
		gradRes1Buf:       make([]float32, t.seqLen*t.dim),
		gradInBuf:         make([]float32, t.seqLen*t.dim),
		params:            params,
		grads:             grads,
		training:          t.training,
		device:            t.device,
	}

	offset := 0
	gradOffset := 0
	newT.mha = t.mha.LightweightClone(params[offset:offset+sizeMHA], grads[gradOffset:gradOffset+gradSizeMHA]).(*MultiHeadAttention)
	offset += sizeMHA
	gradOffset += gradSizeMHA

	newT.norm1 = t.norm1.LightweightClone(params[offset:offset+sizeNorm1], grads[gradOffset:gradOffset+gradSizeNorm1])
	offset += sizeNorm1
	gradOffset += gradSizeNorm1

	newT.norm2 = t.norm2.LightweightClone(params[offset:offset+sizeNorm2], grads[gradOffset:gradOffset+gradSizeNorm2])
	offset += sizeNorm2
	gradOffset += gradSizeNorm2

	newT.ff1 = t.ff1.LightweightClone(params[offset:offset+sizeFF1], grads[gradOffset:gradOffset+gradSizeFF1])
	offset += sizeFF1
	gradOffset += gradSizeFF1

	newT.ff2 = t.ff2.LightweightClone(params[offset:offset+sizeFF2], grads[gradOffset:gradOffset+gradSizeFF2]).(*Dense)

	return newT
}
func (t *TransformerBlock) AccumulateBackward(grad []float32) []float32 { return t.Backward(grad) }

// GlobalAveragePooling1D pools across the sequence dimension.
type GlobalAveragePooling1D struct {
	seqLen int
	dim    int

	outputBuf []float32
	gradInBuf []float32

	training bool
}

func NewGlobalAveragePooling1D(seqLen, dim int) *GlobalAveragePooling1D {
	return &GlobalAveragePooling1D{
		seqLen:    seqLen,
		dim:       dim,
		outputBuf: make([]float32, dim),
		gradInBuf: make([]float32, seqLen*dim),
	}
}

// SetTraining sets whether the layer is in training mode.
func (g *GlobalAveragePooling1D) SetTraining(training bool) {
	g.training = training
}

func (g *GlobalAveragePooling1D) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	if len(g.outputBuf) < g.dim {
		g.outputBuf = make([]float32, g.dim)
	}
	for i := 0; i < g.dim; i++ {
		sum := float32(0.0)
		for s := 0; s < g.seqLen; s++ {
			sum += x[s*g.dim+i]
		}
		g.outputBuf[i] = sum / float32(g.seqLen)
	}
	return g.outputBuf[:g.dim]
}

func (g *GlobalAveragePooling1D) Forward(x []float32) []float32 {
	return g.ForwardWithArena(x, nil, nil)
}

func (g *GlobalAveragePooling1D) ForwardBatch(x []float32, batchSize int) []float32 {
	return g.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (g *GlobalAveragePooling1D) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return g.ForwardWithArena(x, arena, offset)
	}
	inSize := g.InSize()
	outSize := g.OutSize()
	if len(g.outputBuf) < batchSize*outSize {
		g.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := g.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(g.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return g.outputBuf[:batchSize*outSize]
}

func (g *GlobalAveragePooling1D) Backward(grad []float32) []float32 {
	if len(g.gradInBuf) < g.seqLen*g.dim {
		g.gradInBuf = make([]float32, g.seqLen*g.dim)
	}
	for s := 0; s < g.seqLen; s++ {
		for i := 0; i < g.dim; i++ {
			g.gradInBuf[s*g.dim+i] = grad[i] / float32(g.seqLen)
		}
	}
	return g.gradInBuf[:g.seqLen*g.dim]
}

func (g *GlobalAveragePooling1D) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return g.Backward(grad)
	}
	inSize := g.InSize()
	outSize := g.OutSize()
	if len(g.gradInBuf) < batchSize*inSize {
		g.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		out := g.Backward(grad[i*outSize : (i+1)*outSize])
		copy(g.gradInBuf[i*inSize:(i+1)*inSize], out)
	}
	return g.gradInBuf[:batchSize*inSize]
}

func (g *GlobalAveragePooling1D) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return g.BackwardBatch(grad, batchSize)
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
func (g *GlobalAveragePooling1D) Clone() Layer {
	return NewGlobalAveragePooling1D(g.seqLen, g.dim)
}

func (g *GlobalAveragePooling1D) LightweightClone(params []float32, grads []float32) Layer {
	return NewGlobalAveragePooling1D(g.seqLen, g.dim)
}
func (g *GlobalAveragePooling1D) AccumulateBackward(grad []float32) []float32 { return g.Backward(grad) }

// CLSPooling extracts the first vector (CLS token) from a sequence.
type CLSPooling struct {
	seqLen int
	dim    int

	outputBuf []float32
	gradInBuf []float32

	training bool
}

func NewCLSPooling(seqLen, dim int) *CLSPooling {
	return &CLSPooling{
		seqLen:    seqLen,
		dim:       dim,
		outputBuf: make([]float32, dim),
		gradInBuf: make([]float32, seqLen*dim),
	}
}

// SetTraining sets whether the layer is in training mode.
func (c *CLSPooling) SetTraining(training bool) {
	c.training = training
}

func (c *CLSPooling) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	// x is [seqLen * dim]
	if len(c.outputBuf) < c.dim {
		c.outputBuf = make([]float32, c.dim)
	}
	copy(c.outputBuf, x[0:c.dim])
	return c.outputBuf[:c.dim]
}

func (c *CLSPooling) Forward(x []float32) []float32 {
	return c.ForwardWithArena(x, nil, nil)
}

func (c *CLSPooling) ForwardBatch(x []float32, batchSize int) []float32 {
	return c.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (c *CLSPooling) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return c.ForwardWithArena(x, arena, offset)
	}
	inSize := c.InSize()
	outSize := c.OutSize()
	if len(c.outputBuf) < batchSize*outSize {
		c.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := c.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(c.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return c.outputBuf[:batchSize*outSize]
}

func (c *CLSPooling) Backward(grad []float32) []float32 {
	// grad is [dim]
	if len(c.gradInBuf) < c.seqLen*c.dim {
		c.gradInBuf = make([]float32, c.seqLen*c.dim)
	}
	for i := range c.gradInBuf {
		c.gradInBuf[i] = 0
	}
	copy(c.gradInBuf[0:c.dim], grad)
	return c.gradInBuf[:c.seqLen*c.dim]
}

func (c *CLSPooling) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return c.Backward(grad)
	}
	inSize := c.InSize()
	outSize := c.OutSize()
	if len(c.gradInBuf) < batchSize*inSize {
		c.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		out := c.Backward(grad[i*outSize : (i+1)*outSize])
		copy(c.gradInBuf[i*inSize:(i+1)*inSize], out)
	}
	return c.gradInBuf[:batchSize*inSize]
}

func (c *CLSPooling) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return c.BackwardBatch(grad, batchSize)
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
func (c *CLSPooling) Clone() Layer {
	return NewCLSPooling(c.seqLen, c.dim)
}

func (c *CLSPooling) LightweightClone(params []float32, grads []float32) Layer {
	return NewCLSPooling(c.seqLen, c.dim)
}
func (c *CLSPooling) AccumulateBackward(grad []float32) []float32 { return c.Backward(grad) }
