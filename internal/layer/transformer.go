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

	training bool

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

// SetTraining sets whether the layer is in training mode.
func (p *PositionalEncoding) SetTraining(training bool) {
	p.training = training
}

func (p *PositionalEncoding) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	// x is [seqLen * dim]
	if len(p.outputBuf) != len(x) {
		p.outputBuf = make([]float32, len(x))
	}
	for i := range x {
		p.outputBuf[i] = x[i] + p.weights[i]
	}
	return p.outputBuf
}

func (p *PositionalEncoding) Forward(x []float32) []float32 {
	return p.ForwardWithArena(x, nil, nil)
}


func (p *PositionalEncoding) Backward(grad []float32) []float32 {
	// Gradient w.r.t. parameters is same as gradient w.r.t. output
	// Gradient w.r.t. input is also same as gradient w.r.t. output
	for i := range grad {
		p.gradBuf[i] += grad[i]
	}
	return grad
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

	// Contiguous parameter and gradient buffers
	params []float32
	grads  []float32

	training bool

	device Device
}

func NewMultiHeadAttention(dim, numHeads, seqLen int, causal bool) *MultiHeadAttention {
	headDim := dim / numHeads
	if headDim*numHeads != dim {
		panic("dim must be divisible by numHeads")
	}

	wQ := NewDense(dim, dim, activations.Linear{})
	wK := NewDense(dim, dim, activations.Linear{})
	wV := NewDense(dim, dim, activations.Linear{})
	wOut := NewDense(dim, dim, activations.Linear{})

	// Calculate total parameters
	paramSize := len(wQ.Params())
	totalSize := paramSize * 4
	params := make([]float32, totalSize)
	grads := make([]float32, totalSize)

	// Link sub-layers to these contiguous buffers
	wQ.SetParams(params[0:paramSize])
	wK.SetParams(params[paramSize : 2*paramSize])
	wV.SetParams(params[2*paramSize : 3*paramSize])
	wOut.SetParams(params[3*paramSize : 4*paramSize])

	wQ.SetGradients(grads[0:paramSize])
	wK.SetGradients(grads[paramSize : 2*paramSize])
	wV.SetGradients(grads[2*paramSize : 3*paramSize])
	wOut.SetGradients(grads[3*paramSize : 4*paramSize])

	return &MultiHeadAttention{
		dim:         dim,
		numHeads:    numHeads,
		headDim:     headDim,
		seqLen:      seqLen,
		Causal:      causal,
		wQ:          wQ,
		wK:          wK,
		wV:          wV,
		wOut:        wOut,
		params:      params,
		grads:       grads,
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
		dim:         m.dim,
		numHeads:    m.numHeads,
		headDim:     m.headDim,
		seqLen:      m.seqLen,
		Causal:      m.Causal,
		params:      params,
		grads:       grads,
		outputBuf:   make([]float32, m.seqLen*m.dim),
		gradInBuf:   make([]float32, m.seqLen*m.dim),
		qBuf:        make([]float32, m.seqLen*m.dim),
		kBuf:        make([]float32, m.seqLen*m.dim),
		vBuf:        make([]float32, m.seqLen*m.dim),
		scoresBuf:   make([]float32, m.seqLen*m.seqLen),
		finalOutBuf: make([]float32, m.seqLen*m.dim),
		training:    m.training,
		device:      m.device,
	}

	newM.wQ = m.wQ.LightweightClone(params[0:paramSize], grads[0:gradSize]).(*Dense)
	newM.wK = m.wK.LightweightClone(params[paramSize:2*paramSize], grads[gradSize:2*gradSize]).(*Dense)
	newM.wV = m.wV.LightweightClone(params[2*paramSize:3*paramSize], grads[2*gradSize:3*gradSize]).(*Dense)
	newM.wOut = m.wOut.LightweightClone(params[3*paramSize:4*paramSize], grads[3*gradSize:4*gradSize]).(*Dense)

	return newM
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

	// Pre-allocated backward buffers
	gradRes2Buf       []float32
	gradNorm1OutFFBuf []float32
	combinedGradBuf   []float32
	gradRes1Buf       []float32
	gradInBuf         []float32

	// Contiguous parameter and gradient buffers
	params []float32
	grads  []float32

	training bool
}

func NewTransformerBlock(dim, numHeads, seqLen, ffDim int, causal bool) *TransformerBlock {
	mha := NewMultiHeadAttention(dim, numHeads, seqLen, causal)
	norm1 := NewLayerNorm(dim, 1e-5, true)
	norm2 := NewLayerNorm(dim, 1e-5, true)
	ff1 := NewDense(dim, ffDim, activations.ReLU{})
	ff2 := NewDense(ffDim, dim, activations.Linear{})

	// Aggregate parameters
	sizeMHA := len(mha.Params())
	sizeNorm1 := len(norm1.Params())
	sizeNorm2 := len(norm2.Params())
	sizeFF1 := len(ff1.Params())
	sizeFF2 := len(ff2.Params())

	totalParams := sizeMHA + sizeNorm1 + sizeNorm2 + sizeFF1 + sizeFF2
	params := make([]float32, totalParams)
	grads := make([]float32, totalParams)

	offset := 0
	mha.SetParams(params[offset : offset+sizeMHA])
	mha.SetGradients(grads[offset : offset+sizeMHA])
	offset += sizeMHA

	norm1.SetParams(params[offset : offset+sizeNorm1])
	norm1.SetGradients(grads[offset : offset+sizeNorm1])
	offset += sizeNorm1

	norm2.SetParams(params[offset : offset+sizeNorm2])
	norm2.SetGradients(grads[offset : offset+sizeNorm2])
	offset += sizeNorm2

	ff1.SetParams(params[offset : offset+sizeFF1])
	ff1.SetGradients(grads[offset : offset+sizeFF1])
	offset += sizeFF1

	ff2.SetParams(params[offset : offset+sizeFF2])
	ff2.SetGradients(grads[offset : offset+sizeFF2])

	return &TransformerBlock{
		mha:               mha,
		norm1:             norm1,
		norm2:             norm2,
		ff1:               ff1,
		ff2:               ff2,
		dim:               dim,
		seqLen:            seqLen,
		outputBuf:         make([]float32, seqLen*dim),
		res1Buf:           make([]float32, seqLen*dim),
		norm1OutBuf:       make([]float32, seqLen*dim),
		ffOutBuf:          make([]float32, seqLen*dim),
		res2Buf:           make([]float32, seqLen*dim),
		gradRes2Buf:       make([]float32, seqLen*dim),
		gradNorm1OutFFBuf: make([]float32, seqLen*dim),
		combinedGradBuf:   make([]float32, seqLen*dim),
		gradRes1Buf:       make([]float32, seqLen*dim),
		gradInBuf:         make([]float32, seqLen*dim),
		params:            params,
		grads:             grads,
	}
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
		norm1Out = t.norm1.ForwardWithArena(t.res1Buf, arena, offset)
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
			mid = t.ff1.ForwardWithArena(t.norm1OutBuf[start:end], arena, offset)
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
		output = t.norm2.ForwardWithArena(t.res2Buf, arena, offset)
	} else {
		output = t.norm2.Forward(t.res2Buf)
	}
	copy(t.outputBuf, output)

	return t.outputBuf
}

func (t *TransformerBlock) Forward(x []float32) []float32 {
	return t.ForwardWithArena(x, nil, nil)
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
func (t *TransformerBlock) SetDevice(d Device)         { t.mha.SetDevice(d) }
func (t *TransformerBlock) InSize() int                { return t.seqLen * t.dim }
func (t *TransformerBlock) OutSize() int               { return t.seqLen * t.dim }

func (t *TransformerBlock) NamedParams() []NamedParam {
	var params []NamedParam
	for _, p := range t.mha.NamedParams() {
		params = append(params, NamedParam{Name: "mha_" + p.Name, Shape: p.Shape, Data: p.Data})
	}
	for _, p := range t.norm1.NamedParams() {
		params = append(params, NamedParam{Name: "norm1_" + p.Name, Shape: p.Shape, Data: p.Data})
	}
	for _, p := range t.norm2.NamedParams() {
		params = append(params, NamedParam{Name: "norm2_" + p.Name, Shape: p.Shape, Data: p.Data})
	}
	for _, p := range t.ff1.NamedParams() {
		params = append(params, NamedParam{Name: "ff1_" + p.Name, Shape: p.Shape, Data: p.Data})
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
	return NewTransformerBlock(t.dim, t.mha.numHeads, t.seqLen, t.ff1.OutSize(), t.mha.Causal)
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
	}

	offset := 0
	gradOffset := 0
	newT.mha = t.mha.LightweightClone(params[offset:offset+sizeMHA], grads[gradOffset:gradOffset+gradSizeMHA]).(*MultiHeadAttention)
	offset += sizeMHA
	gradOffset += gradSizeMHA

	newT.norm1 = t.norm1.LightweightClone(params[offset:offset+sizeNorm1], grads[gradOffset:gradOffset+gradSizeNorm1]).(*LayerNorm)
	offset += sizeNorm1
	gradOffset += gradSizeNorm1

	newT.norm2 = t.norm2.LightweightClone(params[offset:offset+sizeNorm2], grads[gradOffset:gradOffset+gradSizeNorm2]).(*LayerNorm)
	offset += sizeNorm2
	gradOffset += gradSizeNorm2

	newT.ff1 = t.ff1.LightweightClone(params[offset:offset+sizeFF1], grads[gradOffset:gradOffset+gradSizeFF1]).(*Dense)
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

	training bool
}

func NewGlobalAveragePooling1D(seqLen, dim int) *GlobalAveragePooling1D {
	return &GlobalAveragePooling1D{seqLen: seqLen, dim: dim}
}

// SetTraining sets whether the layer is in training mode.
func (g *GlobalAveragePooling1D) SetTraining(training bool) {
	g.training = training
}

func (g *GlobalAveragePooling1D) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
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

func (g *GlobalAveragePooling1D) Forward(x []float32) []float32 {
	return g.ForwardWithArena(x, nil, nil)
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
func (g *GlobalAveragePooling1D) Clone() Layer {
	return &GlobalAveragePooling1D{seqLen: g.seqLen, dim: g.dim, training: g.training}
}

func (g *GlobalAveragePooling1D) LightweightClone(params []float32, grads []float32) Layer {
	return &GlobalAveragePooling1D{seqLen: g.seqLen, dim: g.dim, training: g.training}
}
func (g *GlobalAveragePooling1D) AccumulateBackward(grad []float32) []float32 { return g.Backward(grad) }

// CLSPooling extracts the first vector (CLS token) from a sequence.
type CLSPooling struct {
	seqLen int
	dim    int

	outputBuf []float32
	gradBuf   []float32

	training bool
}

func NewCLSPooling(seqLen, dim int) *CLSPooling {
	return &CLSPooling{
		seqLen:    seqLen,
		dim:       dim,
		outputBuf: make([]float32, dim),
		gradBuf:   make([]float32, seqLen*dim),
	}
}

// SetTraining sets whether the layer is in training mode.
func (c *CLSPooling) SetTraining(training bool) {
	c.training = training
}

func (c *CLSPooling) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	// x is [seqLen * dim]
	copy(c.outputBuf, x[0:c.dim])
	return c.outputBuf
}

func (c *CLSPooling) Forward(x []float32) []float32 {
	return c.ForwardWithArena(x, nil, nil)
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
func (c *CLSPooling) Clone() Layer {
	return &CLSPooling{seqLen: c.seqLen, dim: c.dim, outputBuf: make([]float32, c.dim), gradBuf: make([]float32, c.seqLen*c.dim), training: c.training}
}

func (c *CLSPooling) LightweightClone(params []float32, grads []float32) Layer {
	return &CLSPooling{
		seqLen:    c.seqLen,
		dim:       c.dim,
		outputBuf: make([]float32, c.dim),
		gradBuf:   make([]float32, c.seqLen*c.dim),
		training:  c.training,
	}
}
func (c *CLSPooling) AccumulateBackward(grad []float32) []float32 { return c.Backward(grad) }
