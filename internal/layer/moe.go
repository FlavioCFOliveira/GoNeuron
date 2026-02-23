package layer

import (
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"math"
	"sort"
)

type expertWeight struct {
	id     int
	weight float32
}

// MoE (Mixture of Experts) layer
type MoE struct {
	gating      *Dense
	experts     []Layer
	k           int // Top-k experts to use
	inSize      int
	outSize     int
	training    bool
	numExperts  int

	// Persistent buffers for zero-allocation and backprop
	inputBuf      []float32
	outputBuf     []float32
	gateLogits    []float32
	gateProbs     []float32
	lastActive    []int
	expertOutputs [][]float32 // [numExperts][outSize]
	gateGradBuf   []float32
	dxTotalBuf    []float32
	expertGradBuf []float32 // reused for each expert backward
	weightsBuf    []expertWeight

	rng *RNG

	params []float32
	grads  []float32
}

func NewMoE(inSize, outSize, numExperts, k int) *MoE {
	experts := make([]Layer, numExperts)
	expertOutputs := make([][]float32, numExperts)
	for i := 0; i < numExperts; i++ {
		experts[i] = NewDense(inSize, outSize, activations.NewLeakyReLU(0.1))
		expertOutputs[i] = make([]float32, outSize)
	}

	// Gating uses Linear activation so we can add noise before Softmax
	gating := NewDense(inSize, numExperts, activations.Linear{})

	// Aggregate parameters
	gLen := len(gating.Params())
	eLen := len(experts[0].Params())
	totalParams := gLen + numExperts*eLen
	params := make([]float32, totalParams)
	grads := make([]float32, totalParams)

	offset := 0
	gating.SetParams(params[offset : offset+gLen])
	gating.SetGradients(grads[offset : offset+gLen])
	offset += gLen

	for i := 0; i < numExperts; i++ {
		experts[i].SetParams(params[offset : offset+eLen])
		experts[i].SetGradients(grads[offset : offset+eLen])
		offset += eLen
	}

	return &MoE{
		gating:        gating,
		experts:       experts,
		k:             k,
		inSize:        inSize,
		outSize:       outSize,
		numExperts:    numExperts,
		inputBuf:      make([]float32, inSize),
		outputBuf:     make([]float32, outSize),
		gateLogits:    make([]float32, numExperts),
		gateProbs:     make([]float32, numExperts),
		lastActive:    make([]int, k),
		expertOutputs: expertOutputs,
		gateGradBuf:   make([]float32, numExperts),
		dxTotalBuf:    make([]float32, inSize),
		expertGradBuf: make([]float32, outSize),
		weightsBuf:    make([]expertWeight, numExperts),
		rng:           NewRNG(42),
		params:        params,
		grads:         grads,
	}
}

func (m *MoE) Forward(x []float32) []float32 {
	copy(m.inputBuf, x)

	// 1. Gating Network: get logits
	logits := m.gating.Forward(x)
	copy(m.gateLogits, logits)

	// Add noise during training to encourage exploration and prevent collapse
	if m.training {
		for i := range m.gateLogits {
			// Simple noise: +/- 5% of range or small absolute noise
			m.gateLogits[i] += (m.rng.RandFloat()*2 - 1) * 0.05
		}
	}

	// Apply Softmax manually
	maxLogit := m.gateLogits[0]
	for _, v := range m.gateLogits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	sumExp := float32(0)
	for i, v := range m.gateLogits {
		m.gateProbs[i] = float32(math.Exp(float64(v - maxLogit)))
		sumExp += m.gateProbs[i]
	}
	for i := range m.gateProbs {
		m.gateProbs[i] /= sumExp
	}

	// 2. Selecionar Top-K experts
	for i, w := range m.gateProbs {
		m.weightsBuf[i] = expertWeight{i, w}
	}

	sort.Slice(m.weightsBuf, func(i, j int) bool {
		return m.weightsBuf[i].weight > m.weightsBuf[j].weight
	})

	// Reset output buffer
	for i := range m.outputBuf {
		m.outputBuf[i] = 0
	}

	// 3. Forward apenas nos K especialistas ativos e somar ponderado
	for i := 0; i < m.k; i++ {
		idx := m.weightsBuf[i].id
		m.lastActive[i] = idx
		w := m.weightsBuf[i].weight

		expertOut := m.experts[idx].Forward(x)
		// Store expert output for backward pass
		copy(m.expertOutputs[idx], expertOut)

		for j := 0; j < m.outSize; j++ {
			m.outputBuf[j] += expertOut[j] * w
		}
	}

	return m.outputBuf
}

func (m *MoE) Backward(grad []float32) []float32 {
	// Reset buffers
	for i := range m.dxTotalBuf {
		m.dxTotalBuf[i] = 0
	}
	for i := range m.gateGradBuf {
		m.gateGradBuf[i] = 0
	}

	// 1. Calculate gradients for active experts and gating weights
	for _, idx := range m.lastActive {
		w := m.gateProbs[idx]

		// dL/dExpert = dL/dY * dY/dExpert = grad * w
		for i := 0; i < m.outSize; i++ {
			m.expertGradBuf[i] = grad[i] * w
		}

		// Backward through expert
		dxExpert := m.experts[idx].Backward(m.expertGradBuf)
		for i := 0; i < m.inSize; i++ {
			m.dxTotalBuf[i] += dxExpert[i]
		}

		// 2. dL/dGate_i = dL/dY * dY/dGate_i = grad dot expertOutput_i
		dot := float32(0)
		expertOut := m.expertOutputs[idx]
		for i := 0; i < m.outSize; i++ {
			dot += grad[i] * expertOut[i]
		}
		m.gateGradBuf[idx] = dot
	}

	// 3. Add Load Balancing Gradient
	// We want to penalize large deviations from uniform distribution
	// Loss_lb = lambda * sum((p_i - 1/N)^2)
	// dLoss_lb/dp_i = 2 * lambda * (p_i - 1/N)
	if m.training {
		lambda := float32(0.1) // Load balancing strength
		targetProb := float32(1.0) / float32(m.numExperts)
		for i := 0; i < m.numExperts; i++ {
			m.gateGradBuf[i] += 2 * lambda * (m.gateProbs[i] - targetProb)
		}
	}

	// 4. Backprop through Softmax
	// dL/dLogit_i = p_i * (dL/dp_i - sum_j(p_j * dL/dp_j))
	sumPGrad := float32(0)
	for i := 0; i < m.numExperts; i++ {
		sumPGrad += m.gateProbs[i] * m.gateGradBuf[i]
	}

	for i := 0; i < m.numExperts; i++ {
		m.gateGradBuf[i] = m.gateProbs[i] * (m.gateGradBuf[i] - sumPGrad)
	}

	// 5. Backprop through gating network (Linear)
	dxGating := m.gating.Backward(m.gateGradBuf)
	for i := 0; i < m.inSize; i++ {
		m.dxTotalBuf[i] += dxGating[i]
	}

	return m.dxTotalBuf
}

func (m *MoE) Params() []float32 {
	return m.params
}

func (m *MoE) SetParams(p []float32) {
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

func (m *MoE) updateViews() {
	gLen := len(m.gating.Params())
	eLen := len(m.experts[0].Params())

	offset := 0
	m.gating.SetParams(m.params[offset : offset+gLen])
	offset += gLen

	for i := 0; i < m.numExperts; i++ {
		m.experts[i].SetParams(m.params[offset : offset+eLen])
		offset += eLen
	}
}

func (m *MoE) Gradients() []float32 {
	return m.grads
}

func (m *MoE) SetGradients(g []float32) {
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

func (m *MoE) updateGradViews() {
	gLen := len(m.gating.Gradients())
	eLen := len(m.experts[0].Gradients())

	offset := 0
	m.gating.SetGradients(m.grads[offset : offset+gLen])
	offset += gLen

	for i := 0; i < m.numExperts; i++ {
		m.experts[i].SetGradients(m.grads[offset : offset+eLen])
		offset += eLen
	}
}

func (m *MoE) Reset() {
	m.gating.Reset()
	for _, e := range m.experts {
		e.Reset()
	}
}

func (m *MoE) ClearGradients() {
	m.gating.ClearGradients()
	for _, e := range m.experts {
		e.ClearGradients()
	}
}

func (m *MoE) Clone() Layer {
	newM := &MoE{
		gating:        m.gating.Clone().(*Dense),
		experts:       make([]Layer, len(m.experts)),
		k:             m.k,
		inSize:        m.inSize,
		outSize:       m.outSize,
		numExperts:    m.numExperts,
		inputBuf:      make([]float32, m.inSize),
		outputBuf:     make([]float32, m.outSize),
		gateLogits:    make([]float32, m.numExperts),
		gateProbs:     make([]float32, m.numExperts),
		lastActive:    make([]int, m.k),
		expertOutputs: make([][]float32, len(m.expertOutputs)),
		gateGradBuf:   make([]float32, m.numExperts),
		dxTotalBuf:    make([]float32, m.inSize),
		expertGradBuf: make([]float32, m.outSize),
		weightsBuf:    make([]expertWeight, m.numExperts),
		rng:           NewRNG(42),
	}
	for i, e := range m.experts {
		newM.experts[i] = e.Clone()
		newM.expertOutputs[i] = make([]float32, m.outSize)
	}
	return newM
}

func (m *MoE) SetDevice(d Device) {
	m.gating.SetDevice(d)
	for _, e := range m.experts {
		e.SetDevice(d)
	}
}

func (m *MoE) SetTraining(t bool) {
	m.training = t
	for _, e := range m.experts {
		if st, ok := e.(interface{ SetTraining(bool) }); ok {
			st.SetTraining(t)
		}
	}
}

func (m *MoE) InSize() int  { return m.inSize }
func (m *MoE) OutSize() int { return m.outSize }

func (m *MoE) AccumulateBackward(grad []float32) []float32 {
	return m.Backward(grad)
}
