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
	gradInBuf     []float32 // for BackwardBatch
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

	m := &MoE{
		gating:        gating,
		experts:       experts,
		k:             k,
		inSize:        inSize,
		outSize:       outSize,
		numExperts:    numExperts,
		outputBuf:     make([]float32, outSize),
		gateLogits:    make([]float32, numExperts),
		gateProbs:     make([]float32, numExperts),
		lastActive:    make([]int, k),
		expertOutputs: expertOutputs,
		gateGradBuf:   make([]float32, numExperts),
		expertGradBuf: make([]float32, outSize),
		weightsBuf:    make([]expertWeight, numExperts),
		rng:           NewRNG(42),
	}

	if inSize != -1 {
		m.Build(inSize)
	}

	return m
}

func (m *MoE) Build(inSize int) {
	m.inSize = inSize

	// Build sub-layers if they are deferred
	if m.gating.InSize() <= 0 {
		m.gating.Build(inSize)
	}
	for _, e := range m.experts {
		if e.InSize() <= 0 {
			if de, ok := e.(DeferredLayer); ok {
				de.Build(inSize)
			}
		}
	}

	// Aggregate parameters
	gLen := len(m.gating.Params())
	eLen := len(m.experts[0].Params())
	totalParams := gLen + m.numExperts*eLen
	m.params = make([]float32, totalParams)
	m.grads = make([]float32, totalParams)

	offset := 0
	m.gating.SetParams(m.params[offset : offset+gLen])
	m.gating.SetGradients(m.grads[offset : offset+gLen])
	offset += gLen

	for i := 0; i < m.numExperts; i++ {
		m.experts[i].SetParams(m.params[offset : offset+eLen])
		m.experts[i].SetGradients(m.grads[offset : offset+eLen])
		offset += eLen
	}

	m.inputBuf = make([]float32, inSize)
	m.dxTotalBuf = make([]float32, inSize)
	m.gradInBuf = make([]float32, inSize)
}

func (m *MoE) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	copy(m.inputBuf, x)

	// 1. Gating Network
	var logits []float32
	if arena != nil && offset != nil {
		logits = m.gating.ForwardWithArena(x, arena, offset)
	} else {
		logits = m.gating.Forward(x)
	}
	copy(m.gateLogits, logits)

	if m.training {
		for i := range m.gateLogits {
			m.gateLogits[i] += (m.rng.RandFloat()*2 - 1) * 0.05
		}
	}

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

	for i, w := range m.gateProbs {
		m.weightsBuf[i] = expertWeight{i, w}
	}

	sort.Slice(m.weightsBuf, func(i, j int) bool {
		return m.weightsBuf[i].weight > m.weightsBuf[j].weight
	})

	for i := range m.outputBuf {
		if i < m.outSize {
			m.outputBuf[i] = 0
		}
	}

	for i := 0; i < m.k; i++ {
		idx := m.weightsBuf[i].id
		m.lastActive[i] = idx
		w := m.weightsBuf[i].weight

		var expertOut []float32
		if al, ok := m.experts[idx].(ArenaLayer); ok && arena != nil && offset != nil {
			expertOut = al.ForwardWithArena(x, arena, offset)
		} else {
			expertOut = m.experts[idx].Forward(x)
		}

		if arena != nil && offset != nil {
			if len(*arena) < *offset+m.outSize {
				newArena := make([]float32, (*offset+m.outSize)*2)
				copy(newArena, *arena)
				*arena = newArena
			}
			saved := (*arena)[*offset : *offset+m.outSize]
			copy(saved, expertOut)
			m.expertOutputs[idx] = saved
			*offset += m.outSize
		} else {
			copy(m.expertOutputs[idx], expertOut)
		}

		for j := 0; j < m.outSize; j++ {
			m.outputBuf[j] += expertOut[j] * w
		}
	}

	return m.outputBuf[:m.outSize]
}

func (m *MoE) Forward(x []float32) []float32 {
	return m.ForwardWithArena(x, nil, nil)
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
	newM := NewMoE(m.inSize, m.outSize, m.numExperts, m.k)
	copy(newM.params, m.params)
	return newM
}

func (m *MoE) LightweightClone(params []float32, grads []float32) Layer {
	gLen := len(m.gating.Params())
	gGradLen := len(m.gating.Gradients())
	eLen := len(m.experts[0].Params())
	eGradLen := len(m.experts[0].Gradients())

	newM := &MoE{
		experts:       make([]Layer, m.numExperts),
		k:             m.k,
		inSize:        m.inSize,
		outSize:       m.outSize,
		numExperts:    m.numExperts,
		inputBuf:      make([]float32, m.inSize),
		outputBuf:     make([]float32, m.outSize),
		gateLogits:    make([]float32, m.numExperts),
		gateProbs:     make([]float32, m.numExperts),
		lastActive:    make([]int, m.k),
		expertOutputs: make([][]float32, m.numExperts),
		gateGradBuf:   make([]float32, m.numExperts),
		dxTotalBuf:    make([]float32, m.inSize),
		gradInBuf:     make([]float32, m.inSize),
		expertGradBuf: make([]float32, m.outSize),
		weightsBuf:    make([]expertWeight, m.numExperts),
		rng:           NewRNG(42),
		params:        params,
		grads:         grads,
		training:      m.training,
	}

	offset := 0
	gradOffset := 0
	newM.gating = m.gating.LightweightClone(params[offset:offset+gLen], grads[gradOffset:gradOffset+gGradLen]).(*Dense)
	offset += gLen
	gradOffset += gGradLen

	for i := 0; i < m.numExperts; i++ {
		newM.experts[i] = m.experts[i].LightweightClone(params[offset:offset+eLen], grads[gradOffset:gradOffset+eGradLen])
		newM.expertOutputs[i] = make([]float32, m.outSize)
		offset += eLen
		gradOffset += eGradLen
	}

	return newM
}

func (m *MoE) ForwardBatch(x []float32, batchSize int) []float32 {
	return m.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (m *MoE) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return m.ForwardWithArena(x, arena, offset)
	}
	inSize := m.inSize
	outSize := m.outSize
	if len(m.outputBuf) < batchSize*outSize {
		m.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		m.Reset()
		out := m.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(m.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return m.outputBuf[:batchSize*outSize]
}

func (m *MoE) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return m.Backward(grad)
	}
	inSize := m.inSize
	outSize := m.outSize
	if len(m.gradInBuf) < batchSize*inSize {
		m.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		out := m.Backward(grad[i*outSize : (i+1)*outSize])
		copy(m.gradInBuf[i*inSize:(i+1)*inSize], out)
	}
	return m.gradInBuf[:batchSize*inSize]
}

func (m *MoE) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return m.BackwardBatch(grad, batchSize)
}

func (m *MoE) SetDevice(d Device) {
	m.gating.SetDevice(d)
	for _, e := range m.experts {
		e.SetDevice(d)
	}
}

func (m *MoE) SetTraining(t bool) {
	m.training = t
	m.gating.SetTraining(t)
	for _, e := range m.experts {
		e.SetTraining(t)
	}
}

func (m *MoE) InSize() int  { return m.inSize }
func (m *MoE) OutSize() int { return m.outSize }

func (m *MoE) AccumulateBackward(grad []float32) []float32 {
	return m.Backward(grad)
}
