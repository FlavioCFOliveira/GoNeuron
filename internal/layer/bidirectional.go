package layer

// Bidirectional is a wrapper layer that runs two RNNs (forward and backward) and concatenates their outputs.
type Bidirectional struct {
	ForwardLayer  Layer
	BackwardLayer Layer

	// Buffers for forward pass
	storedInput [][]float64
	forwardOut  [][]float64
	backwardOut [][]float64

	// Output buffer
	outputBuf []float64

	// Current sequence length (during Forward)
	timeStep int
}

// NewBidirectional creates a new bidirectional wrapper for two layers.
func NewBidirectional(forward, backward Layer) *Bidirectional {
	return &Bidirectional{
		ForwardLayer:  forward,
		BackwardLayer: backward,
		storedInput:   make([][]float64, 0),
		forwardOut:    make([][]float64, 0),
		backwardOut:   make([][]float64, 0),
	}
}

// Forward performs a forward pass for one time step.
func (b *Bidirectional) Forward(x []float64) []float64 {
	fOut := b.ForwardLayer.Forward(x)

	xCopy := make([]float64, len(x))
	copy(xCopy, x)
	b.storedInput = append(b.storedInput, xCopy)

	fOutCopy := make([]float64, len(fOut))
	copy(fOutCopy, fOut)
	b.forwardOut = append(b.forwardOut, fOutCopy)

	outSize := b.ForwardLayer.OutSize() + b.BackwardLayer.OutSize()
	if len(b.outputBuf) != outSize {
		b.outputBuf = make([]float64, outSize)
	}

	copy(b.outputBuf[:b.ForwardLayer.OutSize()], fOut)
	for i := b.ForwardLayer.OutSize(); i < outSize; i++ {
		b.outputBuf[i] = 0
	}

	b.timeStep++
	return b.outputBuf
}

// ComputeBackwardHiddenStates runs the backward RNN over the stored sequence.
func (b *Bidirectional) ComputeBackwardHiddenStates() {
	b.BackwardLayer.Reset()
	n := len(b.storedInput)
	b.backwardOut = make([][]float64, n)
	for i := n - 1; i >= 0; i-- {
		out := b.BackwardLayer.Forward(b.storedInput[i])
		outCopy := make([]float64, len(out))
		copy(outCopy, out)
		b.backwardOut[i] = outCopy
	}
}

// ProcessSequence processes an entire sequence and returns all concatenated outputs.
func (b *Bidirectional) ProcessSequence(seq [][]float64) [][]float64 {
	b.Reset()
	n := len(seq)
	b.storedInput = seq
	b.forwardOut = make([][]float64, n)

	// Forward pass
	for i := 0; i < n; i++ {
		out := b.ForwardLayer.Forward(seq[i])
		outCopy := make([]float64, len(out))
		copy(outCopy, out)
		b.forwardOut[i] = outCopy
	}

	// Backward pass
	b.ComputeBackwardHiddenStates()

	results := make([][]float64, n)
	fSize := b.ForwardLayer.OutSize()
	bSize := b.BackwardLayer.OutSize()
	for i := 0; i < n; i++ {
		res := make([]float64, fSize+bSize)
		copy(res[:fSize], b.forwardOut[i])
		copy(res[fSize:], b.backwardOut[i])
		results[i] = res
	}

	return results
}

// Backward performs backpropagation for one time step (stub for compatibility).
func (b *Bidirectional) Backward(grad []float64) []float64 {
	ts := b.timeStep - 1
	if ts < 0 {
		return make([]float64, b.InSize())
	}
	fDx := b.ForwardLayer.Backward(grad[:b.ForwardLayer.OutSize()])
	b.timeStep--
	return fDx
}

// Reset resets both layers and clears stored sequence.
func (b *Bidirectional) Reset() {
	b.ForwardLayer.Reset()
	b.BackwardLayer.Reset()
	b.storedInput = b.storedInput[:0]
	b.forwardOut = b.forwardOut[:0]
	b.backwardOut = b.backwardOut[:0]
	b.timeStep = 0
}

func (b *Bidirectional) Params() []float64 {
	fParams := b.ForwardLayer.Params()
	bParams := b.BackwardLayer.Params()
	params := make([]float64, len(fParams)+len(bParams))
	copy(params, fParams)
	copy(params[len(fParams):], bParams)
	return params
}

func (b *Bidirectional) SetParams(params []float64) {
	fLen := len(b.ForwardLayer.Params())
	b.ForwardLayer.SetParams(params[:fLen])
	b.BackwardLayer.SetParams(params[fLen:])
}

func (b *Bidirectional) Gradients() []float64 {
	fGrads := b.ForwardLayer.Gradients()
	bGrads := b.BackwardLayer.Gradients()
	grads := make([]float64, len(fGrads)+len(bGrads))
	copy(grads, fGrads)
	copy(grads[len(fGrads):], bGrads)
	return grads
}

func (b *Bidirectional) SetGradients(grads []float64) {
	fLen := len(b.ForwardLayer.Gradients())
	b.ForwardLayer.SetGradients(grads[:fLen])
	b.BackwardLayer.SetGradients(grads[fLen:])
}

func (b *Bidirectional) ClearGradients() {
	b.ForwardLayer.ClearGradients()
	b.BackwardLayer.ClearGradients()
}

func (b *Bidirectional) OutSize() int {
	return b.ForwardLayer.OutSize() + b.BackwardLayer.OutSize()
}

func (b *Bidirectional) InSize() int {
	return b.ForwardLayer.InSize()
}

func (b *Bidirectional) Clone() Layer {
	return NewBidirectional(b.ForwardLayer.Clone(), b.BackwardLayer.Clone())
}

func (b *Bidirectional) AccumulateBackward(grad []float64) []float64 {
	return b.Backward(grad)
}

func (b *Bidirectional) GetForwardOutputAt(t int) []float64 {
	return b.forwardOut[t]
}

func (b *Bidirectional) GetBackwardOutputAt(t int) []float64 {
	return b.backwardOut[t]
}
