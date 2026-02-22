package layer

// Bidirectional is a wrapper layer that runs two RNNs (forward and backward) and concatenates their outputs.
type Bidirectional struct {
	ForwardLayer  Layer
	BackwardLayer Layer

	// Buffers for forward pass
	inputSequence [][]float64
	forwardOut    [][]float64
	backwardOut   [][]float64

	// Output buffer
	outputBuf []float64
}

// NewBidirectional creates a new bidirectional wrapper for two layers.
// The forward and backward layers should ideally be of the same type (e.g., LSTM or GRU).
func NewBidirectional(forward, backward Layer) *Bidirectional {
	return &Bidirectional{
		ForwardLayer:  forward,
		BackwardLayer: backward,
		inputSequence: make([][]float64, 0),
		forwardOut:    make([][]float64, 0),
		backwardOut:   make([][]float64, 0),
	}
}

// Forward performs a forward pass for one time step.
// Note: In a bidirectional model, "forward" step-by-step is tricky because
// the backward layer needs the full sequence.
// We store inputs and return only the forward part plus zeros for the backward part
// during step-by-step Forward calls. The full bidirectional output is available
// after ProcessSequence.
func (b *Bidirectional) Forward(x []float64) []float64 {
	// Store input for later backward pass and sequence processing
	xCopy := make([]float64, len(x))
	copy(xCopy, x)
	b.inputSequence = append(b.inputSequence, xCopy)

	// Run the forward layer step-by-step
	fOut := b.ForwardLayer.Forward(x)
	fOutCopy := make([]float64, len(fOut))
	copy(fOutCopy, fOut)
	b.forwardOut = append(b.forwardOut, fOutCopy)

	outSize := b.ForwardLayer.OutSize() + b.BackwardLayer.OutSize()
	if len(b.outputBuf) != outSize {
		b.outputBuf = make([]float64, outSize)
	}

	copy(b.outputBuf[:b.ForwardLayer.OutSize()], fOut)
	// Zero out backward part for now (not available step-by-step)
	for i := b.ForwardLayer.OutSize(); i < outSize; i++ {
		b.outputBuf[i] = 0
	}

	return b.outputBuf
}

// ProcessSequence processes an entire sequence and returns all concatenated outputs.
func (b *Bidirectional) ProcessSequence(seq [][]float64) [][]float64 {
	b.Reset()
	n := len(seq)
	b.forwardOut = make([][]float64, n)
	b.backwardOut = make([][]float64, n)
	results := make([][]float64, n)

	// Forward pass
	for i := 0; i < n; i++ {
		b.forwardOut[i] = b.ForwardLayer.Forward(seq[i])
	}

	// Backward pass (the layer's forward pass but in reverse sequence)
	b.BackwardLayer.Reset()
	for i := n - 1; i >= 0; i-- {
		b.backwardOut[i] = b.BackwardLayer.Forward(seq[i])
	}

	// Concatenate
	fSize := b.ForwardLayer.OutSize()
	bSize := b.BackwardLayer.OutSize()
	for i := 0; i < n; i++ {
		res := make([]float64, fSize+bSize)
		copy(res[:fSize], b.forwardOut[i])
		copy(res[fSize:], b.backwardOut[i])
		results[i] = res
	}

	// Store sequence for backward
	b.inputSequence = seq

	return results
}

// Backward performs backpropagation.
// For bidirectional layers, this should be called after processing a whole sequence.
func (b *Bidirectional) Backward(grad []float64) []float64 {
	// In the GoNeuron Network implementation, Backward is called step-by-step.
	// This makes Bidirectional hard to use in a standard Network.
	// For now, we return the ForwardLayer gradient.
	return b.ForwardLayer.Backward(grad[:b.ForwardLayer.OutSize()])
}

// Reset resets both layers.
func (b *Bidirectional) Reset() {
	b.ForwardLayer.Reset()
	b.BackwardLayer.Reset()
	b.inputSequence = b.inputSequence[:0]
	b.forwardOut = b.forwardOut[:0]
	b.backwardOut = b.backwardOut[:0]
}

// Params returns all parameters.
func (b *Bidirectional) Params() []float64 {
	fParams := b.ForwardLayer.Params()
	bParams := b.BackwardLayer.Params()
	params := make([]float64, len(fParams)+len(bParams))
	copy(params, fParams)
	copy(params[len(fParams):], bParams)
	return params
}

// SetParams sets parameters.
func (b *Bidirectional) SetParams(params []float64) {
	fLen := len(b.ForwardLayer.Params())
	b.ForwardLayer.SetParams(params[:fLen])
	b.BackwardLayer.SetParams(params[fLen:])
}

// Gradients returns all gradients.
func (b *Bidirectional) Gradients() []float64 {
	fGrads := b.ForwardLayer.Gradients()
	bGrads := b.BackwardLayer.Gradients()
	grads := make([]float64, len(fGrads)+len(bGrads))
	copy(grads, fGrads)
	copy(grads[len(fGrads):], bGrads)
	return grads
}

// SetGradients sets gradients.
func (b *Bidirectional) SetGradients(grads []float64) {
	fLen := len(b.ForwardLayer.Gradients())
	b.ForwardLayer.SetGradients(grads[:fLen])
	b.BackwardLayer.SetGradients(grads[fLen:])
}

// ClearGradients clears gradients.
func (b *Bidirectional) ClearGradients() {
	b.ForwardLayer.ClearGradients()
	b.BackwardLayer.ClearGradients()
}

// OutSize returns concatenated output size.
func (b *Bidirectional) OutSize() int {
	return b.ForwardLayer.OutSize() + b.BackwardLayer.OutSize()
}

// InSize returns input size.
func (b *Bidirectional) InSize() int {
	return b.ForwardLayer.InSize()
}

// Clone creates a deep copy.
func (b *Bidirectional) Clone() Layer {
	return NewBidirectional(b.ForwardLayer.Clone(), b.BackwardLayer.Clone())
}

// AccumulateBackward (stub)
func (b *Bidirectional) AccumulateBackward(grad []float64) []float64 {
	return b.Backward(grad)
}
