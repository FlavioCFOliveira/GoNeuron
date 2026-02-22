// Package layer provides neural network layer implementations.
package layer

import (
	"sync"
)

// Bidirectional is a layer wrapper that processes sequences in both forward and backward directions.
// It uses two separate layers (typically LSTMs or GRUs) and concatenates their outputs.
type Bidirectional struct {
	forward  Layer
	backward Layer

	// Buffers
	outputBuf []float32

	// For sequences
	inputHistory    [][]float32
	forwardHistory  [][]float32
	backwardHistory [][]float32
	timeStep        int
}

// NewBidirectional creates a new bidirectional wrapper for the given layer.
// It clones the provided layer to create the backward direction layer.
func NewBidirectional(l Layer) *Bidirectional {
	forward := l
	backward := l.Clone()

	outSize := l.OutSize()

	return &Bidirectional{
		forward:         forward,
		backward:        backward,
		outputBuf:       make([]float32, outSize*2),
		inputHistory:    make([][]float32, 0),
		forwardHistory:  make([][]float32, 0),
		backwardHistory: make([][]float32, 0),
		timeStep:        0,
	}
}

// Forward performs a forward pass. Note: In bidirectional layers, the full sequence
// is accumulated to allow the backward layer to process it later.
func (b *Bidirectional) Forward(x []float32) []float32 {
	// Store input for backward pass
	input := make([]float32, len(x))
	copy(input, x)
	b.inputHistory = append(b.inputHistory, input)

	// Forward pass for forward layer
	fOut := b.forward.Forward(x)
	fOutCopy := make([]float32, len(fOut))
	copy(fOutCopy, fOut)
	b.forwardHistory = append(b.forwardHistory, fOutCopy)

	// Combine (Forward part + Zeros for backward part for now)
	// The actual backward layer pass will happen during a specialized "SequenceForward"
	// or during the first Backward call when we have the full history.
	outSize := b.forward.OutSize()
	copy(b.outputBuf[:outSize], fOut)
	// Clear backward part (will be filled when full sequence is known or during Backward)
	for i := outSize; i < outSize*2; i++ {
		b.outputBuf[i] = 0
	}

	b.timeStep++
	return b.outputBuf
}

// Backward performs backpropagation.
// When Backward is called, we assume the sequence has ended, and we first
// run the backward layer's forward pass over the accumulated history if not already done.
func (b *Bidirectional) Backward(grad []float32) []float32 {
	outSize := b.forward.OutSize()
	inSize := b.forward.InSize()

	// If this is the first backward call for this sequence, compute backward layer forward pass
	if b.timeStep > 0 && len(b.backwardHistory) == 0 {
		b.ComputeBackwardHiddenStates()
	}

	ts := b.timeStep - 1
	if ts < 0 {
		return make([]float32, inSize)
	}

	// Split gradient
	fGrad := grad[:outSize]
	bGrad := grad[outSize:]

	// Backward for forward layer
	fInGrad := b.forward.Backward(fGrad)

	// Backward for backward layer (it was fed inputs in reverse order)
	bInGrad := b.backward.Backward(bGrad)

	// Combine input gradients
	combinedInGrad := make([]float32, inSize)
	for i := 0; i < inSize; i++ {
		combinedInGrad[i] = fInGrad[i] + bInGrad[i]
	}

	b.timeStep--
	if b.timeStep == 0 {
		// Sequence finished, clear history for next one
		b.inputHistory = b.inputHistory[:0]
		b.backwardHistory = b.backwardHistory[:0]
	}

	return combinedInGrad
}

// Params returns concatenated parameters.
func (b *Bidirectional) Params() []float32 {
	fParams := b.forward.Params()
	bParams := b.backward.Params()
	params := make([]float32, len(fParams)+len(bParams))
	copy(params, fParams)
	copy(params[len(fParams):], bParams)
	return params
}

// SetParams sets parameters for both layers.
func (b *Bidirectional) SetParams(params []float32) {
	fLen := len(b.forward.Params())
	b.forward.SetParams(params[:fLen])
	b.backward.SetParams(params[fLen:])
}

// Gradients returns concatenated gradients.
func (b *Bidirectional) Gradients() []float32 {
	fGrads := b.forward.Gradients()
	bGrads := b.backward.Gradients()
	grads := make([]float32, len(fGrads)+len(bGrads))
	copy(grads, fGrads)
	copy(grads[len(fGrads):], bGrads)
	return grads
}

// SetGradients sets gradients for both layers.
func (b *Bidirectional) SetGradients(grads []float32) {
	fLen := len(b.forward.Gradients())
	b.forward.SetGradients(grads[:fLen])
	b.backward.SetGradients(grads[fLen:])
}

// SetDevice sets the computation device for both layers.
func (b *Bidirectional) SetDevice(device Device) {
	b.forward.SetDevice(device)
	b.backward.SetDevice(device)
}

// Reset resets both layers and histories.
func (b *Bidirectional) Reset() {
	b.forward.Reset()
	b.backward.Reset()
	b.inputHistory = b.inputHistory[:0]
	b.forwardHistory = b.forwardHistory[:0]
	b.backwardHistory = b.backwardHistory[:0]
	b.timeStep = 0
}

// ComputeBackwardHiddenStates processes the accumulated history in reverse.
func (b *Bidirectional) ComputeBackwardHiddenStates() {
	b.backward.Reset()
	seqLen := len(b.inputHistory)
	if cap(b.backwardHistory) < seqLen {
		b.backwardHistory = make([][]float32, seqLen)
	} else {
		b.backwardHistory = b.backwardHistory[:seqLen]
	}

	// Process input history in reverse
	for i := seqLen - 1; i >= 0; i-- {
		out := b.backward.Forward(b.inputHistory[i])
		outCopy := make([]float32, len(out))
		copy(outCopy, out)
		b.backwardHistory[i] = outCopy
	}
}

// GetForwardOutputAt returns the forward layer output at the given time step.
func (b *Bidirectional) GetForwardOutputAt(t int) []float32 {
	if t < 0 || t >= len(b.forwardHistory) {
		return nil
	}
	return b.forwardHistory[t]
}

// GetBackwardOutputAt returns the backward layer output at the given time step.
func (b *Bidirectional) GetBackwardOutputAt(t int) []float32 {
	if t < 0 || t >= len(b.backwardHistory) {
		return nil
	}
	return b.backwardHistory[t]
}

// ClearGradients clears gradients for both layers.
func (b *Bidirectional) ClearGradients() {
	b.forward.ClearGradients()
	b.backward.ClearGradients()
}

// Clone creates a deep copy.
func (b *Bidirectional) Clone() Layer {
	return NewBidirectional(b.forward.Clone())
}

// InSize returns input size.
func (b *Bidirectional) InSize() int {
	return b.forward.InSize()
}

// OutSize returns output size (2x hidden size).
func (b *Bidirectional) OutSize() int {
	return b.forward.OutSize() * 2
}

// AccumulateBackward accumulates gradients.
func (b *Bidirectional) AccumulateBackward(grad []float32) []float32 {
	return b.Backward(grad)
}

// ProcessSequence processes an entire sequence and returns all concatenated outputs.
// This is more efficient for bidirectional layers.
func (b *Bidirectional) ProcessSequence(seq [][]float32) [][]float32 {
	b.Reset()
	seqLen := len(seq)

	fOuts := make([][]float32, seqLen)
	bOuts := make([][]float32, seqLen)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := 0; i < seqLen; i++ {
			out := b.forward.Forward(seq[i])
			outCopy := make([]float32, len(out))
			copy(outCopy, out)
			fOuts[i] = outCopy
		}
	}()

	go func() {
		defer wg.Done()
		for i := seqLen - 1; i >= 0; i-- {
			out := b.backward.Forward(seq[i])
			outCopy := make([]float32, len(out))
			copy(outCopy, out)
			bOuts[i] = outCopy
		}
	}()

	wg.Wait()

	results := make([][]float32, seqLen)
	fSize := b.forward.OutSize()
	bSize := b.backward.OutSize()
	for i := 0; i < seqLen; i++ {
		res := make([]float32, fSize+bSize)
		copy(res[:fSize], fOuts[i])
		copy(res[fSize:], bOuts[i])
		results[i] = res
	}

	return results
}
