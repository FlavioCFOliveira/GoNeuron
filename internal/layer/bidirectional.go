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
	outputBuf []float64

	// For sequences
	inputHistory    [][]float64
	backwardHistory [][]float64
	timeStep        int
}

// NewBidirectional creates a new bidirectional wrapper for the given layer.
// It clones the provided layer to create the backward direction layer.
func NewBidirectional(l Layer) *Bidirectional {
	forward := l
	backward := l.Clone()

	outSize := l.OutSize()

	return &Bidirectional{
		forward:        forward,
		backward:       backward,
		outputBuf:      make([]float64, outSize*2),
		inputHistory:    make([][]float64, 0),
		backwardHistory: make([][]float64, 0),
		timeStep:        0,
	}
}

// Forward performs a forward pass. Note: In bidirectional layers, the full sequence
// is accumulated to allow the backward layer to process it later.
func (b *Bidirectional) Forward(x []float64) []float64 {
	// Store input for backward pass
	input := make([]float64, len(x))
	copy(input, x)
	b.inputHistory = append(b.inputHistory, input)

	// Forward pass for forward layer
	fOut := b.forward.Forward(x)

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
func (b *Bidirectional) Backward(grad []float64) []float64 {
	outSize := b.forward.OutSize()
	inSize := b.forward.InSize()

	// If this is the first backward call for this sequence, compute backward layer forward pass
	if b.timeStep > 0 && len(b.backwardHistory) == 0 {
		b.computeBackwardLayerForward()
	}

	ts := b.timeStep - 1
	if ts < 0 {
		return make([]float64, inSize)
	}

	// Split gradient
	fGrad := grad[:outSize]
	bGrad := grad[outSize:]

	// Backward for forward layer
	fInGrad := b.forward.Backward(fGrad)

	// Backward for backward layer (it was fed inputs in reverse order)
	bInGrad := b.backward.Backward(bGrad)

	// Combine input gradients
	combinedInGrad := make([]float64, inSize)
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

func (b *Bidirectional) computeBackwardLayerForward() {
	b.backward.Reset()
	seqLen := len(b.inputHistory)
	b.backwardHistory = make([][]float64, seqLen)

	// Process input history in reverse
	for i := seqLen - 1; i >= 0; i-- {
		out := b.backward.Forward(b.inputHistory[i])
		outCopy := make([]float64, len(out))
		copy(outCopy, out)
		b.backwardHistory[i] = outCopy
	}
}

// Params returns concatenated parameters.
func (b *Bidirectional) Params() []float64 {
	fParams := b.forward.Params()
	bParams := b.backward.Params()
	params := make([]float64, len(fParams)+len(bParams))
	copy(params, fParams)
	copy(params[len(fParams):], bParams)
	return params
}

// SetParams sets parameters for both layers.
func (b *Bidirectional) SetParams(params []float64) {
	fLen := len(b.forward.Params())
	b.forward.SetParams(params[:fLen])
	b.backward.SetParams(params[fLen:])
}

// Gradients returns concatenated gradients.
func (b *Bidirectional) Gradients() []float64 {
	fGrads := b.forward.Gradients()
	bGrads := b.backward.Gradients()
	grads := make([]float64, len(fGrads)+len(bGrads))
	copy(grads, fGrads)
	copy(grads[len(fGrads):], bGrads)
	return grads
}

// SetGradients sets gradients for both layers.
func (b *Bidirectional) SetGradients(grads []float64) {
	fLen := len(b.forward.Gradients())
	b.forward.SetGradients(grads[:fLen])
	b.backward.SetGradients(grads[fLen:])
}

// Reset resets both layers and histories.
func (b *Bidirectional) Reset() {
	b.forward.Reset()
	b.backward.Reset()
	b.inputHistory = b.inputHistory[:0]
	b.backwardHistory = b.backwardHistory[:0]
	b.timeStep = 0
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
func (b *Bidirectional) AccumulateBackward(grad []float64) []float64 {
	return b.Backward(grad)
}

// ProcessSequence processes an entire sequence and returns all concatenated outputs.
// This is more efficient for bidirectional layers.
func (b *Bidirectional) ProcessSequence(seq [][]float64) [][]float64 {
	b.Reset()
	seqLen := len(seq)

	fOuts := make([][]float64, seqLen)
	bOuts := make([][]float64, seqLen)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := 0; i < seqLen; i++ {
			out := b.forward.Forward(seq[i])
			outCopy := make([]float64, len(out))
			copy(outCopy, out)
			fOuts[i] = outCopy
		}
	}()

	go func() {
		defer wg.Done()
		for i := seqLen - 1; i >= 0; i-- {
			out := b.backward.Forward(seq[i])
			outCopy := make([]float64, len(out))
			copy(outCopy, out)
			bOuts[i] = outCopy
		}
	}()

	wg.Wait()

	results := make([][]float64, seqLen)
	fSize := b.forward.OutSize()
	bSize := b.backward.OutSize()
	for i := 0; i < seqLen; i++ {
		res := make([]float64, fSize+bSize)
		copy(res[:fSize], fOuts[i])
		copy(res[fSize:], bOuts[i])
		results[i] = res
	}

	return results
}
