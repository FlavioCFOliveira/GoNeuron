package layer

import (
	"math"
)

// Attention implements a simple dot-product attention mechanism.
// It takes a sequence of hidden states and produces a weighted context vector.
type Attention struct {
	inSize  int
	outSize int

	// Parameters (Query, Key, Value weights)
	// For simplicity, we implement a self-attention like mechanism or
	// attention over a sequence with a learnable query.
	queryWeights []float64 // [outSize * inSize]
	keyWeights   []float64 // [outSize * inSize]
	valueWeights []float64 // [outSize * inSize]

	// Buffers
	inputBuf     []float64
	outputBuf    []float64
	gradQueryW   []float64
	gradKeyW     []float64
	gradValueW   []float64
	gradInBuf    []float64

	// Sequence state
	storedInputs [][]float64
}

// NewAttention creates a new Attention layer.
func NewAttention(inSize, outSize int) *Attention {
	rng := NewRNG(42)
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
		outputBuf:    make([]float64, outSize),
		gradQueryW:   make([]float64, outSize*inSize),
		gradKeyW:     make([]float64, outSize*inSize),
		gradValueW:   make([]float64, outSize*inSize),
		gradInBuf:    make([]float64, inSize),
		storedInputs: make([][]float64, 0),
	}
}

// Forward performs a forward pass.
// For now, it just stores the input to build a sequence context.
// Real attention usually operates on the whole sequence at once.
func (a *Attention) Forward(x []float64) []float64 {
	xCopy := make([]float64, len(x))
	copy(xCopy, x)
	a.storedInputs = append(a.storedInputs, xCopy)

	// Return a placeholder for step-by-step
	if len(a.outputBuf) != a.outSize {
		a.outputBuf = make([]float64, a.outSize)
	}
	return a.outputBuf
}

// Backward performs a backward pass.
func (a *Attention) Backward(grad []float64) []float64 {
	// Stub
	return a.gradInBuf
}

// Reset resets the attention state.
func (a *Attention) Reset() {
	a.storedInputs = a.storedInputs[:0]
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

// AccumulateBackward (stub)
func (a *Attention) AccumulateBackward(grad []float64) []float64 {
	return a.Backward(grad)
}
