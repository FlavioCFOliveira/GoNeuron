// Package layer provides neural network layer implementations.
package layer

// Flatten reshapes input to 2D (batch_size, flattened_features).
// This is useful for connecting convolutional layers to dense layers.
type Flatten struct {
	// Start and end dimensions for flattening
	startDim int
	endDim   int

	// Original input shape for backward pass
	inputShape []int

	// Output size (flattened)
	outSize int

	// Reusable buffers
	outputBuf []float64
}

// NewFlatten creates a new flatten layer.
// startDim is the first dimension to flatten (default 1, skips batch dimension).
// endDim is the last dimension to flatten (default -1, all remaining).
func NewFlatten() *Flatten {
	return &Flatten{
		startDim:   1,
		endDim:     -1,
		outputBuf:  make([]float64, 0),
		inputShape: make([]int, 0),
		outSize:    0,
	}
}

// SetDimensions sets the start and end dimensions for flattening.
// startDim: first dimension to flatten (default 1, skips batch dimension)
// endDim: last dimension to flatten (default -1, all remaining)
// A negative endDim counts from the end (-1 = all remaining)
func (f *Flatten) SetDimensions(startDim, endDim int) {
	f.startDim = startDim
	f.endDim = endDim
}

// Forward performs a forward pass, flattening the input.
// input: multi-dimensional array, typically [batch, channels, height, width]
// Returns: 2D array [batch, flattened_features]
func (f *Flatten) Forward(x []float64) []float64 {
	// Infer input shape from length
	total := len(x)

	// For inference, we assume batch dimension is first
	// and flatten everything else
	f.inputShape = []int{1, total}

	flattened := total
	if f.outSize == 0 || len(f.outputBuf) < flattened {
		f.outputBuf = make([]float64, flattened)
		f.outSize = flattened
	}

	copy(f.outputBuf, x)
	return f.outputBuf[:flattened]
}

// Backward performs backpropagation, reshaping gradient to original shape.
func (f *Flatten) Backward(grad []float64) []float64 {
	// Gradient flows through unchanged - just need to return correct size
	return grad
}

// AccumulateBackward performs backpropagation and accumulates gradients.
func (f *Flatten) AccumulateBackward(grad []float64) []float64 {
	// Gradient flows through unchanged - just need to return correct size
	return grad
}

// Params returns layer parameters (empty for Flatten).
func (f *Flatten) Params() []float64 {
	return make([]float64, 0)
}

// SetParams sets layer parameters (no-op for Flatten).
func (f *Flatten) SetParams(params []float64) {
	// No parameters to set
}

// Gradients returns layer gradients (empty for Flatten).
func (f *Flatten) Gradients() []float64 {
	return make([]float64, 0)
}

// SetGradients sets layer gradients (no-op for Flatten).
func (f *Flatten) SetGradients(gradients []float64) {
	// No parameters to set
}

// InSize returns the input size (flattened).
func (f *Flatten) InSize() int {
	// For Flatten, this returns the expected input size after flattening
	return f.outSize
}

// OutSize returns the output size (flattened).
func (f *Flatten) OutSize() int {
	return f.outSize
}

// Reset resets the flatten layer.
func (f *Flatten) Reset() {
	f.inputShape = make([]int, 0)
}

// ClearGradients zeroes out the accumulated gradients (no-op for Flatten).
func (f *Flatten) ClearGradients() {
	// No parameters to clear
}

// Clone creates a deep copy of the flatten layer.
func (f *Flatten) Clone() Layer {
	newF := NewFlatten()
	newF.startDim = f.startDim
	newF.endDim = f.endDim
	newF.outSize = f.outSize
	return newF
}

// OutputShape returns the output shape as a slice.
func (f *Flatten) OutputShape() []int {
	if f.outSize > 0 {
		return []int{1, f.outSize}
	}
	return []int{1, 0}
}

// SetOutputBuf allows setting a custom output buffer.
func (f *Flatten) SetOutputBuf(buf []float64) {
	f.outputBuf = buf
}
