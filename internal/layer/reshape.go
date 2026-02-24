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
	outputBuf []float32

	training bool

	device Device
}

// NewFlatten creates a new flatten layer.
// startDim is the first dimension to flatten (default 1, skips batch dimension).
// endDim is the last dimension to flatten (default -1, all remaining).
func NewFlatten() *Flatten {
	return &Flatten{
		startDim:   1,
		endDim:     -1,
		outputBuf:  make([]float32, 0),
		inputShape: make([]int, 0),
		outSize:    0,
		device:     &CPUDevice{},
	}
}

// SetDevice sets the computation device.
func (f *Flatten) SetDevice(device Device) {
	f.device = device
}

// Build initializes the layer with the given input size.
func (f *Flatten) Build(inSize int) {
	f.outSize = inSize
}

// SetDimensions sets the start and end dimensions for flattening.
// startDim: first dimension to flatten (default 1, skips batch dimension)
// endDim: last dimension to flatten (default -1, all remaining)
// A negative endDim counts from the end (-1 = all remaining)
func (f *Flatten) SetDimensions(startDim, endDim int) {
	f.startDim = startDim
	f.endDim = endDim
}

// SetTraining sets whether the layer is in training mode.
func (f *Flatten) SetTraining(training bool) {
	f.training = training
}

func (f *Flatten) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	total := len(x)
	if cap(f.inputShape) < 2 {
		f.inputShape = make([]int, 0, 2)
	}
	f.inputShape = append(f.inputShape[:0], 1, total)

	flattened := total
	if f.outSize == 0 || len(f.outputBuf) < flattened {
		f.outputBuf = make([]float32, flattened)
		f.outSize = flattened
	}

	copy(f.outputBuf, x)
	return f.outputBuf[:flattened]
}

// Forward performs a forward pass, flattening the input.
func (f *Flatten) Forward(x []float32) []float32 {
	return f.ForwardWithArena(x, nil, nil)
}

// Backward performs backpropagation, reshaping gradient to original shape.
func (f *Flatten) Backward(grad []float32) []float32 {
	// Gradient flows through unchanged - just need to return correct size
	return grad
}

// AccumulateBackward performs backpropagation and accumulates gradients.
func (f *Flatten) AccumulateBackward(grad []float32) []float32 {
	// Gradient flows through unchanged - just need to return correct size
	return grad
}

var emptyParams = make([]float32, 0)

func (f *Flatten) ForwardBatch(x []float32, batchSize int) []float32 {
	return f.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (f *Flatten) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	return f.ForwardWithArena(x, arena, offset)
}

func (f *Flatten) BackwardBatch(grad []float32, batchSize int) []float32 {
	return grad
}

func (f *Flatten) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return grad
}

// Params returns layer parameters (empty for Flatten).
func (f *Flatten) Params() []float32 {
	return emptyParams
}

// SetParams sets layer parameters (no-op for Flatten).
func (f *Flatten) SetParams(params []float32) {
	// No parameters to set
}

// Gradients returns layer gradients (empty for Flatten).
func (f *Flatten) Gradients() []float32 {
	return emptyParams
}

// SetGradients sets layer gradients (no-op for Flatten).
func (f *Flatten) SetGradients(gradients []float32) {
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
	f.inputShape = f.inputShape[:0]
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
	newF.device = f.device
	return newF
}

func (f *Flatten) LightweightClone(params []float32, grads []float32) Layer {
	newF := &Flatten{
		startDim:   f.startDim,
		endDim:     f.endDim,
		outSize:    f.outSize,
		outputBuf:  make([]float32, 0),
		inputShape: make([]int, 0),
		training:   f.training,
		device:     f.device,
	}
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
func (f *Flatten) SetOutputBuf(buf []float32) {
	f.outputBuf = buf
}
