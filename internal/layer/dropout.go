// Package layer provides neural network layer implementations.
package layer

// Dropout implements dropout regularization.
// During training, randomly sets inputs to 0 with probability p.
// During inference, passes inputs through unchanged.
type Dropout struct {
	// Probability of dropping a neuron
	p float32

	// Training mode
	training bool

	// Input size
	inSize int

	// Output size (same as inSize for standard dropout)
	outSize int

	// Reusable buffers
	inputBuf   []float32
	outputBuf  []float32
	maskBuf    []float32 // Stores dropout mask for backward pass
	gradInBuf  []float32 // Gradient w.r.t. input
	gradParams []float32 // No learnable parameters, just for interface

	// Saved state for backward pass
	savedMaskOffsets []int
	arenaPtr         *[]float32

	// RNG for dropout masks
	rng *RNG

	device Device
}

// NewDropout creates a new dropout layer.
// p is the probability of dropping a neuron (default 0.5).
// A higher p means more aggressive dropout.
func NewDropout(p float32, inSize int) *Dropout {
	return &Dropout{
		p:                p,
		training:         true,
		inSize:           inSize,
		outSize:          inSize,
		inputBuf:         make([]float32, inSize),
		outputBuf:        make([]float32, inSize),
		maskBuf:          make([]float32, inSize),
		gradInBuf:        make([]float32, inSize),
		gradParams:       make([]float32, 0),
		savedMaskOffsets: make([]int, 0, 16),
		rng:              NewRNG(42),
		device:           &CPUDevice{},
	}
}

// SetDevice sets the computation device.
func (d *Dropout) SetDevice(device Device) {
	d.device = device
}

// SetTraining sets whether the layer should be in training or inference mode.
func (d *Dropout) SetTraining(training bool) {
	d.training = training
}

// IsTraining returns whether the layer is in training mode.
func (d *Dropout) IsTraining() bool {
	return d.training
}

func (d *Dropout) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	if !d.training {
		if len(d.outputBuf) < len(x) {
			d.outputBuf = make([]float32, len(x))
		}
		copy(d.outputBuf, x)
		return d.outputBuf[:len(x)]
	}

	inSize := len(x)
	if len(d.outputBuf) < inSize {
		d.outputBuf = make([]float32, inSize)
	}

	var mask []float32
	if arena != nil && offset != nil {
		d.arenaPtr = arena
		if len(*arena) < *offset+inSize {
			newArena := make([]float32, (*offset+inSize)*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		mask = (*arena)[*offset : *offset+inSize]
		d.savedMaskOffsets = append(d.savedMaskOffsets, *offset)
		*offset += inSize
	} else {
		if cap(d.maskBuf) < inSize {
			d.maskBuf = make([]float32, inSize)
		}
		mask = d.maskBuf[:inSize]
		d.savedMaskOffsets = append(d.savedMaskOffsets[:0], 0)
		d.arenaPtr = &d.maskBuf
	}

	// GPU acceleration
	if m, ok := d.device.(*MetalDevice); ok && m.IsAvailable() {
		inBuf := m.CreateBuffer(x)
		outBuf := m.CreateBuffer(d.outputBuf[:inSize])
		mBuf := m.CreateBuffer(mask)
		defer inBuf.Free()
		defer outBuf.Free()
		defer mBuf.Free()

		m.DropoutPersistent(inBuf, outBuf, mBuf, inSize, d.p, true, d.rng.RandUint64())
		outBuf.Read(d.outputBuf[:inSize])
		mBuf.Read(mask)
		return d.outputBuf[:inSize]
	}

	keepProb := 1.0 - d.p
	scale := 1.0 / keepProb

	for i := 0; i < inSize; i++ {
		if d.rng.RandFloat() < d.p {
			mask[i] = 0
			d.outputBuf[i] = 0
		} else {
			mask[i] = 1
			d.outputBuf[i] = x[i] * scale
		}
	}

	return d.outputBuf[:inSize]
}

// Forward performs a forward pass through the dropout layer.
func (d *Dropout) Forward(x []float32) []float32 {
	return d.ForwardWithArena(x, nil, nil)
}

func (d *Dropout) ForwardBatch(x []float32, batchSize int) []float32 {
	return d.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (d *Dropout) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return d.ForwardWithArena(x, arena, offset)
	}

	inSize := len(x) / batchSize
	requiredSize := batchSize * inSize
	if len(d.outputBuf) < requiredSize {
		d.outputBuf = make([]float32, requiredSize)
	}

	if !d.training {
		copy(d.outputBuf[:requiredSize], x)
		return d.outputBuf[:requiredSize]
	}

	var mask []float32
	if arena != nil && offset != nil {
		d.arenaPtr = arena
		if len(*arena) < *offset+requiredSize {
			newArena := make([]float32, (*offset+requiredSize)*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		mask = (*arena)[*offset : *offset+requiredSize]
		d.savedMaskOffsets = d.savedMaskOffsets[:0]
		for i := 0; i < batchSize; i++ {
			d.savedMaskOffsets = append(d.savedMaskOffsets, *offset+i*inSize)
		}
		*offset += requiredSize
	} else {
		if cap(d.maskBuf) < requiredSize {
			d.maskBuf = make([]float32, requiredSize)
		}
		mask = d.maskBuf[:requiredSize]
		d.arenaPtr = &d.maskBuf
		d.savedMaskOffsets = d.savedMaskOffsets[:0]
		for i := 0; i < batchSize; i++ {
			d.savedMaskOffsets = append(d.savedMaskOffsets, i*inSize)
		}
	}

	// GPU acceleration
	if m, ok := d.device.(*MetalDevice); ok && m.IsAvailable() {
		inBuf := m.CreateBuffer(x)
		outBuf := m.CreateBuffer(d.outputBuf[:requiredSize])
		mBuf := m.CreateBuffer(mask)
		defer inBuf.Free()
		defer outBuf.Free()
		defer mBuf.Free()

		m.DropoutPersistent(inBuf, outBuf, mBuf, requiredSize, d.p, true, d.rng.RandUint64())
		outBuf.Read(d.outputBuf[:requiredSize])
		mBuf.Read(mask)
		return d.outputBuf[:requiredSize]
	}

	keepProb := 1.0 - d.p
	scale := 1.0 / keepProb

	for i := 0; i < requiredSize; i++ {
		if d.rng.RandFloat() < d.p {
			mask[i] = 0
			d.outputBuf[i] = 0
		} else {
			mask[i] = 1
			d.outputBuf[i] = x[i] * scale
		}
	}

	return d.outputBuf[:requiredSize]
}

// Backward performs backpropagation through the dropout layer.
func (d *Dropout) Backward(grad []float32) []float32 {
	if !d.training {
		if len(d.gradInBuf) < len(grad) {
			d.gradInBuf = make([]float32, len(grad))
		}
		copy(d.gradInBuf, grad)
		return d.gradInBuf[:len(grad)]
	}

	numSaved := len(d.savedMaskOffsets)
	if numSaved == 0 {
		return nil
	}

	ts := numSaved - 1
	offset := d.savedMaskOffsets[ts]
	inSize := len(grad)
	mask := (*d.arenaPtr)[offset : offset+inSize]

	if len(d.gradInBuf) < inSize {
		d.gradInBuf = make([]float32, inSize)
	}

	// GPU acceleration
	if m, ok := d.device.(*MetalDevice); ok && m.IsAvailable() {
		gOutBuf := m.CreateBuffer(grad)
		gInBuf := m.CreateBuffer(d.gradInBuf[:inSize])
		mBuf := m.CreateBuffer(mask)
		defer gOutBuf.Free()
		defer gInBuf.Free()
		defer mBuf.Free()

		m.DropoutBackwardPersistent(gOutBuf, gInBuf, mBuf, inSize, d.p)
		gInBuf.Read(d.gradInBuf[:inSize])

		d.savedMaskOffsets = d.savedMaskOffsets[:ts]
		return d.gradInBuf[:inSize]
	}

	keepProb := 1.0 - d.p
	scale := 1.0 / keepProb

	for i := 0; i < inSize; i++ {
		if mask[i] > 0 {
			d.gradInBuf[i] = grad[i] * scale
		} else {
			d.gradInBuf[i] = 0
		}
	}

	d.savedMaskOffsets = d.savedMaskOffsets[:ts]
	return d.gradInBuf[:inSize]
}

func (d *Dropout) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return d.Backward(grad)
	}

	inSize := len(grad) / batchSize
	requiredSize := batchSize * inSize
	if len(d.gradInBuf) < requiredSize {
		d.gradInBuf = make([]float32, requiredSize)
	}

	if !d.training {
		copy(d.gradInBuf[:requiredSize], grad)
		return d.gradInBuf[:requiredSize]
	}

	numSaved := len(d.savedMaskOffsets)
	if numSaved < batchSize {
		// Fallback to sequential if not enough masks
		for i := batchSize - 1; i >= 0; i-- {
			dx := d.Backward(grad[i*inSize : (i+1)*inSize])
			copy(d.gradInBuf[i*inSize:(i+1)*inSize], dx)
		}
		return d.gradInBuf[:requiredSize]
	}

	ts := numSaved - batchSize
	offset := d.savedMaskOffsets[ts]
	mask := (*d.arenaPtr)[offset : offset+requiredSize]

	// GPU acceleration
	if m, ok := d.device.(*MetalDevice); ok && m.IsAvailable() {
		gOutBuf := m.CreateBuffer(grad)
		gInBuf := m.CreateBuffer(d.gradInBuf[:requiredSize])
		mBuf := m.CreateBuffer(mask)
		defer gOutBuf.Free()
		defer gInBuf.Free()
		defer mBuf.Free()

		m.DropoutBackwardPersistent(gOutBuf, gInBuf, mBuf, requiredSize, d.p)
		gInBuf.Read(d.gradInBuf[:requiredSize])

		d.savedMaskOffsets = d.savedMaskOffsets[:ts]
		return d.gradInBuf[:requiredSize]
	}

	keepProb := 1.0 - d.p
	scale := 1.0 / keepProb

	for i := 0; i < requiredSize; i++ {
		if mask[i] > 0 {
			d.gradInBuf[i] = grad[i] * scale
		} else {
			d.gradInBuf[i] = 0
		}
	}

	d.savedMaskOffsets = d.savedMaskOffsets[:ts]
	return d.gradInBuf[:requiredSize]
}

func (d *Dropout) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	return d.BackwardBatch(grad, batchSize)
}

func (d *Dropout) AccumulateBackward(grad []float32) []float32 {
	return d.Backward(grad)
}

// Params returns layer parameters (empty for Dropout).
func (d *Dropout) Params() []float32 {
	return d.gradParams
}

// SetParams sets layer parameters (no-op for Dropout).
func (d *Dropout) SetParams(params []float32) {
}

// Gradients returns layer gradients (empty for Dropout).
func (d *Dropout) Gradients() []float32 {
	return d.gradParams
}

// SetGradients sets layer gradients (no-op for Dropout).
func (d *Dropout) SetGradients(gradients []float32) {
}

// InSize returns the input size of the layer.
func (d *Dropout) InSize() int {
	return d.inSize
}

// OutSize returns the output size of the layer.
func (d *Dropout) OutSize() int {
	return d.outSize
}

// Reset resets the dropout layer.
func (d *Dropout) Reset() {
	d.rng = NewRNG(42)
}

// ClearGradients zeroes out the accumulated gradients (no-op for Dropout).
func (d *Dropout) ClearGradients() {
}

// Clone creates a deep copy of the dropout layer.
func (d *Dropout) Clone() Layer {
	newD := NewDropout(d.p, d.inSize)
	newD.training = d.training
	newD.device = d.device
	return newD
}

func (d *Dropout) LightweightClone(params []float32, grads []float32) Layer {
	newD := &Dropout{
		p:                d.p,
		training:         d.training,
		inSize:           d.inSize,
		outSize:          d.outSize,
		inputBuf:         make([]float32, d.inSize),
		outputBuf:        make([]float32, d.inSize),
		maskBuf:          make([]float32, d.inSize),
		gradInBuf:        make([]float32, d.inSize),
		gradParams:       make([]float32, 0),
		savedMaskOffsets: make([]int, 0, 128),
		rng:              NewRNG(42),
		device:           d.device,
	}
	return newD
}

// GetP returns the dropout probability.
func (d *Dropout) GetP() float32 {
	return d.p
}

// SetP sets the dropout probability.
func (d *Dropout) SetP(p float32) {
	d.p = p
}
