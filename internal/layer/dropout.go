// Package layer provides neural network layer implementations.
package layer

import (
)

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

	// RNG for dropout masks
	rng *RNG

	device Device
}

// NewDropout creates a new dropout layer.
// p is the probability of dropping a neuron (default 0.5).
// A higher p means more aggressive dropout.
func NewDropout(p float32, inSize int) *Dropout {
	return &Dropout{
		p:         p,
		training:  true,
		inSize:    inSize,
		outSize:   inSize,
		inputBuf:  make([]float32, inSize),
		outputBuf: make([]float32, inSize),
		maskBuf:   make([]float32, inSize),
		gradInBuf: make([]float32, inSize),
		gradParams: make([]float32, 0), // No parameters
		rng:       NewRNG(42),
		device:    &CPUDevice{},
	}
}

// SetDevice sets the computation device.
func (d *Dropout) SetDevice(device Device) {
	d.device = device
}

// SetTraining sets whether the layer should be in training or inference mode.
// In training mode, dropout is applied.
// In inference mode, inputs pass through unchanged.
func (d *Dropout) SetTraining(training bool) {
	d.training = training
}

// IsTraining returns whether the layer is in training mode.
func (d *Dropout) IsTraining() bool {
	return d.training
}

func (d *Dropout) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	copy(d.inputBuf, x)

	if d.training {
		// Generate dropout mask
		keepProb := 1.0 - d.p
		scale := 1.0 / keepProb

		var mask []float32
		if arena != nil && offset != nil {
			if len(*arena) < *offset+d.inSize {
				newArena := make([]float32, (*offset+d.inSize)*2)
				copy(newArena, *arena)
				*arena = newArena
			}
			mask = (*arena)[*offset : *offset+d.inSize]
			d.maskBuf = mask // Update reference for Backward
			*offset += d.inSize
		} else {
			mask = d.maskBuf
		}

		for i := 0; i < d.inSize; i++ {
			r := d.rng.RandFloat()
			if r < d.p {
				mask[i] = 0
				d.outputBuf[i] = 0
			} else {
				mask[i] = 1
				d.outputBuf[i] = d.inputBuf[i] * scale
			}
		}
	} else {
		// Inference mode: pass through unchanged
		copy(d.outputBuf, x)
	}

	return d.outputBuf
}

// Forward performs a forward pass through the dropout layer.
func (d *Dropout) Forward(x []float32) []float32 {
	return d.ForwardWithArena(x, nil, nil)
}


// Backward performs backpropagation through the dropout layer.
func (d *Dropout) Backward(grad []float32) []float32 {
	if d.training {
		// Gradient flows through where mask was 1, scaled by 1/keepProb
		keepProb := 1.0 - d.p
		scale := 1.0 / keepProb

		for i := 0; i < d.inSize; i++ {
			if d.maskBuf[i] > 0 {
				d.gradInBuf[i] = grad[i] * scale
			} else {
				d.gradInBuf[i] = 0
			}
		}
	} else {
		// No dropout in inference, gradient passes through unchanged
		copy(d.gradInBuf, grad)
	}

	return d.gradInBuf
}

// AccumulateBackward performs backpropagation and accumulates gradients.
// For Dropout, we just apply the mask and return the input gradient.
func (d *Dropout) AccumulateBackward(grad []float32) []float32 {
	return d.Backward(grad)
}

// Params returns layer parameters (empty for Dropout - no learnable params).
func (d *Dropout) Params() []float32 {
	return d.gradParams
}

// SetParams sets layer parameters (no-op for Dropout).
func (d *Dropout) SetParams(params []float32) {
	// No parameters to set
}

// Gradients returns layer gradients (empty for Dropout).
func (d *Dropout) Gradients() []float32 {
	return d.gradParams
}

// SetGradients sets layer gradients (no-op for Dropout).
func (d *Dropout) SetGradients(gradients []float32) {
	// No parameters to set
}

// InSize returns the input size of the layer.
func (d *Dropout) InSize() int {
	return d.inSize
}

// OutSize returns the output size of the layer.
func (d *Dropout) OutSize() int {
	return d.outSize
}

// Reset resets the dropout layer (clears RNG state for reproducibility).
func (d *Dropout) Reset() {
	d.rng = NewRNG(42)
}

// ClearGradients zeroes out the accumulated gradients (no-op for Dropout).
func (d *Dropout) ClearGradients() {
	// No parameters to clear
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
		p:          d.p,
		training:   d.training,
		inSize:     d.inSize,
		outSize:    d.outSize,
		inputBuf:   make([]float32, d.inSize),
		outputBuf:  make([]float32, d.inSize),
		maskBuf:    make([]float32, d.inSize),
		gradInBuf:  make([]float32, d.inSize),
		gradParams: make([]float32, 0),
		rng:        NewRNG(42),
		device:     d.device,
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
