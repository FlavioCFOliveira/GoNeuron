// Package layer provides neural network layer implementations.
package layer

import (
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"math"
)

// RNG is a simple, fast random number generator using LCG for deterministic behavior.
type RNG struct {
	seed uint64
}

// NewRNG creates a new RNG with the given seed.
func NewRNG(seed uint64) *RNG {
	return &RNG{seed: seed}
}

// RandFloat returns a random float32 in [0, 1).
func (r *RNG) RandFloat() float32 {
	r.seed = r.seed*6364136223846793005 + 1
	return float32(r.seed>>33) / float32(1<<31)
}

// Layer is a neural network layer.
type Layer interface {
	Forward(x []float32) []float32
	Backward(grad []float32) []float32
	AccumulateBackward(grad []float32) []float32
	Params() []float32
	SetParams([]float32)
	Gradients() []float32
	SetGradients([]float32)

	// Reset clears any internal state for the layer (e.g., LSTM hidden states).
	Reset()

	// ClearGradients zeroes out the accumulated gradients.
	ClearGradients()

	// Clone creates a deep copy of the layer.
	Clone() Layer

	// SetDevice sets the computation device.
	SetDevice(Device)

	// InSize returns the number of input features.
	InSize() int

	// OutSize returns the number of output features.
	OutSize() int
}

// NamedParam represents a named parameter with its shape and data.
type NamedParam struct {
	Name  string
	Shape []int
	Data  []float32
}

// NamedParamProvider is an optional interface for layers that can provide named parameters.
type NamedParamProvider interface {
	NamedParams() []NamedParam
}

// Dense is a fully connected layer optimized for performance.
type Dense struct {
	params    []float32 // Contiguous weights + biases
	weights   []float32 // View of params
	biases    []float32 // View of params
	act       activations.Activation
	outSize   int
	inSize    int
	device    Device

	// Reusable buffers
	inputBuf      []float32
	outputBuf     []float32
	preActBuf     []float32
	grads         []float32 // Contiguous gradW + gradB
	gradWBuf      []float32 // View of grads
	gradBBuf      []float32 // View of grads
	gradInBuf     []float32
	dzBuf         []float32

	savedInputs  [][]float32
	savedPreActs [][]float32
	savedOutput []float32

	// Metal persistent buffers
	bufWeights *MetalBuffer
	bufBiases  *MetalBuffer
	bufIn      *MetalBuffer
	bufOut     *MetalBuffer
}

// NewDense creates a new dense layer with pre-allocated buffers.
func NewDense(in, out int, act activations.Activation) *Dense {
	return NewDenseWithDevice(in, out, act, &CPUDevice{})
}

// NewDenseWithDevice creates a new dense layer with a specific device.
func NewDenseWithDevice(in, out int, act activations.Activation, device Device) *Dense {
	params := make([]float32, out*in+out)
	weights := params[:out*in]
	biases := params[out*in:]

	scale := float32(math.Sqrt(2.0 / (float64(in) + float64(out))))
	rng := NewRNG(uint64(in*1000 + out*100 + 42))
	for i := range weights {
		weights[i] = rng.RandFloat()*2*scale - scale
	}
	for i := range biases {
		biases[i] = rng.RandFloat()*0.2 - 0.1
	}

	grads := make([]float32, out*in+out)

	d := &Dense{
		params:       params,
		weights:      weights,
		biases:       biases,
		act:          act,
		outSize:      out,
		inSize:       in,
		device:       device,
		inputBuf:     make([]float32, in),
		outputBuf:    make([]float32, out),
		preActBuf:    make([]float32, out),
		grads:        grads,
		gradWBuf:     grads[:out*in],
		gradBBuf:     grads[out*in:],
		gradInBuf:    make([]float32, in),
		dzBuf:        make([]float32, out),
		savedInputs:  make([][]float32, 0),
		savedPreActs: make([][]float32, 0),
	}

	if metal, ok := device.(*MetalDevice); ok && metal.IsAvailable() {
		d.bufWeights = metal.CreateBuffer(weights)
		d.bufBiases = metal.CreateBuffer(biases)
		d.bufIn = metal.CreateEmptyBuffer(in)
		d.bufOut = metal.CreateEmptyBuffer(out)
	}

	return d
}

func (d *Dense) Forward(x []float32) []float32 {
	copy(d.inputBuf, x)
	d.saveInput(x)

	outSize := d.outSize
	inSize := d.inSize
	weights := d.weights
	biases := d.biases
	input := d.inputBuf
	preAct := d.preActBuf
	output := d.outputBuf

	if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() && d.bufWeights != nil {
		d.bufIn.Update(x)
		metal.MatMulPersistent(d.bufWeights, d.bufIn, d.bufOut, outSize, 1, inSize, 0.0)
		// We add biases manually or could implement persistent matmul with bias
		d.bufOut.Read(preAct)
		for i := 0; i < outSize; i++ {
			preAct[i] += biases[i]
		}
	} else {
		for o := 0; o < outSize; o++ {
			sum := biases[o]
			wBase := o * inSize
			for i := 0; i < inSize; i++ {
				sum += weights[wBase+i] * input[i]
			}
			preAct[o] = sum
		}
	}

	d.savePreAct(preAct)

	if batchAct, ok := d.act.(activations.BatchActivation); ok {
		copy(output, preAct)
		batchAct.ActivateBatch(output)
		if cap(d.savedOutput) < outSize {
			d.savedOutput = make([]float32, outSize)
		}
		copy(d.savedOutput[:outSize], output[:outSize])
	} else {
		for o := 0; o < outSize; o++ {
			output[o] = d.act.Activate(preAct[o])
		}
	}

	return output[:outSize]
}

func (d *Dense) Backward(grad []float32) []float32 {
	outSize := d.outSize
	inSize := d.inSize
	weights := d.weights
	input := d.inputBuf
	dz := d.dzBuf
	gradW := d.gradWBuf
	gradB := d.gradBBuf
	gradIn := d.gradInBuf

	if _, isBatchAct := d.act.(activations.BatchActivation); isBatchAct {
		output := d.savedOutput[:outSize]
		if _, isSoftmax := d.act.(activations.Softmax); isSoftmax {
			for k := 0; k < outSize; k++ {
				sum := float32(0.0)
				outK := output[k]
				for i := 0; i < outSize; i++ {
					delta := float32(0.0)
					if i == k { delta = 1.0 }
					sum += grad[i] * output[i] * (delta - outK)
				}
				dz[k] = sum
				gradB[k] = sum
			}
		} else if _, isLogSoftmax := d.act.(activations.LogSoftmax); isLogSoftmax {
			for k := 0; k < outSize; k++ {
				sum := float32(0.0)
				softmaxK := float32(math.Exp(float64(output[k])))
				for i := 0; i < outSize; i++ {
					delta := float32(0.0)
					if i == k { delta = 1.0 }
					sum += grad[i] * (delta - softmaxK)
				}
				dz[k] = sum
				gradB[k] = sum
			}
		}
	} else {
		for o := 0; o < outSize; o++ {
			deriv := d.act.Derivative(d.preActBuf[o])
			dz[o] = grad[o] * deriv
			gradB[o] += dz[o]
		}
	}

	for o := 0; o < outSize; o++ {
		dzo := dz[o]
		wBase := o * inSize
		for i := 0; i < inSize; i++ {
			gradW[wBase+i] += dzo * input[i]
		}
	}

	if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
		metal.MatMul(dz, weights, gradIn, 1, inSize, outSize)
	} else {
		const chunkSize = 8
		for chunk := 0; chunk < inSize; chunk += chunkSize {
			limit := chunk + chunkSize
			if limit > inSize { limit = inSize }
			for i := chunk; i < limit; i++ {
				sum := float32(0.0)
				for o := 0; o < outSize; o++ {
					sum += dz[o] * weights[o*inSize+i]
				}
				gradIn[i] = sum
			}
		}
	}

	return gradIn[:inSize]
}

func (d *Dense) Params() []float32 {
	return d.params
}

func (d *Dense) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &d.params[0] != &params[0] {
		if len(params) == len(d.params) {
			d.params = params
			d.updateViews()
		} else {
			copy(d.params, params)
		}
	}
	d.syncGPU()
}

func (d *Dense) updateViews() {
	d.weights = d.params[:d.outSize*d.inSize]
	d.biases = d.params[d.outSize*d.inSize:]
}

func (d *Dense) syncGPU() {
	if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() && d.bufWeights != nil {
		d.bufWeights.Update(d.weights)
		d.bufBiases.Update(d.biases)
	}
}

func (d *Dense) Gradients() []float32 {
	return d.grads
}

func (d *Dense) SetGradients(gradients []float32) {
	if len(gradients) == 0 {
		return
	}
	if &d.grads[0] != &gradients[0] {
		if len(gradients) == len(d.grads) {
			d.grads = gradients
			d.updateGradViews()
		} else {
			copy(d.grads, gradients)
		}
	}
}

func (d *Dense) updateGradViews() {
	d.gradWBuf = d.grads[:d.outSize*d.inSize]
	d.gradBBuf = d.grads[d.outSize*d.inSize:]
}

func (d *Dense) SetDevice(device Device) {
	d.device = device
}

func (d *Dense) InSize() int { return d.inSize }
func (d *Dense) OutSize() int { return d.outSize }

func (d *Dense) NamedParams() []NamedParam {
	return []NamedParam{
		{
			Name:  "weights",
			Shape: []int{d.outSize, d.inSize},
			Data:  d.weights,
		},
		{
			Name:  "biases",
			Shape: []int{d.outSize},
			Data:  d.biases,
		},
	}
}

// GetActivation returns the activation function.
func (d *Dense) GetActivation() activations.Activation {
	return d.act
}

// GetWeights returns the weight matrix.
func (d *Dense) GetWeights() []float32 {
	return d.weights
}

// SetWeight sets a specific weight.
func (d *Dense) SetWeight(out, in int, val float32) {
	d.weights[out*d.inSize+in] = val
}

// GetWeight gets a specific weight.
func (d *Dense) GetWeight(out, in int) float32 {
	return d.weights[out*d.inSize+in]
}

// GetBiases returns the biases.
func (d *Dense) GetBiases() []float32 {
	return d.biases
}

// SetBias sets a specific bias.
func (d *Dense) SetBias(out int, val float32) {
	d.biases[out] = val
}

// GetBias gets a specific bias.
func (d *Dense) GetBias(out int) float32 {
	return d.biases[out]
}

func (d *Dense) Reset() {
	d.savedInputs = d.savedInputs[:0]
	d.savedPreActs = d.savedPreActs[:0]
	d.savedOutput = d.savedOutput[:0]
}

func (d *Dense) ClearGradients() {
	for i := range d.grads {
		d.grads[i] = 0
	}
}

func (d *Dense) Clone() Layer {
	newD := NewDense(d.inSize, d.outSize, d.act)
	copy(newD.params, d.params)
	newD.device = d.device
	return newD
}

func (d *Dense) saveInput(x []float32) {
	// Simple append to avoid complex logic that leads to panics
	// and fix the regression. We can optimize this later if needed.
	saved := make([]float32, len(x))
	copy(saved, x)
	d.savedInputs = append(d.savedInputs, saved)
}

func (d *Dense) savePreAct(preAct []float32) {
	saved := make([]float32, len(preAct))
	copy(saved, preAct)
	d.savedPreActs = append(d.savedPreActs, saved)
}

func (d *Dense) AccumulateBackward(grad []float32) []float32 {
	outSize := d.outSize
	inSize := d.inSize
	weights := d.weights
	dz := d.dzBuf
	gradW := d.gradWBuf
	gradB := d.gradBBuf
	gradIn := d.gradInBuf

	numSaved := len(d.savedInputs)
	if numSaved == 0 { return nil }

	s := numSaved - 1
	savedInput := d.savedInputs[s]
	savedPreAct := d.savedPreActs[s]

	if _, isBatchAct := d.act.(activations.BatchActivation); isBatchAct {
		output := make([]float32, outSize)
		copy(output, savedPreAct)
		d.act.(activations.BatchActivation).ActivateBatch(output)

		if _, isSoftmax := d.act.(activations.Softmax); isSoftmax {
			for k := 0; k < outSize; k++ {
				sum := float32(0.0)
				outK := output[k]
				for i := 0; i < outSize; i++ {
					delta := float32(0.0)
					if i == k { delta = 1.0 }
					sum += grad[i] * output[i] * (delta - outK)
				}
				dz[k] = sum
			}
		} else if _, isLogSoftmax := d.act.(activations.LogSoftmax); isLogSoftmax {
			for k := 0; k < outSize; k++ {
				sum := float32(0.0)
				softmaxK := float32(math.Exp(float64(output[k])))
				for i := 0; i < outSize; i++ {
					delta := float32(0.0)
					if i == k { delta = 1.0 }
					sum += grad[i] * (delta - softmaxK)
				}
				dz[k] = sum
			}
		}
	} else {
		for o := 0; o < outSize; o++ {
			deriv := d.act.Derivative(savedPreAct[o])
			dz[o] = grad[o] * deriv
		}
	}

	for o := 0; o < outSize; o++ {
		dzo := dz[o]
		gradB[o] += dzo
		wBase := o * inSize
		for i := 0; i < inSize; i++ {
			gradW[wBase+i] += dzo * savedInput[i]
		}
	}

	for i := 0; i < inSize; i++ {
		sum := float32(0.0)
		for o := 0; o < outSize; o++ {
			sum += dz[o] * weights[o*inSize+i]
		}
		gradIn[i] = sum
	}

	d.savedInputs = d.savedInputs[:s]
	d.savedPreActs = d.savedPreActs[:s]

	return gradIn[:inSize]
}
