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

// Dense is a fully connected layer optimized for performance.
type Dense struct {
	weights   []float32
	biases    []float32
	act       activations.Activation
	outSize   int
	inSize    int
	device    Device

	// Reusable buffers
	inputBuf      []float32
	outputBuf     []float32
	preActBuf     []float32
	gradWBuf      []float32
	gradBBuf      []float32
	gradInBuf     []float32
	dzBuf         []float32

	savedInputs  [][]float32
	savedPreActs [][]float32
	savedOutput []float32
}

// NewDense creates a new dense layer with pre-allocated buffers.
func NewDense(in, out int, act activations.Activation) *Dense {
	return NewDenseWithDevice(in, out, act, &CPUDevice{})
}

// NewDenseWithDevice creates a new dense layer with a specific device.
func NewDenseWithDevice(in, out int, act activations.Activation, device Device) *Dense {
	weights := make([]float32, out*in)
	biases := make([]float32, out)

	scale := float32(math.Sqrt(2.0 / (float64(in) + float64(out))))
	rng := NewRNG(uint64(in*1000 + out*100 + 42))
	for i := range weights {
		weights[i] = rng.RandFloat()*2*scale - scale
	}
	for i := range biases {
		biases[i] = rng.RandFloat()*0.2 - 0.1
	}

	return &Dense{
		weights:      weights,
		biases:       biases,
		act:          act,
		outSize:      out,
		inSize:       in,
		device:       device,
		inputBuf:     make([]float32, in),
		outputBuf:    make([]float32, out),
		preActBuf:    make([]float32, out),
		gradWBuf:     make([]float32, out*in),
		gradBBuf:     make([]float32, out),
		gradInBuf:    make([]float32, in),
		dzBuf:        make([]float32, out),
		savedInputs:  make([][]float32, 0),
		savedPreActs: make([][]float32, 0),
	}
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

	if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
		metal.MatMulFused(d.weights, d.inputBuf, d.biases, d.preActBuf, outSize, 1, inSize, ActivationNone)
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
	total := len(d.weights) + len(d.biases)
	params := make([]float32, total)
	copy(params, d.weights)
	copy(params[len(d.weights):], d.biases)
	return params
}

func (d *Dense) SetParams(params []float32) {
	copy(d.weights, params[:len(d.weights)])
	copy(d.biases, params[len(d.weights):])
}

func (d *Dense) Gradients() []float32 {
	total := len(d.gradWBuf) + len(d.gradBBuf)
	gradients := make([]float32, total)
	copy(gradients, d.gradWBuf)
	copy(gradients[len(d.gradWBuf):], d.gradBBuf)
	return gradients
}

func (d *Dense) SetGradients(gradients []float32) {
	copy(d.gradWBuf, gradients[:len(d.gradWBuf)])
	copy(d.gradBBuf, gradients[len(d.gradWBuf):])
}

func (d *Dense) SetDevice(device Device) {
	d.device = device
}

func (d *Dense) InSize() int { return d.inSize }
func (d *Dense) OutSize() int { return d.outSize }

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
	for i := range d.gradWBuf { d.gradWBuf[i] = 0 }
	for i := range d.gradBBuf { d.gradBBuf[i] = 0 }
}

func (d *Dense) Clone() Layer {
	newD := NewDense(d.inSize, d.outSize, d.act)
	copy(newD.weights, d.weights)
	copy(newD.biases, d.biases)
	newD.device = d.device
	return newD
}

func (d *Dense) saveInput(x []float32) {
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
