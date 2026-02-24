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

// RandUint64 returns a random uint64.
func (r *RNG) RandUint64() uint64 {
	r.seed = r.seed*6364136223846793005 + 1
	return r.seed
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

	// Training state
	SetTraining(bool)

	// Batch methods
	ForwardBatch(x []float32, batchSize int) []float32
	BackwardBatch(grad []float32, batchSize int) []float32
	AccumulateBackwardBatch(grad []float32, batchSize int) []float32

	// LightweightClone creates a new layer that shares the same parameters and gradients
	// but has its own internal reusable buffers.
	LightweightClone(params []float32, grads []float32) Layer
}

// DeferredLayer is a layer that can defer its initialization until the input size is known.
type DeferredLayer interface {
	Layer
	// Build initializes the layer with the given input size.
	Build(inSize int)
}

// ArenaLayer is an optional interface for layers that support zero-allocation activation saving.
type ArenaLayer interface {
	Layer
	ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32
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

func getActivationType(act activations.Activation) int {
	switch act.(type) {
	case activations.Sigmoid, *activations.Sigmoid:
		return MetalActivationSigmoid
	case activations.Tanh, *activations.Tanh:
		return MetalActivationTanh
	case activations.ReLU, *activations.ReLU:
		return MetalActivationReLU
	case activations.GELU, *activations.GELU:
		return MetalActivationGELU
	default:
		return MetalActivationNone
	}
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

	training bool

	savedInputOffsets  []int
	savedPreActOffsets []int
	savedOutput        []float32
	arenaPtr           *[]float32

	// Metal persistent buffers
	bufWeights *MetalBuffer
	bufBiases  *MetalBuffer
	bufIn      *MetalBuffer
	bufOut     *MetalBuffer
	bufGradW   *MetalBuffer
	bufGradB   *MetalBuffer
	bufGradIn  *MetalBuffer
	bufDZ      *MetalBuffer
}

// NewDense creates a new dense layer with pre-allocated buffers.
func NewDense(in, out int, act activations.Activation) *Dense {
	return NewDenseWithDevice(in, out, act, &CPUDevice{})
}

// NewDenseWithDevice creates a new dense layer with a specific device.
func NewDenseWithDevice(in, out int, act activations.Activation, device Device) *Dense {
	d := &Dense{
		act:                act,
		outSize:            out,
		inSize:             in,
		device:             device,
		savedInputOffsets:  make([]int, 0, 1),
		savedPreActOffsets: make([]int, 0, 1),
	}

	if in != -1 && out != -1 {
		d.Build(in)
	}

	return d
}

// Build initializes the layer with the given input size.
func (d *Dense) Build(in int) {
	// Robustness for deferred initialization
	if in <= 0 || d.outSize <= 0 {
		return
	}
	d.inSize = in
	out := d.outSize

	d.params = make([]float32, out*in+out)
	d.weights = d.params[:out*in]
	d.biases = d.params[out*in:]

	scale := float32(math.Sqrt(2.0 / (float64(in) + float64(out))))
	rng := NewRNG(uint64(in*1000 + out*100 + 42))
	for i := range d.weights {
		d.weights[i] = rng.RandFloat()*2*scale - scale
	}
	for i := range d.biases {
		d.biases[i] = rng.RandFloat()*0.2 - 0.1
	}

	d.grads = make([]float32, out*in+out)
	d.gradWBuf = d.grads[:out*in]
	d.gradBBuf = d.grads[out*in:]

	d.inputBuf = make([]float32, in)
	d.outputBuf = make([]float32, out)
	d.preActBuf = make([]float32, out)
	d.gradInBuf = make([]float32, in)
	d.dzBuf = make([]float32, out)

	if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
		d.bufWeights = metal.CreateBuffer(d.weights)
		d.bufBiases = metal.CreateBuffer(d.biases)
		d.bufIn = metal.CreateEmptyBuffer(in)
		d.bufOut = metal.CreateEmptyBuffer(out)
		d.bufGradW = metal.CreateEmptyBuffer(out * in)
		d.bufGradB = metal.CreateEmptyBuffer(out)
		d.bufGradIn = metal.CreateEmptyBuffer(in)
		d.bufDZ = metal.CreateEmptyBuffer(out)
	}
}

func (d *Dense) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	copy(d.inputBuf, x)

	// Save input to arena
	if arena != nil && offset != nil {
		d.arenaPtr = arena
		inSize := len(x)
		if len(*arena) < *offset+inSize {
			// Grow arena if needed
			newArena := make([]float32, (*offset+inSize)*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		saved := (*arena)[*offset : *offset+inSize]
		copy(saved, x)
		d.savedInputOffsets = append(d.savedInputOffsets, *offset)
		*offset += inSize
	} else {
		d.saveInput(x)
	}

	outSize := d.outSize
	inSize := d.inSize
	weights := d.weights
	biases := d.biases
	input := d.inputBuf
	preAct := d.preActBuf
	output := d.outputBuf

	metalUsed := false
	if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() && d.bufWeights != nil {
		d.bufIn.Update(x)
		actType := MetalActivationNone
		if !d.training {
			actType = getActivationType(d.act)
		}
		metal.MatMulFusedPersistent(d.bufWeights, d.bufIn, d.bufBiases, d.bufOut, outSize, 1, inSize, actType)
		if actType != MetalActivationNone {
			d.bufOut.Read(output)
			copy(preAct, output)
		} else {
			d.bufOut.Read(preAct)
		}
		metalUsed = true
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

	// Save pre-activation to arena
	if arena != nil && offset != nil {
		if len(*arena) < *offset+outSize {
			newArena := make([]float32, (*offset+outSize)*2)
			copy(newArena, *arena)
			*arena = newArena
		}
		saved := (*arena)[*offset : *offset+outSize]
		copy(saved, preAct)
		d.savedPreActOffsets = append(d.savedPreActOffsets, *offset)
		*offset += outSize
	} else {
		d.savePreAct(preAct)
	}

	if !metalUsed || d.training {
		if batchAct, ok := d.act.(activations.BatchActivation); ok {
			copy(output, preAct)
			batchAct.ActivateBatch(output)
		} else {
			for o := 0; o < outSize; o++ {
				output[o] = d.act.Activate(preAct[o])
			}
		}
	}

	if cap(d.savedOutput) < outSize {
		d.savedOutput = make([]float32, outSize)
	}
	copy(d.savedOutput[:outSize], output[:outSize])

	return output[:outSize]
}

func (d *Dense) Forward(x []float32) []float32 {
	return d.ForwardWithArena(x, nil, nil)
}

func (d *Dense) ForwardBatch(x []float32, batchSize int) []float32 {
	return d.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (d *Dense) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 {
	if batchSize <= 1 {
		return d.ForwardWithArena(x, arena, offset)
	}
	inSize := d.inSize
	outSize := d.outSize
	if len(d.outputBuf) < batchSize*outSize {
		d.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out := d.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		copy(d.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return d.outputBuf[:batchSize*outSize]
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
			if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
				d.bufOut.Update(grad)
				d.bufDZ.Update(output)
				metal.SoftmaxBackwardPersistent(d.bufOut, d.bufDZ, d.bufDZ, 1, outSize)
				d.bufDZ.Read(dz)
				for k := 0; k < outSize; k++ {
					gradB[k] += dz[k]
				}
			} else {
				for k := 0; k < outSize; k++ {
					sum := float32(0.0)
					outK := output[k]
					for i := 0; i < outSize; i++ {
						delta := float32(0.0)
						if i == k {
							delta = 1.0
						}
						sum += grad[i] * output[i] * (delta - outK)
					}
					dz[k] = sum
					gradB[k] += sum
				}
			}
		} else if _, isLogSoftmax := d.act.(activations.LogSoftmax); isLogSoftmax {
			if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
				// For LogSoftmax, we need softmax(preAct)
				softmax := make([]float32, outSize)
				for i := range softmax {
					softmax[i] = float32(math.Exp(float64(output[i])))
				}
				d.bufOut.Update(grad)
				d.bufDZ.Update(softmax)
				metal.SoftmaxBackwardPersistent(d.bufOut, d.bufDZ, d.bufDZ, 1, outSize)
				d.bufDZ.Read(dz)
				for k := 0; k < outSize; k++ {
					gradB[k] += dz[k]
				}
			} else {
				for k := 0; k < outSize; k++ {
					sum := float32(0.0)
					softmaxK := float32(math.Exp(float64(output[k])))
					for i := 0; i < outSize; i++ {
						delta := float32(0.0)
						if i == k {
							delta = 1.0
						}
						sum += grad[i] * (delta - softmaxK)
					}
					dz[k] = sum
					gradB[k] += sum
				}
			}
		}
	} else {
		for o := 0; o < outSize; o++ {
			deriv := d.act.Derivative(d.preActBuf[o])
			dz[o] = grad[o] * deriv
			gradB[o] += dz[o]
		}
	}

	if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
		d.bufOut.Update(dz) // dz has same size as grad (outSize)
		d.bufIn.Update(input)
		d.bufGradW.Update(gradW)
		d.bufGradB.Update(gradB)

		metal.DenseBackwardPersistent(d.bufOut, d.bufIn, d.bufGradW, d.bufGradB, 1, inSize, outSize)
		metal.MatMulTransposePersistent(d.bufOut, d.bufWeights, d.bufGradIn, 1, inSize, outSize, 0.0, false, false)

		d.bufGradW.Read(gradW)
		d.bufGradB.Read(gradB)
		d.bufGradIn.Read(gradIn)
	} else {
		for o := 0; o < outSize; o++ {
			dzo := dz[o]
			wBase := o * inSize
			for i := 0; i < inSize; i++ {
				gradW[wBase+i] += dzo * input[i]
			}
		}

		for i := 0; i < inSize; i++ {
			sum := float32(0.0)
			for o := 0; o < outSize; o++ {
				sum += dz[o] * weights[o*inSize+i]
			}
			gradIn[i] = sum
		}
	}

	return gradIn[:inSize]
}

func (d *Dense) BackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return d.Backward(grad)
	}
	inSize := d.inSize
	outSize := d.outSize

	if len(d.gradInBuf) < batchSize*inSize {
		d.gradInBuf = make([]float32, batchSize*inSize)
	}

	for i := batchSize - 1; i >= 0; i-- {
		dx := d.Backward(grad[i*outSize : (i+1)*outSize])
		copy(d.gradInBuf[i*inSize:(i+1)*inSize], dx)
	}
	return d.gradInBuf[:batchSize*inSize]
}

func (d *Dense) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 {
	if batchSize <= 1 {
		return d.AccumulateBackward(grad)
	}
	inSize := d.inSize
	outSize := d.outSize
	if len(d.gradInBuf) < batchSize*inSize {
		d.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		dx := d.AccumulateBackward(grad[i*outSize : (i+1)*outSize])
		copy(d.gradInBuf[i*inSize:(i+1)*inSize], dx)
	}
	return d.gradInBuf[:batchSize*inSize]
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
	if md, ok := device.(*MetalDevice); ok && md.IsAvailable() && d.inSize > 0 {
		if d.bufWeights == nil {
			d.bufWeights = md.CreateBuffer(d.weights)
		}
		if d.bufBiases == nil {
			d.bufBiases = md.CreateBuffer(d.biases)
		}
		if d.bufIn == nil {
			d.bufIn = md.CreateEmptyBuffer(d.inSize)
		}
		if d.bufOut == nil {
			d.bufOut = md.CreateEmptyBuffer(d.outSize)
		}
		if d.bufGradW == nil {
			d.bufGradW = md.CreateEmptyBuffer(d.outSize * d.inSize)
		}
		if d.bufGradB == nil {
			d.bufGradB = md.CreateEmptyBuffer(d.outSize)
		}
		if d.bufGradIn == nil {
			d.bufGradIn = md.CreateEmptyBuffer(d.inSize)
		}
		if d.bufDZ == nil {
			d.bufDZ = md.CreateEmptyBuffer(d.outSize)
		}
	}
}

func (d *Dense) SetTraining(training bool) {
	d.training = training
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
	d.savedInputOffsets = d.savedInputOffsets[:0]
	d.savedPreActOffsets = d.savedPreActOffsets[:0]
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

func (d *Dense) LightweightClone(params []float32, grads []float32) Layer {
	newD := &Dense{
		params:             params,
		weights:            params[:d.outSize*d.inSize],
		biases:             params[d.outSize*d.inSize:],
		act:                d.act,
		outSize:            d.outSize,
		inSize:             d.inSize,
		device:             d.device,
		inputBuf:           make([]float32, d.inSize),
		outputBuf:          make([]float32, d.outSize),
		preActBuf:          make([]float32, d.outSize),
		grads:              grads,
		gradWBuf:           grads[:d.outSize*d.inSize],
		gradBBuf:           grads[d.outSize*d.inSize:],
		gradInBuf:          make([]float32, d.inSize),
		dzBuf:              make([]float32, d.outSize),
		savedInputOffsets:  make([]int, 0, 16),
		savedPreActOffsets: make([]int, 0, 16),
		training:           d.training,
	}

	if md, ok := d.device.(*MetalDevice); ok && md.IsAvailable() && d.inSize > 0 {
		newD.bufWeights = md.CreateBuffer(newD.weights)
		newD.bufBiases = md.CreateBuffer(newD.biases)
		newD.bufIn = md.CreateEmptyBuffer(d.inSize)
		newD.bufOut = md.CreateEmptyBuffer(d.outSize)
		newD.bufGradW = md.CreateEmptyBuffer(d.outSize * d.inSize)
		newD.bufGradB = md.CreateEmptyBuffer(d.outSize)
		newD.bufGradIn = md.CreateEmptyBuffer(d.inSize)
		newD.bufDZ = md.CreateEmptyBuffer(d.outSize)
	}

	return newD
}

func (d *Dense) saveInput(x []float32) {
	inSize := len(x)
	// We use the internal arena even when not provided from outside
	if d.arenaPtr == nil {
		d.arenaPtr = &[]float32{}
	}
	if cap(*d.arenaPtr) < inSize {
		*d.arenaPtr = make([]float32, inSize*2)
	}
	*d.arenaPtr = (*d.arenaPtr)[:inSize]
	copy(*d.arenaPtr, x)
	d.savedInputOffsets = append(d.savedInputOffsets[:0], 0)
}

func (d *Dense) savePreAct(preAct []float32) {
	outSize := len(preAct)
	inSize := len(*d.arenaPtr)
	offset := inSize
	if cap(*d.arenaPtr) < offset+outSize {
		newArena := make([]float32, (offset+outSize)*2)
		copy(newArena, *d.arenaPtr)
		*d.arenaPtr = newArena
	}
	*d.arenaPtr = (*d.arenaPtr)[:offset+outSize]
	copy((*d.arenaPtr)[offset:], preAct)
	d.savedPreActOffsets = append(d.savedPreActOffsets[:0], offset)
}

func (d *Dense) AccumulateBackward(grad []float32) []float32 {
	outSize := d.outSize
	inSize := d.inSize
	weights := d.weights
	dz := d.dzBuf
	gradW := d.gradWBuf
	gradB := d.gradBBuf
	gradIn := d.gradInBuf

	numSaved := len(d.savedInputOffsets)
	if numSaved == 0 {
		return nil
	}

	s := numSaved - 1
	inOff := d.savedInputOffsets[s]
	paOff := d.savedPreActOffsets[s]
	savedInput := (*d.arenaPtr)[inOff : inOff+inSize]
	savedPreAct := (*d.arenaPtr)[paOff : paOff+outSize]

	if _, isBatchAct := d.act.(activations.BatchActivation); isBatchAct {
		output := d.outputBuf[:outSize]
		copy(output, savedPreAct)
		d.act.(activations.BatchActivation).ActivateBatch(output)

		if _, isSoftmax := d.act.(activations.Softmax); isSoftmax {
			if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
				d.bufOut.Update(grad)
				d.bufDZ.Update(output) // Using bufDZ as temp for 'output'
				metal.SoftmaxBackwardPersistent(d.bufOut, d.bufDZ, d.bufDZ, 1, outSize)
				d.bufDZ.Read(dz)
			} else {
				for k := 0; k < outSize; k++ {
					sum := float32(0.0)
					outK := output[k]
					for i := 0; i < outSize; i++ {
						delta := float32(0.0)
						if i == k {
							delta = 1.0
						}
						sum += grad[i] * output[i] * (delta - outK)
					}
					dz[k] = sum
				}
			}
		} else if _, isLogSoftmax := d.act.(activations.LogSoftmax); isLogSoftmax {
			if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
				// For LogSoftmax, we need softmax(preAct)
				softmax := make([]float32, outSize)
				for i := range softmax {
					softmax[i] = float32(math.Exp(float64(output[i])))
				}
				d.bufOut.Update(grad)
				d.bufDZ.Update(softmax)
				metal.SoftmaxBackwardPersistent(d.bufOut, d.bufDZ, d.bufDZ, 1, outSize)
				d.bufDZ.Read(dz)
			} else {
				for k := 0; k < outSize; k++ {
					sum := float32(0.0)
					softmaxK := float32(math.Exp(float64(output[k])))
					for i := 0; i < outSize; i++ {
						delta := float32(0.0)
						if i == k {
							delta = 1.0
						}
						sum += grad[i] * (delta - softmaxK)
					}
					dz[k] = sum
				}
			}
		}
	} else {
		for o := 0; o < outSize; o++ {
			deriv := d.act.Derivative(savedPreAct[o])
			dz[o] = grad[o] * deriv
		}
	}

	if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
		d.bufOut.Update(dz) // dz has same size as grad (outSize)
		d.bufIn.Update(savedInput)
		d.bufGradW.Update(gradW)
		d.bufGradB.Update(gradB)

		metal.DenseBackwardPersistent(d.bufOut, d.bufIn, d.bufGradW, d.bufGradB, 1, inSize, outSize)
		metal.MatMulTransposePersistent(d.bufOut, d.bufWeights, d.bufGradIn, 1, inSize, outSize, 0.0, false, false)

		d.bufGradW.Read(gradW)
		d.bufGradB.Read(gradB)
		d.bufGradIn.Read(gradIn)
	} else {
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
	}

	d.savedInputOffsets = d.savedInputOffsets[:s]
	d.savedPreActOffsets = d.savedPreActOffsets[:s]

	return gradIn[:inSize]
}
