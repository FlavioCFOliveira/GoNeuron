// Package layer provides neural network layer implementations.
package layer

import (
	"math"
)

// SwiGLU implements the Swish-Gated Linear Unit.
// FFN_SwiGLU(x, W, V, b, c) = (Swish(xW + b) * (xV + c))
// In Transformers, it is typically used without biases.
type SwiGLU struct {
	inSize  int
	outSize int

	// Projections: W (gate) and V (up)
	// We store them in a single contiguous block for efficiency
	params []float32 // [W | V]
	w      []float32 // View of params
	v      []float32 // View of params

	grads  []float32 // [gradW | gradV]
	gradW  []float32 // View of grads
	gradV  []float32 // View of grads

	// Reusable buffers
	inputBuf  []float32
	gateBuf   []float32
	upBuf     []float32
	outputBuf []float32
	gradInBuf []float32

	// For backward pass
	savedInputOffsets []int
	savedGateOffsets  []int
	savedUpOffsets    []int
	arena             []float32

	training bool
	device   Device
}

// NewSwiGLU creates a new SwiGLU layer.
func NewSwiGLU(in, out int) *SwiGLU {
	s := &SwiGLU{
		inSize:            in,
		outSize:           out,
		device:            &CPUDevice{},
		savedInputOffsets: make([]int, 0, 16),
		savedGateOffsets:  make([]int, 0, 16),
		savedUpOffsets:    make([]int, 0, 16),
	}

	if in != -1 {
		s.Build(in)
	}

	return s
}

// Build initializes the layer.
func (s *SwiGLU) Build(in int) {
	s.inSize = in
	out := s.outSize

	// Total parameters: 2 * in * out (no biases)
	s.params = make([]float32, 2*in*out)
	s.w = s.params[:in*out]
	s.v = s.params[in*out:]

	// Xavier initialization
	scale := float32(math.Sqrt(2.0 / (float64(in) + float64(out))))
	rng := NewRNG(uint64(in*100 + out + 42))
	for i := range s.params {
		s.params[i] = rng.RandFloat()*2*scale - scale
	}

	s.grads = make([]float32, 2*in*out)
	s.gradW = s.grads[:in*out]
	s.gradV = s.grads[in*out:]

	s.inputBuf = make([]float32, in)
	s.gateBuf = make([]float32, out)
	s.upBuf = make([]float32, out)
	s.outputBuf = make([]float32, out)
	s.gradInBuf = make([]float32, in)
}

func (s *SwiGLU) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
	copy(s.inputBuf, x)

	inSize := s.inSize
	outSize := s.outSize

	// Save input
	if arena != nil && offset != nil {
		s.arena = *arena
		if len(*arena) < *offset+inSize {
			newArena := make([]float32, (*offset+inSize)*2)
			copy(newArena, *arena)
			*arena = newArena
			s.arena = *arena
		}
		copy((*arena)[*offset:*offset+inSize], x)
		s.savedInputOffsets = append(s.savedInputOffsets, *offset)
		*offset += inSize
	} else {
		// Internal arena fallback
		if cap(s.arena) < inSize {
			s.arena = make([]float32, inSize*2)
		}
		s.arena = s.arena[:inSize]
		copy(s.arena, x)
		s.savedInputOffsets = append(s.savedInputOffsets[:0], 0)
		s.savedGateOffsets = append(s.savedGateOffsets[:0], -1) // Marked for internal use
		s.savedUpOffsets = append(s.savedUpOffsets[:0], -1)
	}

	// Compute gate (W) and up (V) projections
	for o := 0; o < outSize; o++ {
		sumW := float32(0.0)
		sumV := float32(0.0)
		wBase := o * inSize
		for i := 0; i < inSize; i++ {
			sumW += s.w[wBase+i] * x[i]
			sumV += s.v[wBase+i] * x[i]
		}
		s.gateBuf[o] = sumW
		s.upBuf[o] = sumV
	}

	// Save pre-activations
	if arena != nil && offset != nil {
		if len(*arena) < *offset+2*outSize {
			newArena := make([]float32, (*offset+2*outSize)*2)
			copy(newArena, *arena)
			*arena = newArena
			s.arena = *arena
		}
		copy((*arena)[*offset:*offset+outSize], s.gateBuf)
		s.savedGateOffsets = append(s.savedGateOffsets, *offset)
		*offset += outSize

		copy((*arena)[*offset:*offset+outSize], s.upBuf)
		s.savedUpOffsets = append(s.savedUpOffsets, *offset)
		*offset += outSize
	} else {
		// Use internal buffers for gate and up
		// We'll just use gateBuf and upBuf directly in Backward if offsets are -1
	}

	// SwiGLU(x) = Swish(gate) * up
	// Swish(x) = x * sigmoid(x)
	for i := 0; i < outSize; i++ {
		g := s.gateBuf[i]
		sig := 1.0 / (1.0 + float32(math.Exp(float64(-g))))
		s.outputBuf[i] = (g * sig) * s.upBuf[i]
	}

	return s.outputBuf
}

func (s *SwiGLU) Forward(x []float32) []float32 {
	return s.ForwardWithArena(x, nil, nil)
}

func (s *SwiGLU) Backward(grad []float32) []float32 {
	numSaved := len(s.savedInputOffsets)
	if numSaved == 0 {
		return nil
	}

	ts := numSaved - 1
	inOff := s.savedInputOffsets[ts]
	gateOff := s.savedGateOffsets[ts]
	upOff := s.savedUpOffsets[ts]

	var savedInput, savedGate, savedUp []float32
	if gateOff == -1 {
		savedInput = s.arena
		savedGate = s.gateBuf
		savedUp = s.upBuf
	} else {
		savedInput = s.arena[inOff : inOff+s.inSize]
		savedGate = s.arena[gateOff : gateOff+s.outSize]
		savedUp = s.arena[upOff : upOff+s.outSize]
	}

	inSize := s.inSize
	outSize := s.outSize

	// SwiGLU = Swish(g) * u
	// dSwiGLU/du = Swish(g)
	// dSwiGLU/dg = dSwish(g)/dg * u
	// dSwish(g)/dg = Swish(g) + sigmoid(g) * (1 - Swish(g))

	dGate := make([]float32, outSize)
	dUp := make([]float32, outSize)

	for i := 0; i < outSize; i++ {
		g := savedGate[i]
		u := savedUp[i]
		sig := 1.0 / (1.0 + float32(math.Exp(float64(-g))))
		swish := g * sig

		dUp[i] = grad[i] * swish

		dSwish := swish + sig*(1.0-swish)
		dGate[i] = grad[i] * dSwish * u
	}

	// Backprop to params and input
	for i := 0; i < inSize; i++ {
		s.gradInBuf[i] = 0
	}

	for o := 0; o < outSize; o++ {
		dg := dGate[o]
		du := dUp[o]
		wBase := o * inSize
		for i := 0; i < inSize; i++ {
			s.gradW[wBase+i] += dg * savedInput[i]
			s.gradV[wBase+i] += du * savedInput[i]
			s.gradInBuf[i] += dg*s.w[wBase+i] + du*s.v[wBase+i]
		}
	}

	s.savedInputOffsets = s.savedInputOffsets[:ts]
	s.savedGateOffsets = s.savedGateOffsets[:ts]
	s.savedUpOffsets = s.savedUpOffsets[:ts]

	return s.gradInBuf
}

func (s *SwiGLU) Params() []float32 {
	return s.params
}

func (s *SwiGLU) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if &s.params[0] != &params[0] {
		if len(params) == len(s.params) {
			s.params = params
			s.w = params[:s.inSize*s.outSize]
			s.v = params[s.inSize*s.outSize:]
		} else {
			copy(s.params, params)
		}
	}
}

func (s *SwiGLU) Gradients() []float32 {
	return s.grads
}

func (s *SwiGLU) SetGradients(gradients []float32) {
	if len(gradients) == 0 {
		return
	}
	if &s.grads[0] != &gradients[0] {
		if len(gradients) == len(s.grads) {
			s.grads = gradients
			s.gradW = gradients[:s.inSize*s.outSize]
			s.gradV = gradients[s.inSize*s.outSize:]
		} else {
			copy(s.grads, gradients)
		}
	}
}

func (s *SwiGLU) InSize() int {
	return s.inSize
}

func (s *SwiGLU) OutSize() int {
	return s.outSize
}

func (s *SwiGLU) NamedParams() []NamedParam {
	return []NamedParam{
		{
			Name:  "w",
			Shape: []int{s.outSize, s.inSize},
			Data:  s.w,
		},
		{
			Name:  "v",
			Shape: []int{s.outSize, s.inSize},
			Data:  s.v,
		},
	}
}

func (s *SwiGLU) Reset() {
	s.savedInputOffsets = s.savedInputOffsets[:0]
	s.savedGateOffsets = s.savedGateOffsets[:0]
	s.savedUpOffsets = s.savedUpOffsets[:0]
}

func (s *SwiGLU) ClearGradients() {
	for i := range s.grads {
		s.grads[i] = 0
	}
}

func (s *SwiGLU) Clone() Layer {
	newS := NewSwiGLU(s.inSize, s.outSize)
	copy(newS.params, s.params)
	newS.device = s.device
	return newS
}

func (s *SwiGLU) LightweightClone(params []float32, grads []float32) Layer {
	newS := &SwiGLU{
		inSize:            s.inSize,
		outSize:           s.outSize,
		params:            params,
		w:                 params[:s.inSize*s.outSize],
		v:                 params[s.inSize*s.outSize:],
		grads:             grads,
		gradW:             grads[:s.inSize*s.outSize],
		gradV:             grads[s.inSize*s.outSize:],
		inputBuf:          make([]float32, s.inSize),
		gateBuf:           make([]float32, s.outSize),
		upBuf:             make([]float32, s.outSize),
		outputBuf:         make([]float32, s.outSize),
		gradInBuf:         make([]float32, s.inSize),
		savedInputOffsets: make([]int, 0, 128),
		savedGateOffsets:  make([]int, 0, 128),
		savedUpOffsets:    make([]int, 0, 128),
		training:          s.training,
		device:            s.device,
	}
	return newS
}

func (s *SwiGLU) AccumulateBackward(grad []float32) []float32 {
	return s.Backward(grad)
}

func (s *SwiGLU) SetDevice(device Device) {
	s.device = device
}

func (s *SwiGLU) SetTraining(training bool) {
	s.training = training
}
