package layer

import (
	"fmt"
)

// SequenceUnroller is a wrapper layer that unrolls a base layer over T time steps.
type SequenceUnroller struct {
	base        Layer
	timeSteps   int
	returnSeq   bool
	inputBuf    []float32
	outputBuf   []float32
	gradInBuf   []float32
	storedInput [][]float32
	training    bool
}

// NewSequenceUnroller creates a new sequence unroller layer.
// Returns nil if base layer is nil or timeSteps is invalid.
func NewSequenceUnroller(base Layer, timeSteps int, returnSeq bool) *SequenceUnroller {
	if base == nil {
		return nil
	}
	if timeSteps <= 0 {
		return nil
	}
	if base.InSize() <= 0 || base.OutSize() <= 0 {
		return nil
	}

	outSize := base.OutSize()
	if returnSeq {
		outSize *= timeSteps
	}

	return &SequenceUnroller{
		base:        base,
		timeSteps:   timeSteps,
		returnSeq:   returnSeq,
		inputBuf:    make([]float32, timeSteps*base.InSize()),
		outputBuf:   make([]float32, outSize),
		gradInBuf:   make([]float32, timeSteps*base.InSize()),
		storedInput: make([][]float32, timeSteps),
	}
}

// SetTraining sets the training mode for the base layer.
func (s *SequenceUnroller) SetTraining(training bool) {
	s.training = training
	s.base.SetTraining(training)
}

func (s *SequenceUnroller) ForwardWithArena(x []float32, arena *[]float32, offset *int) ([]float32, error) {
	inSize := s.base.InSize()
	outSize := s.base.OutSize()

	if len(x) != s.timeSteps*inSize {
		return nil, fmt.Errorf("SequenceUnroller: input size mismatch. Expected %d, got %d", s.timeSteps*inSize, len(x))
	}

	s.base.Reset()

	// Special case for Bidirectional: we need to populate its hidden states
	if bi, ok := s.base.(*Bidirectional); ok {
		for t := 0; t < s.timeSteps; t++ {
			if s.storedInput[t] == nil {
				s.storedInput[t] = make([]float32, inSize)
			}
			copy(s.storedInput[t], x[t*inSize:(t+1)*inSize])
			if arena != nil && offset != nil {
				bi.ForwardWithArena(s.storedInput[t], arena, offset)
			} else {
				bi.Forward(s.storedInput[t])
			}
		}
		bi.ComputeBackwardHiddenStates()

		for t := 0; t < s.timeSteps; t++ {
			fOut := bi.GetForwardOutputAt(t)
			bOut := bi.GetBackwardOutputAt(t)
			if s.returnSeq {
				copy(s.outputBuf[t*s.OutSize()/s.timeSteps:t*s.OutSize()/s.timeSteps+len(fOut)], fOut)
				copy(s.outputBuf[t*s.OutSize()/s.timeSteps+len(fOut):(t+1)*s.OutSize()/s.timeSteps], bOut)
			} else if t == s.timeSteps-1 {
				// For non-returning sequences, we want the "last" state from both directions.
				// Forward: last time step (t=T-1)
				// Backward: "last" time step in reverse (t=0)
				bOutLast := bi.GetBackwardOutputAt(0)
				copy(s.outputBuf[:len(fOut)], fOut)
				copy(s.outputBuf[len(fOut):], bOutLast)
			}
		}
	} else {
		for t := 0; t < s.timeSteps; t++ {
			if s.storedInput[t] == nil {
				s.storedInput[t] = make([]float32, inSize)
			}
			copy(s.storedInput[t], x[t*inSize:(t+1)*inSize])

			var out []float32
			var err error
			if al, ok := s.base.(ArenaLayer); ok && arena != nil && offset != nil {
				out, err = al.ForwardWithArena(s.storedInput[t], arena, offset)
				if err != nil {
					return nil, err
				}
			} else {
				out, err = s.base.Forward(s.storedInput[t])
				if err != nil {
					return nil, err
				}
			}

			if s.returnSeq {
				copy(s.outputBuf[t*outSize:(t+1)*outSize], out)
			} else if t == s.timeSteps-1 {
				copy(s.outputBuf, out)
			}
		}
	}

	return s.outputBuf[:s.timeSteps*s.base.OutSize()], nil
}

func (s *SequenceUnroller) Forward(x []float32) ([]float32, error) {
	return s.ForwardWithArena(x, nil, nil)
}

func (s *SequenceUnroller) Backward(grad []float32) ([]float32, error) {
	outSize := s.base.OutSize()
	inSize := s.base.InSize()

	if bi, ok := s.base.(*Bidirectional); ok {
		// Bidirectional BPTT
		fGradNext := make([]float32, bi.forward.OutSize())
		bGradNext := make([]float32, bi.backward.OutSize())

		fOutSize := bi.forward.OutSize()
		bOutSize := bi.backward.OutSize()

		// Forward layer: Backward from T-1 to 0
		for t := s.timeSteps - 1; t >= 0; t-- {
			currentGrad := make([]float32, fOutSize)
			copy(currentGrad, fGradNext)
			if s.returnSeq {
				for i := 0; i < fOutSize; i++ {
					currentGrad[i] += grad[t*(fOutSize+bOutSize)+i]
				}
			} else if t == s.timeSteps-1 {
				for i := 0; i < fOutSize; i++ {
					currentGrad[i] += grad[i]
				}
			}
			dx, err := bi.forward.Backward(currentGrad)
			if err != nil {
				return nil, err
			}
			copy(s.gradInBuf[t*inSize:(t+1)*inSize], dx)
			copy(fGradNext, currentGrad)
		}

		// Backward layer: Backward from 0 to T-1 (its own reverse)
		for t := 0; t < s.timeSteps; t++ {
			currentGrad := make([]float32, bOutSize)
			copy(currentGrad, bGradNext)
			if s.returnSeq {
				for i := 0; i < bOutSize; i++ {
					currentGrad[i] += grad[t*(fOutSize+bOutSize)+fOutSize+i]
				}
			} else if t == 0 { // In returnSeq=false, only last step (T-1) of sequence (first in time) has grad
				for i := 0; i < bOutSize; i++ {
					currentGrad[i] += grad[fOutSize+i]
				}
			}
			dx, err := bi.backward.Backward(currentGrad)
			if err != nil {
				return nil, err
			}
			// Add to gradInBuf (accumulate from both directions)
			for i := 0; i < inSize; i++ {
				s.gradInBuf[t*inSize+i] += dx[i]
			}
			copy(bGradNext, currentGrad)
		}
	} else {
		dhNext := make([]float32, outSize)
		for t := s.timeSteps - 1; t >= 0; t-- {
			currentGrad := make([]float32, outSize)
			copy(currentGrad, dhNext)
			if s.returnSeq {
				for i := 0; i < outSize; i++ {
					currentGrad[i] += grad[t*outSize+i]
				}
			} else if t == s.timeSteps-1 {
				for i := 0; i < outSize; i++ {
					currentGrad[i] += grad[i]
				}
			}
			dx, err := s.base.Backward(currentGrad)
			if err != nil {
				return nil, err
			}
			copy(s.gradInBuf[t*inSize:(t+1)*inSize], dx)
			copy(dhNext, currentGrad)
		}
	}

	return s.gradInBuf[:s.timeSteps*inSize], nil
}

func (s *SequenceUnroller) ForwardBatch(x []float32, batchSize int) ([]float32, error) {
	return s.ForwardBatchWithArena(x, batchSize, nil, nil)
}

func (s *SequenceUnroller) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) ([]float32, error) {
	if batchSize <= 1 {
		return s.ForwardWithArena(x, arena, offset)
	}
	inSize := s.InSize()
	outSize := s.OutSize()
	if len(s.outputBuf) < batchSize*outSize {
		s.outputBuf = make([]float32, batchSize*outSize)
	}
	for i := 0; i < batchSize; i++ {
		out, err := s.ForwardWithArena(x[i*inSize:(i+1)*inSize], arena, offset)
		if err != nil {
			return nil, err
		}
		copy(s.outputBuf[i*outSize:(i+1)*outSize], out)
	}
	return s.outputBuf[:batchSize*outSize], nil
}

func (s *SequenceUnroller) BackwardBatch(grad []float32, batchSize int) ([]float32, error) {
	if batchSize <= 1 {
		return s.Backward(grad)
	}
	inSize := s.InSize()
	outSize := s.OutSize()
	if len(s.gradInBuf) < batchSize*inSize {
		s.gradInBuf = make([]float32, batchSize*inSize)
	}
	for i := batchSize - 1; i >= 0; i-- {
		dx, err := s.Backward(grad[i*outSize : (i+1)*outSize])
		if err != nil {
			return nil, err
		}
		copy(s.gradInBuf[i*inSize:(i+1)*inSize], dx)
	}
	return s.gradInBuf[:batchSize*inSize], nil
}

func (s *SequenceUnroller) AccumulateBackwardBatch(grad []float32, batchSize int) ([]float32, error) {
	return s.BackwardBatch(grad, batchSize)
}

func (s *SequenceUnroller) Params() []float32             { return s.base.Params() }
func (s *SequenceUnroller) SetParams(p []float32)        { s.base.SetParams(p) }
func (s *SequenceUnroller) Gradients() []float32         { return s.base.Gradients() }
func (s *SequenceUnroller) SetGradients(g []float32)     { s.base.SetGradients(g) }
func (s *SequenceUnroller) SetDevice(device Device)      { s.base.SetDevice(device) }
func (s *SequenceUnroller) Reset()                       { s.base.Reset() }
func (s *SequenceUnroller) ClearGradients()              { s.base.ClearGradients() }
func (s *SequenceUnroller) Clone() Layer {
	return NewSequenceUnroller(s.base.Clone(), s.timeSteps, s.returnSeq)
}

func (s *SequenceUnroller) LightweightClone(params []float32, grads []float32) Layer {
	newS := &SequenceUnroller{
		base:        s.base.LightweightClone(params, grads),
		timeSteps:   s.timeSteps,
		returnSeq:   s.returnSeq,
		inputBuf:    make([]float32, len(s.inputBuf)),
		outputBuf:   make([]float32, len(s.outputBuf)),
		gradInBuf:   make([]float32, len(s.gradInBuf)),
		storedInput: make([][]float32, s.timeSteps),
		training:    s.training,
	}
	return newS
}

func (s *SequenceUnroller) InSize() int                  { return s.timeSteps * s.base.InSize() }
func (s *SequenceUnroller) OutSize() int {
	if s.returnSeq {
		return s.timeSteps * s.base.OutSize()
	}
	return s.base.OutSize()
}

// ArenaSize returns the number of float32 values needed in the activation arena.
// SequenceUnroller delegates to the base layer for each timestep.
func (s *SequenceUnroller) ArenaSize() int {
	return s.timeSteps * s.base.ArenaSize()
}

func (s *SequenceUnroller) AccumulateBackward(grad []float32) ([]float32, error) { return s.Backward(grad) }
