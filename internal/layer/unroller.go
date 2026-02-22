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
}

func NewSequenceUnroller(base Layer, timeSteps int, returnSeq bool) *SequenceUnroller {
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

func (s *SequenceUnroller) Forward(x []float32) []float32 {
	inSize := s.base.InSize()
	outSize := s.base.OutSize()

	if len(x) != s.timeSteps*inSize {
		panic(fmt.Sprintf("SequenceUnroller: input size mismatch. Expected %d, got %d", s.timeSteps*inSize, len(x)))
	}

	s.base.Reset()

	// Special case for Bidirectional: we need to populate its hidden states
	if bi, ok := s.base.(*Bidirectional); ok {
		for t := 0; t < s.timeSteps; t++ {
			if s.storedInput[t] == nil {
				s.storedInput[t] = make([]float32, inSize)
			}
			copy(s.storedInput[t], x[t*inSize:(t+1)*inSize])
			bi.Forward(s.storedInput[t])
		}
		bi.ComputeBackwardHiddenStates()

		for t := 0; t < s.timeSteps; t++ {
			fOut := bi.GetForwardOutputAt(t)
			bOut := bi.GetBackwardOutputAt(t)
			if s.returnSeq {
				copy(s.outputBuf[t*s.OutSize()/s.timeSteps:t*s.OutSize()/s.timeSteps+len(fOut)], fOut)
				copy(s.outputBuf[t*s.OutSize()/s.timeSteps+len(fOut):(t+1)*s.OutSize()/s.timeSteps], bOut)
			} else if t == s.timeSteps-1 {
				copy(s.outputBuf[:len(fOut)], fOut)
				copy(s.outputBuf[len(fOut):], bOut)
			}
		}
	} else {
		for t := 0; t < s.timeSteps; t++ {
			if s.storedInput[t] == nil {
				s.storedInput[t] = make([]float32, inSize)
			}
			copy(s.storedInput[t], x[t*inSize:(t+1)*inSize])

			out := s.base.Forward(s.storedInput[t])

			if s.returnSeq {
				copy(s.outputBuf[t*outSize:(t+1)*outSize], out)
			} else if t == s.timeSteps-1 {
				copy(s.outputBuf, out)
			}
		}
	}

	return s.outputBuf
}

func (s *SequenceUnroller) Backward(grad []float32) []float32 {
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
			dx := bi.forward.Backward(currentGrad)
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
				// Wait, if returnSeq=false, we only returned the last step's hidden state.
				// For Bidirectional, the last hidden state [f(T-1), b(0)].
				// So b(0) has gradient.
				for i := 0; i < bOutSize; i++ {
					currentGrad[i] += grad[fOutSize+i]
				}
			}
			dx := bi.backward.Backward(currentGrad)
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
			dx := s.base.Backward(currentGrad)
			copy(s.gradInBuf[t*inSize:(t+1)*inSize], dx)
			copy(dhNext, currentGrad)
		}
	}

	return s.gradInBuf
}

func (s *SequenceUnroller) Params() []float32             { return s.base.Params() }
func (s *SequenceUnroller) SetParams(p []float32)        { s.base.SetParams(p) }
func (s *SequenceUnroller) Gradients() []float32         { return s.base.Gradients() }
func (s *SequenceUnroller) SetGradients(g []float32)     { s.base.SetGradients(g) }
func (s *SequenceUnroller) SetDevice(device Device)      { s.base.SetDevice(device) }
func (s *SequenceUnroller) Reset()                       { s.base.Reset() }
func (s *SequenceUnroller) ClearGradients()              { s.base.ClearGradients() }
func (s *SequenceUnroller) Clone() Layer                 { return NewSequenceUnroller(s.base.Clone(), s.timeSteps, s.returnSeq) }
func (s *SequenceUnroller) InSize() int                  { return s.timeSteps * s.base.InSize() }
func (s *SequenceUnroller) OutSize() int {
	if s.returnSeq {
		return s.timeSteps * s.base.OutSize()
	}
	return s.base.OutSize()
}
func (s *SequenceUnroller) AccumulateBackward(grad []float32) []float32 { return s.Backward(grad) }
