package net

import (
	"fmt"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// Sequential is a high-level wrapper around Network to provide a Keras-like API.
type Sequential struct {
	*Network
}

// NewSequential creates a new Sequential model.
func NewSequential(layers ...layer.Layer) *Sequential {
	return &Sequential{
		Network: &Network{
			layers: layers,
		},
	}
}

// Compile configures the model for training.
func (s *Sequential) Compile(optimizer opt.Optimizer, lossFn loss.Loss) {
	s.opt = optimizer
	s.loss = lossFn
}

// Predict performs a forward pass and returns the output.
func (s *Sequential) Predict(x []float32) []float32 {
	return s.Forward(x)
}

// PredictBatch performs forward pass on a batch of samples.
func (s *Sequential) PredictBatch(x [][]float32) [][]float32 {
	return s.ForwardBatch(x)
}

// SetTraining sets the training mode for the model.
func (s *Sequential) SetTraining(training bool) {
	s.Network.SetTraining(training)
}

// Evaluate calculates the average loss on a dataset.
func (s *Sequential) Evaluate(x, y [][]float32) float32 {
	totalLoss := float32(0)
	for i := range x {
		pred := s.Forward(x[i])
		totalLoss += s.loss.Forward(pred, y[i])
	}
	return totalLoss / float32(len(x))
}

// Summary prints a summary of the network architecture.
func (s *Sequential) Summary() {
	fmt.Println("Model: Sequential")
	fmt.Println("_________________________________________________________________")
	fmt.Printf("%-25s %-20s %-10s\n", "Layer (type)", "Output Shape", "Param #")
	fmt.Println("=================================================================")

	totalParams := 0
	for i, l := range s.layers {
		lType := fmt.Sprintf("%T", l)
		// Extract simple type name
		for j := len(lType) - 1; j >= 0; j-- {
			if lType[j] == '.' {
				lType = lType[j+1:]
				break
			}
		}

		outShape := fmt.Sprintf("(%d)", l.OutSize())
		params := len(l.Params())
		totalParams += params

		fmt.Printf("%-25s %-20s %-10d\n", fmt.Sprintf("%s_%d", lType, i), outShape, params)
	}
	fmt.Println("=================================================================")
	fmt.Printf("Total params: %d\n", totalParams)
	fmt.Println("_________________________________________________________________")
}
