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

	// If first layer has size, we can build immediately
	if len(s.layers) > 0 && s.layers[0].InSize() > 0 {
		s.Build(s.layers[0].InSize())
	}
}

// Build performs shape inference and initializes buffers.
func (s *Sequential) Build(firstInSize int) {
	lastOutSize := firstInSize
	var lastH, lastW int

	// If the first layer is spatial, we might need its dimensions.
	// Users should use SetInputDimensions on the first layer if it's a Conv2D.
	if cl, ok := s.layers[0].(*layer.Conv2D); ok {
		lastH, lastW = cl.GetOutputDimensions() // This gets input dims if not yet run? No.
		// Actually Conv2D.inputHeight/Width are used.
	}

	for _, l := range s.layers {
		buildSize := lastOutSize
		if lastH > 0 && lastW > 0 {
			// Propagate spatial dimensions if available
			if sl, ok := l.(interface{ SetInputDimensions(int, int) }); ok {
				sl.SetInputDimensions(lastH, lastW)
			}

			// If the layer expects channel count as InSize but we have spatial info,
			// we need to divide the flat size by spatial area to get channels.
			switch l.(type) {
			case *layer.Conv2D, *layer.BatchNorm2D, *layer.MaxPool2D:
				buildSize = lastOutSize / (lastH * lastW)
			}
		}

		if dl, ok := l.(layer.DeferredLayer); ok && l.InSize() <= 0 {
			dl.Build(buildSize)
		}
		lastOutSize = l.OutSize()

		// Update spatial dimensions for next layer
		if sl, ok := l.(interface{ GetOutputDimensions() (int, int) }); ok {
			lastH, lastW = sl.GetOutputDimensions()
		} else if _, ok := l.(*layer.Flatten); ok {
			lastH, lastW = 0, 0 // Flatten destroys spatial info
		}
	}

	s.initBuffers()
}

func (s *Sequential) checkDeferredBuild(x []float32) {
	if len(s.params) == 0 && len(s.layers) > 0 && s.layers[0].InSize() <= 0 {
		s.Build(len(x))
	}
}

// Forward performs a forward pass.
func (s *Sequential) Forward(x []float32) []float32 {
	s.checkDeferredBuild(x)
	return s.Network.Forward(x)
}

// TrainBatch performs training on a batch.
func (s *Sequential) TrainBatch(batchX, batchY [][]float32) float32 {
	if len(batchX) > 0 {
		s.checkDeferredBuild(batchX[0])
	}
	return s.Network.TrainBatch(batchX, batchY)
}

// Fit trains the network.
func (s *Sequential) Fit(trainX, trainY [][]float32, epochs int, batchSize int, callbacks ...Callback) {
	if len(trainX) > 0 {
		s.checkDeferredBuild(trainX[0])
	}
	s.Network.Fit(trainX, trainY, epochs, batchSize, callbacks...)
}

// Predict performs a forward pass and returns the output.
func (s *Sequential) Predict(x []float32) []float32 {
	s.SetTraining(false)
	return s.Forward(x)
}

// PredictBatch performs forward pass on a batch of samples.
func (s *Sequential) PredictBatch(x [][]float32) [][]float32 {
	s.SetTraining(false)
	if len(x) > 0 {
		s.checkDeferredBuild(x[0])
	}
	return s.ForwardBatch(x)
}

// SetTraining sets the training mode for the model.
func (s *Sequential) SetTraining(training bool) {
	s.Network.SetTraining(training)
}

// Evaluate calculates the average loss on a dataset.
func (s *Sequential) Evaluate(x, y [][]float32) float32 {
	s.SetTraining(false)
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
