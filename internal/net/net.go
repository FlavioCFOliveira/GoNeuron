// Package net provides core neural network types.
package net

import (
	"encoding/gob"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// Network is a collection of layers that can be forwarded and backwarded.
type Network struct {
	layers []layer.Layer
	loss   loss.Loss
	opt    opt.Optimizer

	// Pre-allocated gradient buffer for training
	// This avoids allocations in the training loop
	lossGradBuf []float64
}

// New creates a new neural network with the given layers.
func New(layers []layer.Layer, loss loss.Loss, optimizer opt.Optimizer) *Network {
	return &Network{
		layers: layers,
		loss:   loss,
		opt:    optimizer,
	}
}

// Forward performs a forward pass through all layers.
func (n *Network) Forward(x []float64) []float64 {
	curr := x
	for i := range n.layers {
		curr = n.layers[i].Forward(curr)
	}
	return curr
}

// Backward performs a backward pass through all layers.
func (n *Network) Backward(grad []float64) []float64 {
	curr := grad
	for i := len(n.layers) - 1; i >= 0; i-- {
		curr = n.layers[i].Backward(curr)
	}
	return curr
}

// Step performs one optimization step using the stored optimizer.
// This computes updated parameters and sets them back to layers.
func (n *Network) Step() {
	for _, l := range n.layers {
		params := l.Params()
		gradients := l.Gradients()
		// Compute updated parameters
		updated := n.opt.Step(params, gradients)
		// Set updated parameters back to layer
		l.SetParams(updated)
	}
}

// Train performs a training step on a single sample.
// This is optimized for performance with minimal allocations.
func (n *Network) Train(x []float64, y []float64) float64 {
	// Forward pass
	yPred := n.Forward(x)

	// Compute loss
	l := n.loss.Forward(yPred, y)

	// Compute gradient of loss w.r.t. prediction
	// Use pre-allocated buffer if possible
	yPredLen := len(yPred)
	if cap(n.lossGradBuf) < yPredLen {
		n.lossGradBuf = make([]float64, yPredLen)
	}
	grad := n.lossGradBuf[:yPredLen]

	// Use BackwardInPlace if available to avoid allocation
	if backwardInPlace, ok := n.loss.(loss.BackwardInPlacer); ok {
		backwardInPlace.BackwardInPlace(yPred, y, grad)
	} else {
		grad = n.loss.Backward(yPred, y)
	}

	// Backward pass
	_ = n.Backward(grad)

	// Optimization step
	n.Step()

	return l
}

// TrainBatch performs training on a batch of samples.
// This is optimized for cache locality and reduced allocation overhead.
// For large batches, it uses parallel processing to maximize CPU utilization.
func (n *Network) TrainBatch(batchX [][]float64, batchY [][]float64) float64 {
	batchSize := len(batchX)
	if batchSize == 0 {
		return 0
	}

	// Use parallel processing for batches larger than 16 samples
	// Parallel overhead dominates for small batches
	const parallelThreshold = 16

	// Force sequential for now due to bug in parallel training
	return n.trainBatchSequential(batchX, batchY)

	// if batchSize >= parallelThreshold {
	// 	return n.trainBatchParallel(batchX, batchY)
	// }

	// return n.trainBatchSequential(batchX, batchY)
}

// trainBatchSequential performs training on a batch of samples sequentially.
// Gradients are accumulated and averaged over the batch.
func (n *Network) trainBatchSequential(batchX [][]float64, batchY [][]float64) float64 {
	batchSize := float64(len(batchX))
	var totalLoss float64

	// Get gradient buffers from all layers
	allGradients := make([][]float64, len(n.layers))
	for i, l := range n.layers {
		allGradients[i] = l.Gradients()
	}

	// Forward and backward pass for all samples
	// Accumulate gradients over the batch
	for i := 0; i < len(batchX); i++ {
		yPred := n.Forward(batchX[i])
		totalLoss += n.loss.Forward(yPred, batchY[i])

		// Compute loss gradient
		yPredLen := len(yPred)
		if cap(n.lossGradBuf) < yPredLen {
			n.lossGradBuf = make([]float64, yPredLen)
		}
		grad := n.lossGradBuf[:yPredLen]

		if backwardInPlace, ok := n.loss.(loss.BackwardInPlacer); ok {
			backwardInPlace.BackwardInPlace(yPred, batchY[i], grad)
		} else {
			grad = n.loss.Backward(yPred, batchY[i])
		}

		// Backward pass (accumulates gradients into layer gradient buffers)
		_ = n.Backward(grad)
	}

	// Average gradients by dividing each by batch size
	// Apply to each layer's gradient buffer in-place
	for _, grads := range allGradients {
		for i := range grads {
			grads[i] /= batchSize
		}
	}

	// Single optimization step (averaged over batch)
	n.Step()
	return totalLoss / batchSize
}

// trainBatchParallel performs training on a batch of samples using parallel processing.
// Each sample's forward/backward pass is computed in parallel, then gradients are averaged.
func (n *Network) trainBatchParallel(batchX [][]float64, batchY [][]float64) float64 {
	batchSize := len(batchX)

	// Get number of workers (CPU cores)
	numWorkers := min(batchSize, runtime.NumCPU())

	// Pre-allocate gradient buffers for each sample
	// We need to copy gradients after each sample's backward pass
	// because the layer gradient buffers will be overwritten
	totalParams := len(n.Params())
	sampleGradients := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		sampleGradients[i] = make([]float64, totalParams)
	}

	losses := make([]float64, batchSize)

	// Worker function - process samples in parallel
	var wg sync.WaitGroup
	worker := func(start, end int) {
		defer wg.Done()
		for i := start; i < end; i++ {
			// Forward pass
			yPred := n.Forward(batchX[i])

			// Compute loss
			losses[i] = n.loss.Forward(yPred, batchY[i])

			// Compute loss gradient
			yPredLen := len(yPred)
			if cap(n.lossGradBuf) < yPredLen {
				n.lossGradBuf = make([]float64, yPredLen)
			}
			grad := n.lossGradBuf[:yPredLen]

			if backwardInPlace, ok := n.loss.(loss.BackwardInPlacer); ok {
				backwardInPlace.BackwardInPlace(yPred, batchY[i], grad)
			} else {
				grad = n.loss.Backward(yPred, batchY[i])
			}

			// Backward pass
			_ = n.Backward(grad)

			// Save a copy of gradients for this sample
			// Use the pre-allocated buffer instead of allocating
			copy(sampleGradients[i], n.Gradients())
		}
	}

	// Distribute work across workers
	chunkSize := (batchSize + numWorkers - 1) / numWorkers
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := min(start+chunkSize, batchSize)
		if start < end {
			wg.Add(1)
			go worker(start, end)
		}
	}

	wg.Wait()

	// Accumulate gradients from all samples
	totalLoss := 0.0
	for i := 0; i < batchSize; i++ {
		totalLoss += losses[i]
	}

	// Average gradients by dividing each by batch size
	if batchSize > 0 && len(sampleGradients) > 0 {
		avgGrads := sampleGradients[0] // Reuse first buffer as accumulator

		// Sum all gradients into avgGrads
		for i := 1; i < batchSize; i++ {
			for j := range avgGrads {
				avgGrads[j] += sampleGradients[i][j]
			}
		}

		// Average and apply to each layer
		offset := 0
		for _, l := range n.layers {
			gradLen := len(l.Gradients())
			layerGrads := avgGrads[offset : offset+gradLen]

			// Average gradients in-place
			for i := 0; i < gradLen; i++ {
				layerGrads[i] /= float64(batchSize)
			}

			// Apply gradient step in-place to layer's gradients buffer
			// Then use Step to get updated params and set them back
			params := l.Params()
			updated := n.opt.Step(params, layerGrads)
			l.SetParams(updated)

			offset += gradLen
		}
	}

	return totalLoss / float64(batchSize)
}

// Params returns all network parameters flattened (copy).
func (n *Network) Params() []float64 {
	var params []float64
	for _, l := range n.layers {
		params = append(params, l.Params()...)
	}
	return params
}

// Gradients returns all network gradients flattened (copy).
func (n *Network) Gradients() []float64 {
	var gradients []float64
	for _, l := range n.layers {
		gradients = append(gradients, l.Gradients()...)
	}
	return gradients
}

// Layers returns the network's layers slice.
func (n *Network) Layers() []layer.Layer {
	return n.layers
}

// Save saves the network to a file using gob encoding.
// The optimizer state is not saved (will use default).
func (n *Network) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	return n.Encode(file)
}

// Load loads a network from a file.
// Returns the loaded network and the loss type (for reconstruction).
func Load(filename string) (*Network, loss.Loss, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	// Read number of layers
	var numLayers int32
	if err := decoder.Decode(&numLayers); err != nil {
		return nil, nil, fmt.Errorf("failed to read layer count: %w", err)
	}

	// Read loss type
	var lossType string
	if err := decoder.Decode(&lossType); err != nil {
		return nil, nil, fmt.Errorf("failed to read loss type: %w", err)
	}

	// Read optimizer type
	var optType string
	if err := decoder.Decode(&optType); err != nil {
		return nil, nil, fmt.Errorf("failed to read optimizer type: %w", err)
	}

	// Read layer configurations
	layerConfigs := make([]LayerConfig, numLayers)
	for i := 0; i < int(numLayers); i++ {
		var cfg LayerConfig
		if err := decoder.Decode(&cfg); err != nil {
			return nil, nil, fmt.Errorf("failed to read layer %d: %w", i, err)
		}
		layerConfigs[i] = cfg
	}

	// Read parameters for all layers
	var params []float64
	if err := decoder.Decode(&params); err != nil {
		return nil, nil, fmt.Errorf("failed to read parameters: %w", err)
	}

	// Reconstruct layers
	var layers []layer.Layer
	offset := 0
	for _, cfg := range layerConfigs {
		layer, err := cfg.CreateLayer()
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create layer: %w", err)
		}
		numParams := len(cfg.Params)
		layer.SetParams(params[offset : offset+numParams])
		layers = append(layers, layer)
		offset += numParams
	}

	// Create network with default optimizer
	var optimizer opt.Optimizer
	switch optType {
	case "SGD":
		optimizer = &opt.SGD{LearningRate: 0.1}
	case "Adam":
		optimizer = opt.NewAdam(0.001)
	default:
		optimizer = &opt.SGD{LearningRate: 0.1}
	}

	// Create network with reconstructed loss
	var l loss.Loss
	switch lossType {
	case "MSE":
		l = loss.MSE{}
	case "CrossEntropy":
		l = loss.CrossEntropy{}
	case "Huber":
		l = loss.NewHuber(1.0)
	default:
		l = loss.MSE{}
	}

	return New(layers, l, optimizer), l, nil
}

// Encode writes the network to an io.Writer using gob encoding.
func (n *Network) Encode(w io.Writer) error {
	encoder := gob.NewEncoder(w)

	// Write number of layers
	if err := encoder.Encode(int32(len(n.layers))); err != nil {
		return fmt.Errorf("failed to encode layer count: %w", err)
	}

	// Write loss type
	var lossType string
	switch n.loss.(type) {
	case loss.MSE:
		lossType = "MSE"
	case loss.CrossEntropy:
		lossType = "CrossEntropy"
	case *loss.Huber:
		lossType = "Huber"
	default:
		lossType = "MSE"
	}
	if err := encoder.Encode(lossType); err != nil {
		return fmt.Errorf("failed to encode loss: %w", err)
	}

	// Write optimizer type
	var optType string
	switch n.opt.(type) {
	case *opt.SGD:
		optType = "SGD"
	case *opt.Adam:
		optType = "Adam"
	default:
		optType = "SGD"
	}
	if err := encoder.Encode(optType); err != nil {
		return fmt.Errorf("failed to encode optimizer: %w", err)
	}

	// Write layer configurations
	for _, l := range n.layers {
		cfg := ExtractLayerConfig(l)
		if err := encoder.Encode(cfg); err != nil {
			return fmt.Errorf("failed to encode layer: %w", err)
		}
	}

	// Write all parameters
	params := n.Params()
	if err := encoder.Encode(params); err != nil {
		return fmt.Errorf("failed to encode params: %w", err)
	}

	return nil
}

// LayerConfig holds the configuration needed to reconstruct a layer.
type LayerConfig struct {
	Type    string
	InSize  int
	OutSize int
	Params  []float64
	// Activation type for Dense layers
	Activation string
}

// ExtractLayerConfig extracts the configuration from a layer.
func ExtractLayerConfig(l layer.Layer) LayerConfig {
	cfg := LayerConfig{
		InSize:  -1,
		OutSize: -1,
		Params:  l.Params(),
	}

	// For Dense layers, we can extract more information
	if dense, ok := l.(*layer.Dense); ok {
		cfg.Type = "Dense"
		cfg.InSize = dense.InSize()
		cfg.OutSize = dense.OutSize()
		cfg.Activation = getActivationType(dense.Activation())
		cfg.Params = dense.Params()
	}

	// For LSTM layers
	if lstm, ok := l.(*layer.LSTM); ok {
		cfg.Type = "LSTM"
		cfg.InSize = lstm.InSize()
		cfg.OutSize = lstm.OutSize()
		cfg.Params = lstm.Params()
	}

	return cfg
}

// CreateLayer creates a new layer from the configuration.
func (c *LayerConfig) CreateLayer() (layer.Layer, error) {
	if c.Type == "LSTM" {
		lstm := layer.NewLSTM(c.InSize, c.OutSize)
		lstm.SetParams(c.Params)
		return lstm, nil
	}

	if c.Type != "Dense" {
		return nil, fmt.Errorf("unsupported layer type: %s", c.Type)
	}

	var act activations.Activation
	switch c.Activation {
	case "ReLU":
		act = activations.ReLU{}
	case "Sigmoid":
		act = activations.Sigmoid{}
	case "Tanh":
		act = activations.Tanh{}
	case "LeakyReLU":
		act = activations.NewLeakyReLU(0.01)
	default:
		act = activations.ReLU{}
	}

	dense := layer.NewDense(c.InSize, c.OutSize, act)
	dense.SetParams(c.Params)
	return dense, nil
}

// Helper functions for activation type detection
func getActivationType(act activations.Activation) string {
	switch act.(type) {
	case activations.ReLU:
		return "ReLU"
	case activations.Sigmoid:
		return "Sigmoid"
	case activations.Tanh:
		return "Tanh"
	case *activations.LeakyReLU:
		return "LeakyReLU"
	default:
		return "ReLU"
	}
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
