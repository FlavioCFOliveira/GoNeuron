// Package net provides core neural network types.
package net

import (
	"encoding/gob"
	"encoding/json"
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

func init() {
	// Register types for gob encoding/decoding
	// This is required for encoding/decoding interface values containing these types
	gob.Register([]float64{})
	gob.Register([][]float64{})
	gob.Register(activations.ReLU{})
	gob.Register(activations.Sigmoid{})
	gob.Register(activations.Tanh{})
	gob.Register(loss.MSE{})
	gob.Register(&loss.Huber{})
	gob.Register(&opt.SGD{})
	gob.Register(&opt.Adam{})
}

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

// Clone creates a deep copy of the network.
func (n *Network) Clone() *Network {
	newLayers := make([]layer.Layer, len(n.layers))
	for i, l := range n.layers {
		newLayers[i] = l.Clone()
	}

	return &Network{
		layers: newLayers,
		loss:   n.loss, // Loss functions are usually stateless
		opt:    n.opt,  // Note: Optimizer is shared, but we only use it in Step() which is sequential
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
	// Signal a new optimization step (increment timestep for Adam)
	n.opt.NewStep()

	for _, l := range n.layers {
		params := l.Params()
		gradients := l.Gradients()
		// Use StepInPlace for efficiency
		n.opt.StepInPlace(params, gradients)
		// No need for SetParams as StepInPlace modified params in-place
		// unless the layer doesn't return a reference to its internal params slice.
		// Most layers currently return a copy in Params().
		// Let's ensure consistency by calling SetParams with the modified slice.
		l.SetParams(params)
	}
}

// Train performs a training step on a single sample.
// This is optimized for performance with minimal allocations.
func (n *Network) Train(x []float64, y []float64) float64 {
	// Clear gradients before starting
	for _, l := range n.layers {
		l.ClearGradients()
	}

	// Reset internal state for layers that support it (e.g., LSTM, GRU)
	// This is critical to prevent state from one sample affecting another
	for _, l := range n.layers {
		if resettable, ok := l.(interface{ Reset() }); ok {
			resettable.Reset()
		}
	}

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

// Fit trains the network on the given dataset for a specified number of epochs.
func (n *Network) Fit(trainX, trainY [][]float64, epochs int, batchSize int, callbacks ...Callback) {
	for _, cb := range callbacks {
		cb.OnTrainBegin(n)
	}

	numSamples := len(trainX)
	numBatches := (numSamples + batchSize - 1) / batchSize

	for epoch := 0; epoch < epochs; epoch++ {
		for _, cb := range callbacks {
			cb.OnEpochBegin(epoch, n)
		}

		totalLoss := 0.0

		for b := 0; b < numBatches; b++ {
			for _, cb := range callbacks {
				cb.OnBatchBegin(b, n)
			}

			start := b * batchSize
			end := (b + 1) * batchSize
			if end > numSamples {
				end = numSamples
			}

			batchX := trainX[start:end]
			batchY := trainY[start:end]

			loss := n.TrainBatch(batchX, batchY)
			totalLoss += loss

			for _, cb := range callbacks {
				cb.OnBatchEnd(b, loss, n)
			}
		}

		avgLoss := totalLoss / float64(numBatches)

		stop := false
		for _, cb := range callbacks {
			cb.OnEpochEnd(epoch, avgLoss, n)
			if es, ok := cb.(*EarlyStopping); ok && es.Stopped {
				stop = true
			}
		}

		if stop {
			break
		}
	}

	for _, cb := range callbacks {
		cb.OnTrainEnd(n)
	}
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

	if batchSize >= parallelThreshold {
		return n.trainBatchParallel(batchX, batchY)
	}

	return n.trainBatchSequential(batchX, batchY)
}

// ForwardBatch performs forward pass on a batch of samples.
func (n *Network) ForwardBatch(batchX [][]float64) [][]float64 {
	results := make([][]float64, len(batchX))
	for i, x := range batchX {
		// Reset internal state for layers that support it (e.g., LSTM, GRU)
		for _, l := range n.layers {
			if resettable, ok := l.(interface{ Reset() }); ok {
				resettable.Reset()
			}
		}
		pred := n.Forward(x)
		results[i] = make([]float64, len(pred))
		copy(results[i], pred)
	}
	return results
}

// TrainTriplet performs a training step using Triplet Margin Loss.
// It performs three forward passes (anchor, positive, negative) and accumulates gradients.
func (n *Network) TrainTriplet(anchor, positive, negative []float64, margin float64) float64 {
	// Clear gradients at the start
	n.ClearGradients()

	// Reset internal state for layers that support it (e.g., LSTM, GRU)
	for _, l := range n.layers {
		if resettable, ok := l.(interface{ Reset() }); ok {
			resettable.Reset()
		}
	}

	// 1. Forward passes
	aPred := n.Forward(anchor)
	pPred := n.Forward(positive)
	nPred := n.Forward(negative)

	// Concatenate predictions for TripletMarginLoss
	embDim := len(aPred)
	tripletPred := make([]float64, embDim*3)
	copy(tripletPred[0:embDim], aPred)
	copy(tripletPred[embDim:2*embDim], pPred)
	copy(tripletPred[2*embDim:3*embDim], nPred)

	// 2. Compute loss
	// Note: yTrue is not used for TripletMarginLoss, but must have same length as tripletPred
	dummyY := make([]float64, len(tripletPred))
	lVal := n.loss.Forward(tripletPred, dummyY)

	// 3. Compute loss gradient
	if cap(n.lossGradBuf) < len(tripletPred) {
		n.lossGradBuf = make([]float64, len(tripletPred))
	}
	grad := n.lossGradBuf[:len(tripletPred)]

	if backwardInPlace, ok := n.loss.(loss.BackwardInPlacer); ok {
		backwardInPlace.BackwardInPlace(tripletPred, dummyY, grad)
	} else {
		grad = n.loss.Backward(tripletPred, dummyY)
	}

	// 4. Backward passes (accumulating gradients)
	// The order of Backward calls matters for layers with saved state (like Dense with savedInputs)
	// We need to pass the corresponding part of the loss gradient to each pass.
	// Since we use AccumulateBackward, we must be careful about how layers handle multiple passes.

	// Anchor gradient
	_ = n.accumulateBackward(grad[0:embDim])
	// Positive gradient
	_ = n.accumulateBackward(grad[embDim : 2*embDim])
	// Negative gradient
	_ = n.accumulateBackward(grad[2*embDim : 3*embDim])

	// Optimization step
	n.Step()

	return lVal
}

// SetParams sets all network parameters from a flattened slice.
func (n *Network) SetParams(params []float64) {
	offset := 0
	for _, l := range n.layers {
		lLen := len(l.Params())
		l.SetParams(params[offset : offset+lLen])
		offset += lLen
	}
}

// accumulateBackward performs a backward pass through all layers and accumulates gradients.
func (n *Network) accumulateBackward(grad []float64) []float64 {
	curr := grad
	for i := len(n.layers) - 1; i >= 0; i-- {
		curr = n.layers[i].AccumulateBackward(curr)
	}
	return curr
}

// trainBatchParallel performs training on a batch of samples using parallel processing.
// Each worker gets its own Network copy to avoid buffer sharing issues.
func (n *Network) trainBatchParallel(batchX [][]float64, batchY [][]float64) float64 {
	batchSize := len(batchX)
	numWorkers := min(batchSize, runtime.NumCPU())

	// Create network copies for each worker
	workers := make([]*Network, numWorkers)
	for i := 0; i < numWorkers; i++ {
		workers[i] = n.Clone()
		workers[i].ClearGradients()
	}

	losses := make([]float64, batchSize)
	var wg sync.WaitGroup

	// Distribute work across workers
	chunkSize := (batchSize + numWorkers - 1) / numWorkers
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := min(start+chunkSize, batchSize)
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(workerIdx, s, e int) {
			defer wg.Done()
			netCopy := workers[workerIdx]

			for i := s; i < e; i++ {
				// Reset internal state for layers that support it (e.g., LSTM, GRU)
				// This is critical to prevent state from one sample affecting another
				for _, l := range netCopy.layers {
					if resettable, ok := l.(interface{ Reset() }); ok {
						resettable.Reset()
					}
				}

				// Forward pass
				yPred := netCopy.Forward(batchX[i])

				// Compute loss
				losses[i] = netCopy.loss.Forward(yPred, batchY[i])

				// Compute loss gradient
				yPredLen := len(yPred)
				grad := make([]float64, yPredLen)

				if backwardInPlace, ok := netCopy.loss.(loss.BackwardInPlacer); ok {
					backwardInPlace.BackwardInPlace(yPred, batchY[i], grad)
				} else {
					grad = netCopy.loss.Backward(yPred, batchY[i])
				}

				// Backward pass (accumulates into netCopy's layer buffers)
				_ = netCopy.accumulateBackward(grad)
			}
		}(w, start, end)
	}

	wg.Wait()

	// Sum all gradients from workers into a single buffer
	totalParams := len(n.Params())
	sumGrads := make([]float64, totalParams)

	for _, w := range workers {
		wGrads := w.Gradients()
		for i := range sumGrads {
			sumGrads[i] += wGrads[i]
		}
	}

	// Average gradients
	invBatchSize := 1.0 / float64(batchSize)
	for i := range sumGrads {
		sumGrads[i] *= invBatchSize
	}

	// Set averaged gradients back to original network
	n.SetGradients(sumGrads)

	// Optimization step
	n.Step()

	// Compute total loss
	totalLoss := 0.0
	for _, l := range losses {
		totalLoss += l
	}
	return totalLoss / float64(batchSize)
}

// trainBatchSequential performs training on a batch of samples sequentially.
// Gradients are accumulated and averaged over the batch.
func (n *Network) trainBatchSequential(batchX [][]float64, batchY [][]float64) float64 {
	batchSize := float64(len(batchX))
	var totalLoss float64

	// Clear gradients at the start of the batch
	n.ClearGradients()

	// Forward and backward pass for all samples
	// Accumulate gradients over the batch
	for i := 0; i < len(batchX); i++ {
		// Reset internal state for layers that support it (e.g., LSTM, GRU)
		// This is critical to prevent state from one sample affecting another
		for _, l := range n.layers {
			if resettable, ok := l.(interface{ Reset() }); ok {
				resettable.Reset()
			}
		}

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
	// Apply directly to layer gradient buffers (which were accumulated during backward)
	for _, l := range n.layers {
		grads := l.Gradients() // Get current accumulated gradients
		for j := range grads {
			grads[j] /= batchSize
		}
		l.SetGradients(grads) // Set averaged gradients back
	}

	// Single optimization step (averaged over batch)
	n.Step()
	return totalLoss / batchSize
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

// ClearGradients zeroes out all layer gradients.
func (n *Network) ClearGradients() {
	for _, l := range n.layers {
		l.ClearGradients()
	}
}

// SetGradients sets network gradients from a flattened slice.
func (n *Network) SetGradients(gradients []float64) {
	offset := 0
	for _, l := range n.layers {
		lGrads := l.Gradients()
		gradLen := len(lGrads)
		l.SetGradients(gradients[offset : offset+gradLen])
		offset += gradLen
	}
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
		l, err := cfg.CreateLayer()
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create layer: %w", err)
		}
		numParams := len(cfg.Params)
		l.SetParams(params[offset : offset+numParams])
		layers = append(layers, l)
		offset += numParams
	}

	// Read optimizer state (gob-compatible format)
	var optGobState any
	if err := decoder.Decode(&optGobState); err != nil {
		// Older versions might not have optimizer state
		optGobState = nil
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

	if optGobState != nil {
		optimizer.SetGobState(optGobState)
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

	// Write optimizer state using gob-compatible format
	optGobState := n.opt.GobState()
	if err := encoder.Encode(optGobState); err != nil {
		return fmt.Errorf("failed to encode optimizer state: %w", err)
	}

	return nil
}

// SaveJSON saves the network to a file using JSON encoding.
func (n *Network) SaveJSON(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	data := struct {
		Layers    []LayerConfig
		Loss      string
		Optimizer string
		OptState  map[string]any
	}{
		Layers:    make([]LayerConfig, len(n.layers)),
		Loss:      fmt.Sprintf("%T", n.loss),
		Optimizer: fmt.Sprintf("%T", n.opt),
		OptState:  n.opt.State(),
	}

	for i, l := range n.layers {
		data.Layers[i] = ExtractLayerConfig(l)
	}

	return encoder.Encode(data)
}

// LayerConfig holds the configuration needed to reconstruct a layer.
type LayerConfig struct {
	Type    string
	InSize  int
	OutSize int
	Params  []float64

	// Activation type
	Activation string
	Alpha      float64 // For LeakyReLU, PReLU, ELU

	// Conv2D & Pooling parameters
	KernelSize  int
	Stride      int
	Padding     int
	InChannels  int
	OutChannels int

	// Dropout parameters
	Prob float64

	// Normalization parameters
	Eps             float64
	Affine          bool
	NormalizedShape int

	// RBF parameters
	NumCenters int
	Gamma      float64
}

// ExtractLayerConfig extracts the configuration from a layer.
func ExtractLayerConfig(l layer.Layer) LayerConfig {
	cfg := LayerConfig{
		InSize:  -1,
		OutSize: -1,
		Params:  l.Params(),
	}

	// Identify layer type and extract parameters
	switch v := l.(type) {
	case *layer.Dense:
		cfg.Type = "Dense"
		cfg.InSize = v.InSize()
		cfg.OutSize = v.OutSize()
		cfg.Activation = getActivationType(v.Activation())
		if prelu, ok := v.Activation().(*activations.PReLU); ok {
			cfg.Alpha = prelu.Alpha
		} else if leaky, ok := v.Activation().(*activations.LeakyReLU); ok {
			cfg.Alpha = leaky.Alpha
		} else if elu, ok := v.Activation().(activations.ELU); ok {
			cfg.Alpha = elu.Alpha
		} else if eluPtr, ok := v.Activation().(*activations.ELU); ok {
			cfg.Alpha = eluPtr.Alpha
		}
	case *layer.Conv2D:
		cfg.Type = "Conv2D"
		cfg.InChannels = v.InSize() // InSize for Conv2D returns inChannels
		cfg.OutChannels = v.OutSize()
		cfg.KernelSize = v.GetKernelSize()
		cfg.Stride = v.GetStride()
		cfg.Padding = v.GetPadding()
		cfg.Activation = getActivationType(v.GetActivation())
	case *layer.LSTM:
		cfg.Type = "LSTM"
		cfg.InSize = v.InSize()
		cfg.OutSize = v.OutSize()
	case *layer.GRU:
		cfg.Type = "GRU"
		cfg.InSize = v.InSize()
		cfg.OutSize = v.OutSize()
	case *layer.Dropout:
		cfg.Type = "Dropout"
		cfg.Prob = v.GetP()
		cfg.InSize = v.InSize()
	case *layer.LayerNorm:
		cfg.Type = "LayerNorm"
		cfg.NormalizedShape = v.InSize()
		cfg.Eps = v.GetEps()
		cfg.Affine = v.GetGamma() != nil
	case *layer.BatchNorm2D:
		cfg.Type = "BatchNorm2D"
		cfg.InChannels = v.InSize()
		cfg.Eps = v.GetEps()
		cfg.Affine = v.GetGamma() != nil
	case *layer.MaxPool2D:
		cfg.Type = "MaxPool2D"
		cfg.InChannels = v.InSize() / (v.GetKernelSize() * v.GetKernelSize()) // Approximate
		cfg.KernelSize = v.GetKernelSize()
		cfg.Stride = v.GetStride()
		cfg.Padding = v.GetPadding()
	case *layer.AvgPool2D:
		cfg.Type = "AvgPool2D"
		cfg.KernelSize = v.GetKernelSize()
		cfg.Stride = v.GetStride()
		cfg.Padding = v.GetPadding()
	case *layer.Embedding:
		cfg.Type = "Embedding"
		cfg.InSize = v.InSize() // num_embeddings
		cfg.OutSize = v.OutSize() // embedding_dim
	case *layer.Flatten:
		cfg.Type = "Flatten"
	case *layer.RBF:
		cfg.Type = "RBF"
		cfg.InSize = v.InSize()
		cfg.OutSize = v.OutSize()
		cfg.NumCenters = v.NumCenters()
		cfg.Gamma = v.Gamma()
	default:
		cfg.Type = "Unknown"
	}

	return cfg
}

// CreateLayer creates a new layer from the configuration.
func (c *LayerConfig) CreateLayer() (layer.Layer, error) {
	var l layer.Layer
	var err error

	switch c.Type {
	case "Dense":
		act := c.createActivation()
		l = layer.NewDense(c.InSize, c.OutSize, act)
	case "Conv2D":
		act := c.createActivation()
		l = layer.NewConv2D(c.InChannels, c.OutChannels, c.KernelSize, c.Stride, c.Padding, act)
	case "LSTM":
		l = layer.NewLSTM(c.InSize, c.OutSize)
	case "GRU":
		l = layer.NewGRU(c.InSize, c.OutSize)
	case "Dropout":
		l = layer.NewDropout(c.Prob, c.InSize)
	case "LayerNorm":
		l = layer.NewLayerNorm(c.NormalizedShape, c.Eps, c.Affine)
	case "BatchNorm2D":
		l = layer.NewBatchNorm2D(c.InChannels, c.Eps, 0.1, c.Affine)
	case "MaxPool2D":
		l = layer.NewMaxPool2D(c.InChannels, c.KernelSize, c.Stride, c.Padding)
	case "AvgPool2D":
		// Note: AvgPool2D currently assumes 1 channel or handles internally
		l = layer.NewAvgPool2D(c.KernelSize, c.Stride, c.Padding)
	case "Embedding":
		l = layer.NewEmbedding(c.InSize, c.OutSize)
	case "Flatten":
		l = layer.NewFlatten()
	case "RBF":
		l = layer.NewRBF(c.InSize, c.NumCenters, c.OutSize, c.Gamma)
	default:
		return nil, fmt.Errorf("unsupported layer type: %s", c.Type)
	}

	if l != nil {
		l.SetParams(c.Params)
	}

	return l, err
}

func (c *LayerConfig) createActivation() activations.Activation {
	switch c.Activation {
	case "ReLU":
		return activations.ReLU{}
	case "Sigmoid":
		return activations.Sigmoid{}
	case "Tanh":
		return activations.Tanh{}
	case "LeakyReLU":
		return activations.NewLeakyReLU(c.Alpha)
	case "PReLU":
		return activations.NewPReLU(c.Alpha)
	case "ELU":
		return activations.NewELU(c.Alpha)
	case "GELU":
		return activations.GELU{}
	case "SiLU":
		return activations.SiLU{}
	case "SELU":
		return activations.SELU{}
	case "Softmax":
		return activations.Softmax{}
	case "LogSoftmax":
		return activations.LogSoftmax{}
	case "Linear":
		return activations.Linear{}
	default:
		return activations.ReLU{}
	}
}

// Helper functions for activation type detection
func getActivationType(act activations.Activation) string {
	switch act.(type) {
	case activations.ReLU, *activations.ReLU:
		return "ReLU"
	case activations.Sigmoid, *activations.Sigmoid:
		return "Sigmoid"
	case activations.Tanh, *activations.Tanh:
		return "Tanh"
	case *activations.LeakyReLU:
		return "LeakyReLU"
	case *activations.PReLU:
		return "PReLU"
	case activations.ELU, *activations.ELU:
		return "ELU"
	case activations.GELU, *activations.GELU:
		return "GELU"
	case activations.SiLU, *activations.SiLU:
		return "SiLU"
	case activations.SELU, *activations.SELU:
		return "SELU"
	case activations.Softmax, *activations.Softmax:
		return "Softmax"
	case activations.LogSoftmax, *activations.LogSoftmax:
		return "LogSoftmax"
	case activations.Linear, *activations.Linear:
		return "Linear"
	default:
		return "ReLU"
	}
}
