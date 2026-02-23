// Package net provides core neural network types.
package net

import (
	"encoding/binary"
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
	gob.Register([]float32{})
	gob.Register([][]float32{})
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

	// Contiguous global buffers for all parameters and gradients
	params []float32
	grads  []float32

	// Pre-allocated gradient buffer for training
	// This avoids allocations in the training loop
	lossGradBuf []float32

	// Activation Arena: contiguous buffer for all layers' intermediate states
	arena      []float32
	arenaIndex int

	// Worker Pool for parallel training
	workerPool   []*Network
	batchLossBuf []float32
}

// New creates a new neural network with the given layers.
func New(layers []layer.Layer, loss loss.Loss, optimizer opt.Optimizer) *Network {
	n := &Network{
		layers: layers,
		loss:   loss,
		opt:    optimizer,
	}
	n.initBuffers()
	return n
}

// initBuffers initializes the global contiguous buffers and re-binds layers.
func (n *Network) initBuffers() {
	totalSize := 0
	for _, l := range n.layers {
		totalSize += len(l.Params())
	}

	if totalSize == 0 {
		return
	}

	n.params = make([]float32, totalSize)
	n.grads = make([]float32, totalSize)

	offset := 0
	for _, l := range n.layers {
		lLen := len(l.Params())
		if lLen > 0 {
			// Copy existing params to the new global buffer
			copy(n.params[offset:offset+lLen], l.Params())
			// Re-bind layer to use a view of the global buffers
			l.SetParams(n.params[offset : offset+lLen])
			l.SetGradients(n.grads[offset : offset+lLen])
			offset += lLen
		}
	}

	// Invalidate worker pool if buffers are re-initialized
	n.workerPool = nil
}

func (n *Network) initWorkers(num int) {
	if len(n.workerPool) >= num {
		return
	}

	oldPool := n.workerPool
	n.workerPool = make([]*Network, num)
	copy(n.workerPool, oldPool)

	for i := len(oldPool); i < num; i++ {
		n.workerPool[i] = n.LightweightClone()
		// Give each worker its own private gradient buffer for parallel accumulation
		workerGrads := make([]float32, len(n.grads))
		n.workerPool[i].grads = workerGrads
		offset := 0
		for _, l := range n.workerPool[i].layers {
			lLen := len(l.Gradients())
			if lLen > 0 {
				l.SetGradients(workerGrads[offset : offset+lLen])
				offset += lLen
			}
		}
	}
}

// LightweightClone creates a network that shares the same parameters and gradients
// but has its own internal buffers and layer instances.
func (n *Network) LightweightClone() *Network {
	newLayers := make([]layer.Layer, len(n.layers))
	offset := 0
	gradOffset := 0
	for i, l := range n.layers {
		lLen := len(l.Params())
		lGradLen := len(l.Gradients())
		newLayers[i] = l.LightweightClone(n.params[offset:offset+lLen], n.grads[gradOffset:gradOffset+lGradLen])
		offset += lLen
		gradOffset += lGradLen
	}

	net := &Network{
		layers: newLayers,
		loss:   n.loss,
		opt:    n.opt,
		params: n.params,
		grads:  n.grads,
	}
	return net
}

// SetDevice sets the computation device for the entire network.
func (n *Network) SetDevice(device layer.Device) {
	for _, l := range n.layers {
		l.SetDevice(device)
	}
}

// SetTraining sets the training mode for all layers that support it.
func (n *Network) SetTraining(training bool) {
	for _, l := range n.layers {
		if t, ok := l.(interface{ SetTraining(bool) }); ok {
			t.SetTraining(training)
		}
	}
}

// Clone creates a deep copy of the network.
func (n *Network) Clone() *Network {
	newLayers := make([]layer.Layer, len(n.layers))
	for i, l := range n.layers {
		newLayers[i] = l.Clone()
	}

	net := &Network{
		layers: newLayers,
		loss:   n.loss, // Loss functions are usually stateless
		opt:    n.opt,  // Note: Optimizer is shared, but we only use it in Step() which is sequential
	}
	net.initBuffers()
	return net
}

// Forward performs a forward pass through all layers.
func (n *Network) Forward(x []float32) []float32 {
	// Reset arena index for forward pass if in training
	n.arenaIndex = 0

	curr := x
	for i := range n.layers {
		// Check if layer supports Arena saving
		if saver, ok := n.layers[i].(interface {
			ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32
		}); ok {
			curr = saver.ForwardWithArena(curr, &n.arena, &n.arenaIndex)
		} else {
			curr = n.layers[i].Forward(curr)
		}
	}
	return curr
}

// Predict performs a forward pass through all layers in inference mode.
func (n *Network) Predict(x []float32) []float32 {
	n.SetTraining(false)
	return n.Forward(x)
}

// Backward performs a backward pass through all layers.
func (n *Network) Backward(grad []float32) []float32 {
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

	params := n.Params()
	gradients := n.Gradients()

	if len(n.params) > 0 && len(n.grads) > 0 {
		// Optimization step on global buffer
		n.opt.StepInPlace(params, gradients)

		// Sync layers if they use Metal (params might be updated in-place)
		for _, l := range n.layers {
			l.SetParams(l.Params())
		}
	} else {
		// Fallback for non-contiguous networks
		for _, l := range n.layers {
			p := l.Params()
			g := l.Gradients()
			n.opt.StepInPlace(p, g)
			l.SetParams(p)
		}
	}
}

// Train performs a training step on a single sample.
// This is optimized for performance with minimal allocations.
func (n *Network) Train(x []float32, y []float32) float32 {
	n.SetTraining(true)
	// Clear gradients before starting
	n.ClearGradients()

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
		n.lossGradBuf = make([]float32, yPredLen)
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
func (n *Network) Fit(trainX, trainY [][]float32, epochs int, batchSize int, callbacks ...Callback) {
	n.SetTraining(true)
	for _, cb := range callbacks {
		cb.OnTrainBegin(n)
	}

	numSamples := len(trainX)
	numBatches := (numSamples + batchSize - 1) / batchSize

	for epoch := 0; epoch < epochs; epoch++ {
		for _, cb := range callbacks {
			cb.OnEpochBegin(epoch, n)
		}

		totalLoss := float32(0.0)

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

		avgLoss := totalLoss / float32(numBatches)

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
func (n *Network) TrainBatch(batchX [][]float32, batchY [][]float32) float32 {
	batchSize := len(batchX)
	if batchSize == 0 {
		return 0
	}

	// Use parallel processing for batches larger than 64 samples
	// Parallel overhead dominates for small batches
	const parallelThreshold = 64

	if batchSize >= parallelThreshold && runtime.NumCPU() > 1 {
		return n.trainBatchParallel(batchX, batchY)
	}

	return n.trainBatchSequential(batchX, batchY)
}

// ForwardBatch performs forward pass on a batch of samples.
func (n *Network) ForwardBatch(batchX [][]float32) [][]float32 {
	n.SetTraining(false)
	results := make([][]float32, len(batchX))
	for i, x := range batchX {
		// Reset internal state for layers that support it (e.g., LSTM, GRU)
		for _, l := range n.layers {
			if resettable, ok := l.(interface{ Reset() }); ok {
				resettable.Reset()
			}
		}
		pred := n.Forward(x)
		results[i] = make([]float32, len(pred))
		copy(results[i], pred)
	}
	return results
}

// TrainTriplet performs a training step using Triplet Margin Loss.
// It performs three forward passes (anchor, positive, negative) and accumulates gradients.
func (n *Network) TrainTriplet(anchor, positive, negative []float32, margin float32) float32 {
	n.SetTraining(true)
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
	tripletPred := make([]float32, embDim*3)
	copy(tripletPred[0:embDim], aPred)
	copy(tripletPred[embDim:2*embDim], pPred)
	copy(tripletPred[2*embDim:3*embDim], nPred)

	// 2. Compute loss
	// Note: yTrue is not used for TripletMarginLoss, but must have same length as tripletPred
	dummyY := make([]float32, len(tripletPred))
	lVal := n.loss.Forward(tripletPred, dummyY)

	// 3. Compute loss gradient
	if cap(n.lossGradBuf) < len(tripletPred) {
		n.lossGradBuf = make([]float32, len(tripletPred))
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
func (n *Network) SetParams(params []float32) {
	if len(params) == 0 {
		return
	}
	if len(n.params) > 0 && &n.params[0] == &params[0] {
		// Already using this buffer, just sync layers if needed
		return
	}
	if len(params) == len(n.params) {
		copy(n.params, params)
		// Layers are already views into n.params, but they might need to sync to GPU
		for _, l := range n.layers {
			l.SetParams(l.Params())
		}
	} else {
		offset := 0
		for _, l := range n.layers {
			lLen := len(l.Params())
			l.SetParams(params[offset : offset+lLen])
			offset += lLen
		}
	}
}

// accumulateBackward performs a backward pass through all layers and accumulates gradients.
func (n *Network) accumulateBackward(grad []float32) []float32 {
	curr := grad
	for i := len(n.layers) - 1; i >= 0; i-- {
		curr = n.layers[i].AccumulateBackward(curr)
	}
	return curr
}

// trainBatchParallel performs training on a batch of samples using parallel processing.
// It uses a pool of worker networks to avoid allocations and aggregates gradients sequentially.
func (n *Network) trainBatchParallel(batchX [][]float32, batchY [][]float32) float32 {
	batchSize := len(batchX)
	numWorkers := runtime.NumCPU()
	if batchSize < numWorkers {
		numWorkers = batchSize
	}

	n.initWorkers(numWorkers)
	workers := n.workerPool[:numWorkers]

	// Clear global gradients
	n.ClearGradients()

	if cap(n.batchLossBuf) < batchSize {
		n.batchLossBuf = make([]float32, batchSize)
	}
	losses := n.batchLossBuf[:batchSize]
	var wg sync.WaitGroup

	// Distribute work across workers
	chunkSize := (batchSize + numWorkers - 1) / numWorkers
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > batchSize {
			end = batchSize
		}
		if start >= end {
			continue
		}

		workers[w].ClearGradients()

		wg.Add(1)
		go func(worker *Network, s, e int) {
			defer wg.Done()

			for i := s; i < e; i++ {
				// Reset internal state for layers that support it
				for _, l := range worker.layers {
					if resettable, ok := l.(interface{ Reset() }); ok {
						resettable.Reset()
					}
				}

				// Forward pass
				yPred := worker.Forward(batchX[i])

				// Compute loss
				losses[i] = worker.loss.Forward(yPred, batchY[i])

				// Compute loss gradient
				yPredLen := len(yPred)
				// Use worker's lossGradBuf to avoid allocation
				if cap(worker.lossGradBuf) < yPredLen {
					worker.lossGradBuf = make([]float32, yPredLen)
				}
				grad := worker.lossGradBuf[:yPredLen]

				if backwardInPlace, ok := worker.loss.(loss.BackwardInPlacer); ok {
					backwardInPlace.BackwardInPlace(yPred, batchY[i], grad)
				} else {
					grad = worker.loss.Backward(yPred, batchY[i])
				}

				// Backward pass (accumulates into worker's private gradients)
				_ = worker.accumulateBackward(grad)
			}
		}(workers[w], start, end)
	}

	wg.Wait()

	// Sum all gradients from workers into the master gradient buffer
	for w := 0; w < numWorkers; w++ {
		wGrads := workers[w].grads
		for i := range n.grads {
			n.grads[i] += wGrads[i]
		}
	}

	// Average gradients
	invBatchSize := float32(1.0 / float64(batchSize))
	for i := range n.grads {
		n.grads[i] *= invBatchSize
	}

	// Optimization step
	n.Step()

	// Compute total loss
	totalLoss := float32(0.0)
	for _, l := range losses {
		totalLoss += l
	}
	return totalLoss / float32(batchSize)
}

// trainBatchSequential performs training on a batch of samples sequentially.
// Gradients are accumulated and averaged over the batch.
func (n *Network) trainBatchSequential(batchX [][]float32, batchY [][]float32) float32 {
	batchSize := float32(len(batchX))
	var totalLoss float32

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
			n.lossGradBuf = make([]float32, yPredLen)
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
	// Apply directly to global gradient buffer
	if len(n.grads) > 0 {
		invBatchSize := 1.0 / batchSize
		for i := range n.grads {
			n.grads[i] *= invBatchSize
		}
	} else {
		for _, l := range n.layers {
			grads := l.Gradients() // Get current accumulated gradients
			for j := range grads {
				grads[j] /= batchSize
			}
			l.SetGradients(grads) // Set averaged gradients back
		}
	}

	// Single optimization step (averaged over batch)
	n.Step()
	return totalLoss / batchSize
}

// Params returns all network parameters flattened as a direct view.
func (n *Network) Params() []float32 {
	if len(n.params) > 0 {
		return n.params
	}
	// Fallback for empty networks or if not initialized
	var params []float32
	for _, l := range n.layers {
		params = append(params, l.Params()...)
	}
	return params
}

// Gradients returns all network gradients flattened as a direct view.
func (n *Network) Gradients() []float32 {
	if len(n.grads) > 0 {
		return n.grads
	}
	// Fallback for empty networks or if not initialized
	var gradients []float32
	for _, l := range n.layers {
		gradients = append(gradients, l.Gradients()...)
	}
	return gradients
}

// ClearGradients zeroes out all layer gradients.
func (n *Network) ClearGradients() {
	if len(n.grads) > 0 {
		for i := range n.grads {
			n.grads[i] = 0
		}
	} else {
		for _, l := range n.layers {
			l.ClearGradients()
		}
	}
}

// SetGradients sets network gradients from a flattened slice.
func (n *Network) SetGradients(gradients []float32) {
	if len(gradients) == 0 {
		return
	}
	if len(n.grads) > 0 && &n.grads[0] == &gradients[0] {
		return
	}
	if len(gradients) == len(n.grads) {
		copy(n.grads, gradients)
	} else {
		offset := 0
		for _, l := range n.layers {
			lGrads := l.Gradients()
			gradLen := len(lGrads)
			l.SetGradients(gradients[offset : offset+gradLen])
			offset += gradLen
		}
	}
}

// Layers returns the network's layers slice.
func (n *Network) Layers() []layer.Layer {
	return n.layers
}

// Loss returns the network's loss function.
func (n *Network) Loss() loss.Loss {
	return n.loss
}

// Optimizer returns the network's optimizer.
func (n *Network) Optimizer() opt.Optimizer {
	return n.opt
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
	var params []float32
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

// SaveBinaryWeights saves all network parameters to a simple binary format.
// Format:
// - Magic: "GONN" (4 bytes)
// - Version: 1 (int32)
// - NumTensors: (int32)
// - For each tensor:
//   - NameLen: (int32)
//   - Name: (string)
//   - Rank: (int32)
//   - Shape: (int32 array)
//   - DataLen: (int32, number of float32s)
//   - Data: (float32 array)
func (n *Network) SaveBinaryWeights(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	// 1. Write magic and version
	if _, err := file.Write([]byte("GONN")); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, int32(1)); err != nil {
		return err
	}

	// 2. Collect all named parameters from all layers
	type tensor struct {
		name  string
		shape []int32
		data  []float32
	}
	var tensors []tensor

	for i, l := range n.layers {
		prefix := fmt.Sprintf("layer_%d_", i)
		typeName := "unknown"
		// Use ExtractLayerConfig to get type if possible, or just use type switching
		cfg := ExtractLayerConfig(l)
		typeName = cfg.Type
		prefix = fmt.Sprintf("layer_%d_%s_", i, typeName)

		if provider, ok := l.(layer.NamedParamProvider); ok {
			named := provider.NamedParams()
			for _, p := range named {
				shape32 := make([]int32, len(p.Shape))
				for j, v := range p.Shape {
					shape32[j] = int32(v)
				}
				tensors = append(tensors, tensor{
					name:  prefix + p.Name,
					shape: shape32,
					data:  p.Data,
				})
			}
		} else {
			// Fallback: just export Params() as a flat tensor
			params := l.Params()
			if len(params) > 0 {
				tensors = append(tensors, tensor{
					name:  prefix + "params",
					shape: []int32{int32(len(params))},
					data:  params,
				})
			}
		}
	}

	// 3. Write number of tensors
	if err := binary.Write(file, binary.LittleEndian, int32(len(tensors))); err != nil {
		return err
	}

	// 4. Write each tensor
	for _, t := range tensors {
		// Name
		nameBytes := []byte(t.name)
		if err := binary.Write(file, binary.LittleEndian, int32(len(nameBytes))); err != nil {
			return err
		}
		if _, err := file.Write(nameBytes); err != nil {
			return err
		}

		// Rank
		if err := binary.Write(file, binary.LittleEndian, int32(len(t.shape))); err != nil {
			return err
		}

		// Shape
		if err := binary.Write(file, binary.LittleEndian, t.shape); err != nil {
			return err
		}

		// Data Length
		if err := binary.Write(file, binary.LittleEndian, int32(len(t.data))); err != nil {
			return err
		}

		// Data
		if err := binary.Write(file, binary.LittleEndian, t.data); err != nil {
			return err
		}
	}

	return nil
}

// LayerConfig holds the configuration needed to reconstruct a layer.
type LayerConfig struct {
	Type    string
	InSize  int
	OutSize int
	Params  []float32

	// Activation type
	Activation string
	Alpha      float32 // For LeakyReLU, PReLU, ELU

	// Conv2D & Pooling parameters
	KernelSize  int
	Stride      int
	Padding     int
	InChannels  int
	OutChannels int

	// Dropout parameters
	Prob float32

	// Normalization parameters
	Eps             float32
	Affine          bool
	NormalizedShape int

	// RBF parameters
	NumCenters int
	Gamma      float32
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
		cfg.Activation = getActivationType(v.GetActivation())
		if prelu, ok := v.GetActivation().(*activations.PReLU); ok {
			cfg.Alpha = prelu.Alpha
		} else if leaky, ok := v.GetActivation().(*activations.LeakyReLU); ok {
			cfg.Alpha = leaky.Alpha
		} else if elu, ok := v.GetActivation().(activations.ELU); ok {
			cfg.Alpha = elu.Alpha
		} else if eluPtr, ok := v.GetActivation().(*activations.ELU); ok {
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
