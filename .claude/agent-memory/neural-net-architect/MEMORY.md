# Neural Network Architect Memory

This file contains project-specific knowledge and patterns for the GoNeuron library.

## Common Issues and Fixes

### LSTM State Accumulation Bug (Critical)
**Issue**: LSTM layers accumulate internal state (hidden and cell states) across training samples in a batch, causing incorrect gradients and random performance.

**Root Cause**: The `Network.Train()`, `TrainBatch()`, and `Forward()` functions did not call `Reset()` on layers with internal state before each forward pass.

**Fix**: Added `Reset()` calls before each forward pass in:
- `network.go::Train()` - calls Reset for layers implementing the interface
- `network.go::trainBatchSequential()` - calls Reset before each sample
- `network.go::trainBatchParallel()` - calls Reset for each sample in worker goroutines
- `network.go::TrainTriplet()` - calls Reset before forward passes
- `network.go::ForwardBatch()` - calls Reset before each sample

**Code Pattern**:
```go
for _, l := range n.layers {
    if resettable, ok := l.(interface{ Reset() }); ok {
        resettable.Reset()
    }
}
```

**Impact**: Without this fix, sequential MNIST LSTM was achieving ~10% accuracy (random guessing) because LSTM state accumulated incorrectly across samples.

### XAavier/Glorot Initialization for LSTMs
**Issue**: Standard Xavier initialization (`sqrt(2.0 / (in + out))`) is suboptimal for LSTMs because they have 4 gates.

**Note**: This is not causing random performance but could affect convergence. Consider using orthogonal initialization for recurrent weights for better LSTM training.

### Forget Gate Bias Initialization
**Implementation**: In `lstm.go`, forget gate biases are initialized to 1.0 (indices `outSize` to `2*outSize - 1`) to encourage initial "no forgetting" behavior.

### BatchActivations and Dense Layer
The Dense layer supports `BatchActivation` interface for activations like Softmax and LogSoftmax that need to process the full output vector at once.

**Key Files**:
- `internal/layer/lstm.go` - LSTM implementation with BPTT
- `internal/layer/layer.go` - Dense layer with BatchActivation support
- `internal/net/net.go` - Network training with Reset() calls
- `internal/activations/activations.go` - Activation functions including LogSoftmax
- `internal/loss/loss.go` - Loss functions including NLLLoss

## Training Patterns

### Zero-Allocation Pattern
1. All intermediate buffers are pre-allocated in constructors
2. Forward/Backward reuse buffers without allocations
3. Optimizer modifies params in-place via `StepInPlace()`

### Common Architecture for Sequential Data
```go
// LSTM for sequential processing
lstm := layer.NewLSTM(inputSize, hiddenSize)
dense := layer.NewDense(hiddenSize, outputSize, activations.Linear{})

// For classification with LogSoftmax + NLLLoss
denseLog := layer.NewDense(hiddenSize, numClasses, activations.LogSoftmax{})
loss := loss.NLLLoss{}
```

## Debugging Tips

1. **Check Reset() is called** - Add `fmt.Println("Reset called")` to verify state is cleared
2. **Verify Forward is idempotent** - Same input should give same output after Reset()
3. **Monitor gradients** - Use `Gradients()` to check for zero or NaN gradients
4. **Test with small networks** - Start with 1-2 layers to verify flow works
