# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Neural Network Research Project

## Architecture

A neural network library written in Go with a modular architecture:

```
internal/
├── net/        # Core Network interface and training logic
├── layer/      # Layer implementations (Dense, Conv2D, LSTM, Transformer, etc.)
├── activations/# Activation functions (ReLU, Sigmoid, Tanh, LeakyReLU, LogSoftmax)
├── loss/       # Loss functions (MSE, CrossEntropy, Huber, NLLLoss, BCELoss)
└── opt/        # Optimizers (SGD, Adam)
```

## Performance Optimizations

### Hardware Acceleration
1. **Metal/MPS support**: Optimized backends for Apple Silicon (macOS)
2. **Kernel Fusing**: Combining operations (MatMul + Bias + Activation) into single GPU kernels
3. **Persistent Buffers**: Reusing GPU buffers to minimize synchronization overhead

### Memory Optimization (Zero Allocation Patterns)
1. **Pre-allocated buffers**: All intermediate values are allocated once in layer constructors
   - `inputBuf`, `outputBuf`, `preActBuf`, `gradWBuf`, `gradBBuf`, `gradInBuf`, `dzBuf` in Dense
   - `cellBuf`, `hiddenBuf`, `preActBuf` in LSTM
2. **Contiguous memory layout**: Weights stored as row-major `[]float64` for cache efficiency
3. **In-place updates**: `SetParams` and optimizer step methods

### Key Implementation Patterns
```go
// Forward: Reuse buffers, avoid allocations
copy(d.inputBuf, x)
// ... compute using pre-allocated buffers ...
d.preActBuf[o] = sum  // Store for backward pass

// Backward: Use pre-allocated buffers for gradient computation
for o := 0; o < outSize; o++ {
    d.dzBuf[o] = grad[o] * d.act.Derivative(d.preActBuf[o])
    d.gradBBuf[o] = d.dzBuf[o]
}
```

### Important Implementation Details
1. **Dense layer**: Uses nested loops or Metal MatMul for performance. Accumulates gradients (`gradB += ...`) instead of overwriting.
2. **BatchNorm2D**: Uses NCHW (Channel-major) layout for consistency with Conv2D.
3. **Activation derivatives**:
   - ReLU.Derivative(0) = 0 (x must be > 0 for gradient)
   - LeakyReLU.Derivative(0) = Alpha (not 0.5)
4. **Huber loss**: Gradient is `y_pred - y_true` for small errors, `delta * sign(diff)` for large
5. **Transformer**: Implemented using modular `TransformerBlock`, `MultiHeadAttention`, and `PositionalEncoding`.

## Common Commands

```bash
go mod download      # Download dependencies
go test ./...        # Run tests
go test -bench=. -benchmem ./...  # Run benchmarks with memory stats
```

## Examples

- **Location**: Always use the `examples/` folder at the project root for any neural network examples.
- **Structure**: Each example must have its own dedicated subfolder containing all necessary files (code, data, etc.).

## Adding New Components

1. **Activation**: Implement the `activations.Activation` interface with `Activate()` and `Derivative()` methods
2. **Loss**: Implement the `loss.Loss` interface with `Forward()` and `Backward()` methods
3. **Layer**: Implement the `layer.Layer` interface including `Params()`, `SetParams()`, and `Gradients()` methods
4. **Optimizer**: Implement the `opt.Optimizer` interface with `Step()` and `StepInPlace()` methods

## Implementation Notes

### Key Design Decisions
1. **No external BLAS library**: Dense layer uses nested loops over `[]float64` for cache efficiency
2. **Xavier/Glorot initialization**: Weights initialized with `scale = sqrt(2.0 / (in + out))`
3. **Bias initialization**: Small random values in range `[-0.1, 0.1]`
4. **Pre-activation storage**: Forward pass stores `preAct` values for backward pass gradient computation

### Common Pitfalls
1. **SGD optimizer**: The `Step()` method in Network must use `SetParams` after computing step - `StepInPlace` modifies a copy if used directly on `Params()` return value
2. **ReLU dying neurons**: ReLU activation can cause neurons to die (output 0, no gradient). Use Tanh for problems like XOR.
3. **Huber gradient signs**: Gradient is `(y_pred - y_true)` for small errors, `delta * sign(y_pred - y_true)` for large

### Test Notes
- Tests verify mathematical correctness of forward/backward passes
- TestNetworkXOR uses Tanh instead of ReLU to avoid dying neuron problem
- Huber loss tests verify gradient signs match implementation