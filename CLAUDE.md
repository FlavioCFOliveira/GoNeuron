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
1. **Arena-based state saving**: Intermediate activations and states are saved using offsets into a contiguous `arena []float32` buffer to avoid slice header allocations.
2. **Pre-allocated buffers**: All intermediate values are allocated once in layer constructors or during the first forward pass (growing as needed).
3. **Contiguous memory layout**: Weights and gradients are stored as flat `[]float32` buffers for cache efficiency and thread-safe worker sharing.
4. **Worker Pool**: Persistent worker networks avoid `Clone()` calls during `TrainBatch`.

### Key Implementation Patterns
```go
// ForwardWithArena: Save state using pointers to avoid stale slice headers after resize
if arena != nil && offset != nil {
    d.arenaPtr = arena
    // ... resize logic ...
    copy((*arena)[*offset:], x)
    *offset += len(x)
}

// LightweightClone: Share params/grads but keep private buffers
func (d *Dense) LightweightClone(params, grads []float32) Layer {
    return &Dense{
        params: params,
        grads:  grads,
        inputBuf: make([]float32, d.inSize),
        // ... other private buffers ...
    }
}
```

## Specialized Agents

Use these agents for specific tasks within the GoNeuron ecosystem:

- **`neural-net-architect`**: Design new layers, activation functions, or core network improvements.
- **`golang-performance-architect`**: Refactor/optimize Go code ensuring zero-allocation patterns and cache efficiency.
- **`metal-performance-expert`**: Implement and optimize Metal kernels for Apple Silicon acceleration.
- **`cuda-performance-optimizer`**: Implement and optimize CUDA kernels for NVIDIA GPU support.
- **`ml-training-validator`**: Verify model convergence and ensure mathematical consistency across different backends (CPU, Metal, CUDA).
- **`ml-dataset-fetcher`**: Source and download datasets (MNIST, CIFAR, etc.) into example directories.
- **`ml-architecture-expert`**: Design and implement complex neural network examples (e.g., Transformers, MoE) in the `examples/` folder.
- **`technical-researcher`**: Research mathematical derivations, SOTA architectures, or GPU programming manuals.
- **`security-architect`**: Audit low-level code (memory management, GPU kernels) for stability and safety.

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