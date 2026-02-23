# Metal Optimization Benchmarks (2026-02-23)

## Test Environment
- Device: Apple M4
- OS: Darwin (macOS)
- Architecture: arm64

## Dense Layer Forward Pass (MatMul + Bias + Tanh)

| Configuration (In x Out) | CPU (ns/op) | Metal (ns/op) | Speedup |
|--------------------------|-------------|---------------|---------|
| 784 x 256                | 169,304     | 276,211       | 0.61x   |
| 1024 x 1024              | 837,244     | 491,608       | 1.70x   |
| 4096 x 4096              | 14,002,007  | 1,036,718     | 13.5x   |

### Key Improvements
1. **Fused Bias Addition**: Bias vector is broadcasted and added on GPU using a custom kernel, eliminating the need to read back partial results to CPU.
2. **Fused Activation**: Activation functions (Sigmoid, Tanh, ReLU) are applied directly in the same GPU pipeline after bias addition.
3. **Zero-Copy persistent buffers**: Reusing `MTLBuffer` for weights, biases, and intermediate activations to minimize synchronization overhead.

### Conclusion
Metal acceleration provides significant performance gains for large layers (up to 13.5x for 4096x4096x1). For smaller layers, the overhead of GPU kernel dispatch and synchronization outweighs the computation savings. The library now automatically selects the most efficient path based on device availability and training state.
