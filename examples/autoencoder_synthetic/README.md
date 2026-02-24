# Synthetic Data Autoencoder

This example demonstrates the implementation of a **Symmetric Autoencoder** for dimensionality reduction and data compression. It learns to compress 10-dimensional input vectors into a 2-dimensional latent space and subsequently reconstruct the original signal.

## Technical Objectives

1.  **Latent Representation Learning**: Identifying a compressed bottleneck that captures the most salient features of the input data.
2.  **Self-Supervised Learning**: Training a network where the target output is identical to the input vector.
3.  **Dimensionality Reduction**: Visualizing how high-dimensional relationships can be mapped to lower-order manifolds.

## Key Implementation Details

### 1. Autoencoder Architecture
```go
model := goneuron.NewSequential(
    goneuron.Dense(10, 2, goneuron.ReLU),    // Encoder
    goneuron.Dense(2, 10, goneuron.Sigmoid), // Decoder
)
```
*   **Design**: The network is forced to learn a lossy compression of the 10D input because it must pass through a 2D "bottleneck" layer. The decoder then attempts to expand this 2D representation back to the original 10D space.

### 2. Loss Optimization
By minimizing the reconstruction error (MSE), the model learns a transformation similar to Principal Component Analysis (PCA) but with non-linear mapping capabilities provided by the ReLU activation.

## Execution

```bash
go run main.go
```
