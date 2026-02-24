# Radial Basis Function (RBF) Sine Approximation

This example demonstrates function approximation of a non-linear sine wave using a **Radial Basis Function (RBF)** network. Unlike standard MLPs, RBF networks use distance-based activation functions in the hidden layer.

## Technical Objectives

1.  **RBF Layer Implementation**: Utilizing `NewRBF` for local receptive field modeling.
2.  **Hybrid Training**: Combining distance-based kernels with linear output layers.
3.  **Low-Level API Usage**: Demonstrating direct interaction with `internal/net` and `internal/layer` for custom training logic.

## Key Implementation Details

### 1. RBF Architecture
```go
layers := []layer.Layer{
    layer.NewRBF(1, 50, 1, 0.5), // In, Centers, Out, Gamma
    layer.NewDense(50, 1, activations.Linear{}),
}
```
*   **Mechanism**: The RBF layer transforms the 1D input into a 50D space based on Gaussian kernels centered across the input range. The `Gamma` parameter controls the width of these kernels, determining the "smoothness" of the approximation.

### 2. Functional Mapping
The network is trained to map $x \in [0, 2\pi]$ to $\sin(x)$. This highlights the RBF network's efficiency in approximating periodic and continuous functions with relatively few parameters compared to deep MLPs.

## Execution

Run the RBF simulation:

```bash
go run main.go
```
