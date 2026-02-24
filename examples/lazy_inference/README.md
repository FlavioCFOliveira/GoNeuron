# Lazy Shape Inference

This example demonstrates one of the most developer-friendly features of GoNeuron: **Lazy Shape Inference**. It shows how to build complex neural networks without manually calculating or specifying input dimensions for each layer.

## Technical Objectives

1.  **Automatic Topology Resolution**: Allowing the network to infer its own layer sizes upon the first forward pass.
2.  **Modular Prototyping**: Simplifying the process of re-ordering or adding layers without breaking dimension consistency.

## Key Implementation Details

### 1. Dimensionless Layer Definitions
```go
model := goneuron.NewSequential(
    goneuron.Dense(64, goneuron.ReLU), // Only output size specified
    goneuron.Dense(32, goneuron.ReLU),
    goneuron.Dense(1, goneuron.Sigmoid),
)
```
*   **Mechanism**: Unlike standard definitions where you must specify `Dense(input, output)`, the lazy API allows specifying only the `output` size. The library detects the input size automatically by inspecting the tensor shape during the initial `Predict` or `Forward` call.

### 2. Dynamic Initialization
The underlying buffers (parameters and gradients) are allocated only after the first input is received, ensuring that the memory layout perfectly matches the data provided.

## Execution

```bash
go run main.go
```
