# GGUF Export Tutorial

This example demonstrates the end-to-end workflow of defining, training, and serializing a neural network using the **GGUF v3** format. GGUF is a binary serialization format designed for fast loading and high performance in machine learning inference, specifically optimized for the GGML ecosystem.

## Technical Objectives

1.  **Model Architecture Definition**: Constructing a Multi-Layer Perceptron (MLP) for non-linear function approximation (XOR problem).
2.  **Training Orchestration**: Utilizing the `Fit` method with the Adam optimizer and Mean Squared Error (MSE) loss.
3.  **Serialization Protocols**:
    - Full-precision export (32-bit floating point).
    - Reduced-precision export (16-bit floating point) for optimized memory footprint.

## Key Implementation Details

The implementation highlights the following critical segments:

### 1. Model Initialization
```go
layers := []layer.Layer{
    layer.NewDense(2, 8, activations.Tanh{}),
    layer.NewDense(8, 1, activations.Sigmoid{}),
}
model := net.New(layers, loss.MSE{}, opt.NewAdam(0.01))
```
*   **Rationale**: The use of `activations.Tanh` in the hidden layer prevents the "dead neuron" problem associated with ReLU in small-scale logic problems like XOR.

### 2. GGUF V3 Serialization
```go
// Standard F32 Export
err := model.SaveGGUF("xor_model_f32.gguf")

// Half-Precision F16 Export
err := model.SaveGGUFExt("xor_model_f16.gguf", net.GGMLTypeF16)
```
*   **Significance**: `SaveGGUFExt` leverages the `GGMLTypeF16` enum to cast 32-bit weights to 16-bit representations during serialization. This reduces storage requirements by approximately 50% while maintaining sufficient numerical stability for most inference tasks.

## Execution

Run the tutorial using the Go toolchain:

```bash
go run main.go
```

The output will provide a comparative analysis of the generated binary files, including their byte sizes and serialization status.
