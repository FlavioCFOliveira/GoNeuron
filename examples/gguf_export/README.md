# GGUF Model Serialization

This example showcases the implementation of **GGUF v3** serialization, enabling full interoperability between GoNeuron and the broader LLM ecosystem (e.g., llama.cpp).

## Technical Objectives

1.  **Standardized Binary Serialization**: Implementing the GGUF header and tensor mapping specification.
2.  **Quantization and Precision**: Demonstrating `Float32` and `Float16` export modes.
3.  **Cross-Platform Interop**: Generating files that can be consumed by non-Go inference engines.

## Key Implementation Details

### 1. GGUF Mapping logic
The library automatically maps internal GoNeuron layer names to the standard GGUF convention (`blk.N.weight`, `blk.N.bias`), ensuring compatibility with universal model loaders.

### 2. F16 Quantization
```go
err := model.SaveGGUFExt("model_f16.gguf", net.GGMLTypeF16)
```
*   **Mechanism**: Weights are converted to 16-bit half-precision during the write operation. This halves the model's disk footprint and memory requirement during loading while typically maintaining >99% of the original model's accuracy.

## Execution

```bash
go run main.go
```
