# GGUF Export and Interoperability Examples

This directory contains specialized tutorials for the **GGUF v3** serialization format within the GoNeuron ecosystem. GGUF is the industry-standard binary format for model distribution, optimized for fast loading and cross-platform inference (e.g., via llama.cpp or other GGML-based backends).

## Tutorials

### 1. [End-to-End Export Pipeline](./export_tutorial)
A comprehensive guide to training a non-linear classifier and serializing its state using both full-precision and half-precision protocols.

**Technical Highlights:**
- **Training Orchestration**: MLP optimization for the XOR problem.
- **Full-Precision Serialization**: Native 32-bit floating point export via `SaveGGUF`.
- **Half-Precision Compression**: 16-bit optimization via `SaveGGUFExt` to reduce memory footprint by 50%.

### 2. [Model Restoration and Conversion](./load_and_export)
Demonstrates the restoration of pre-trained models from native `.gob` checkpoints and their immediate conversion to the GGUF ecosystem.

**Technical Highlights:**
- **State Restoration**: Utilizing `net.Load` to dynamically reconstruct network topology and parameters.
- **Cross-Format Bridge**: Converting resident Go memory structures to aligned GGUF binaries.
- **Format Efficiency Analysis**: Comparative metrics between native Go serialization and GGUF.

---

## Architectural Advantages of GGUF

1.  **Universal Interoperability**: Models trained in Go can be consumed by C++, Python, or Rust environments supporting the GGML specification.
2.  **Self-Describing Metadata**: The GGUF header encapsulates model architecture keys (layers, activation types, and tensor mappings) alongside raw weights.
3.  **Numerical Flexibility**: Built-in support for multiple quantization levels and precision formats (F32, F16).

## API Reference

```go
// Standard F32 Serialization
err := model.SaveGGUF("model.gguf")

// Advanced F16 Serialization
import "github.com/FlavioCFOliveira/GoNeuron/internal/net"
err := model.SaveGGUFExt("model_f16.gguf", net.GGMLTypeF16)
```
