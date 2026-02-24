# GGUF Export Example

This example demonstrates how to export a GoNeuron model to the GGUF v3 format.

GGUF (GGML Universal File) is a binary format designed for fast loading and saving of models, and is widely used in the LLM ecosystem (e.g., llama.cpp).

## Features demonstrated:
- Exporting a model to standard Float32 GGUF.
- Exporting a model to compressed Float16 GGUF.
- Automatic mapping of GoNeuron layers to GGUF tensor naming conventions (`blk.N.weight`, `blk.N.bias`).

## Usage

```bash
go run main.go
```

## Why use GGUF?
- **Interoperability**: Share models between GoNeuron and other tools or languages.
- **Efficiency**: Faster loading times compared to GOB or JSON.
- **Compression**: Support for Float16 reduces model size by 50% with minimal precision loss.
