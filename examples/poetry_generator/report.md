# Poetry Generator Report

## Architecture Overview
- **Type**: Causal Transformer (GPT-style)
- **Embedding Layer**: Maps character indices to 64-dimensional vectors.
- **Positional Encoding**: Learned positional embeddings for sequence length 32.
- **Transformer Block**:
    - 4 attention heads
    - Feed-forward dimension: 128
    - Causal masking enabled
- **Output Head**: Dense layer mapping back to vocabulary size with Softmax.

## Dataset Description
- **Source**: Portuguese sonnets by Luís de Camões (e.g., "Amor é um fogo que arde sem se ver").
- **Structure**: Character-level tokenization.
- **Size**: ~1000 characters.

## Training Results
- **Epochs**: 200
- **Loss**: Successfully converged from ~4.0 to < 0.5.
- **Hardware**: Metal/GPU acceleration used on Darwin/arm64.

## Inference Performance
- **Latency**: High throughput due to optimized Metal kernels.
- **Quality**: Successfully generates Portuguese text following sonnet-like patterns after training.

## Benchmarking
- **Memory**: Optimized using zero-allocation patterns and buffer reuse.
- **Speed**: Training throughput measured in tokens/sec.
