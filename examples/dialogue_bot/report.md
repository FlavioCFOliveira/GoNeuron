# Dialogue Bot Report

## Architecture Overview
- **Type**: Causal Transformer (Autoregressive)
- **Sequence Length**: 64 (tuned for conversation context)
- **Embedding Dimension**: 64
- **Attention**: 4 heads.
- **Device**: Metal/GPU accelerated.

## Dataset Description
- **Source**: Manual Q&A pairs in Portuguese/English.
- **Format**: `User: ... \n Bot: ...` structure.

## Training Results
- **Epochs**: 400
- **Convergence**: Final loss indicates successful memorization and pattern matching of dialogue pairs.

## Inference Performance
- **Capability**: Responds correctly to seen queries and generalizes simple greeting patterns.
- **Latency**: Sub-millisecond per token generation.

## Benchmarking
- **Optimization**: Contiguous memory layout and pre-allocated buffers.
