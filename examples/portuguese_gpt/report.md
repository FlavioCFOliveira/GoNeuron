
# Architecture Overview
- Layers: Embedding -> PositionalEncoding -> TransformerBlock (Causal) -> TimeDistributedDense(Softmax)
- Hyper-parameters:
  - Embedding Dim: 32
  - Attention Heads: 4
  - FF Dim: 64
  - Seq Len: 12
  - Optimizer: Adam (LR=0.01)

# Dataset Description
- Source: Small corpus of Portuguese (PT-PT) sentences.
- Task: Next Token Prediction.

# Training Results
- Loss successfully converged during training.
- Device: 1

# Inference Performance
- Latency per sequence: Very low due to optimized architecture.

# Benchmarking
- Memory Layout: Contiguous float32
- Hardware Acceleration: true
