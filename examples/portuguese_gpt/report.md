
# Architecture Overview
- Layers: Embedding -> PositionalEncoding -> TransformerBlock (Causal) -> SequenceUnroller(Dense+Softmax)
- Hyper-parameters:
  - Embedding Dim: 32
  - Attention Heads: 4
  - FF Dim: 64
  - Seq Len: 12
  - Optimizer: Adam (LR=0.002)

# Dataset Description
- Source: Small corpus of Portuguese (PT-PT) sentences.
- Task: Next Token Prediction.

# Training Results
- Final Loss: Evaluation results stored in logs.
- Accuracy: Tracked during training and final evaluation.
- Device: 1

# Inference Performance
- Latency per sequence: Optimized with 1
- Sampling: Temperature-based and Top-K for diverse generation.

# Benchmarking
- Memory Layout: Contiguous float32
- Hardware Acceleration: true
