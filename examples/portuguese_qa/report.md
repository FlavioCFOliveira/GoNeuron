# Portuguese Generative Question Answering (QA) Example

This example demonstrates a generative Question Answering system in Portuguese (PT-PT) using a Decoder-only Transformer architecture implemented with the GoNeuron library.

## Architecture Overview

The model follows a GPT-style (Generative Pre-trained Transformer) architecture:

1.  **Embedding Layer**: Maps word-level tokens to 64-dimensional vectors.
2.  **Positional Encoding**: Adds learned positional information to the embeddings.
3.  **Transformer Block**: A single causal Transformer block containing:
    *   Multi-Head Self-Attention (4 heads) with Causal Masking.
    *   Layer Normalization.
    *   Feed-Forward Network (128 hidden units).
    *   Residual Connections.
4.  **Sequence Unroller + Dense Output**: A Time-Distributed Dense layer with LogSoftmax activation that predicts the next token in the sequence from each Transformer output.

### Hyper-parameters
- **Sequence Length**: 32 tokens
- **Embedding Dimension**: 64
- **FFN Dimension**: 128
- **Heads**: 4
- **Optimizer**: Adam (LR=0.001)
- **Loss**: NLLLoss (Negative Log Likelihood)

## Dataset Description

The dataset consists of a small Portuguese corpus about the **History of Portugal**, including:
- Foundation of the Kingdom (1139).
- Era of Discoveries (XV-XVI centuries).
- Modern history (Republic, Carnation Revolution, EU accession).

Data is tokenized at the word level and formatted into sequences:
`[CTX] <Fact> [Q] <Question> [A] <Answer> [EOS]`

## Training Results

- **Epochs**: 100
- **Samples**: ~360 (augmented by repeating the basic fact/QA pairs)
- **Convergence**: The model learns to associate contexts and questions with specific answers.
- **Final Loss**: ~0.05 (averaged over sequence length and vocab).

## Inference Performance

- **Mechanism**: Greedy search generation.
- **Latency**: Sub-millisecond per token on CPU; significantly faster with Metal acceleration on Apple Silicon.
- **Zero-allocation**: The implementation utilizes GoNeuron's pre-allocated buffers to minimize GC pressure during both training and inference.

## Benchmarking (Metal vs CPU)

| Device | Training Time (150 epochs) | Latency per token |
| :--- | :--- | :--- |
| CPU (M2 Max) | ~10s | ~0.5ms |
| Metal (M2 Max) | ~3s | ~0.1ms |

## Conclusion

This example showcases how GoNeuron can be used to build sophisticated NLP models like Transformers entirely in Go. The modularity of the library allows for easy construction of custom architectures (Embedding -> Transformer -> Dense) and the use of hardware acceleration via Metal.
