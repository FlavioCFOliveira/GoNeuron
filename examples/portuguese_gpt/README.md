# Mini-GPT for Portuguese Text Generation

This example demonstrates the implementation of a generative **Transformer** model optimized for the Portuguese language. It follows the GPT architecture (Generative Pre-trained Transformer), utilizing causal self-attention for sequence completion.

## Technical Objectives

1.  **Causal Self-Attention**: Implementing a `TransformerBlock` with masking to ensure the model only attends to previous tokens.
2.  **Positional Encoding**: Injecting structural sequence information into the non-recurrent Transformer architecture.
3.  **Generative Sampling**: Demonstrating **Temperature** and **Top-K** sampling techniques for coherent and diverse text generation.

## Key Implementation Details

### 1. Transformer Core
```go
model := goneuron.NewSequential(
    goneuron.Embedding(vocabSize, embedDim),
    goneuron.PositionalEncoding(maxLen, embedDim),
    goneuron.TransformerBlock(embedDim, numHeads, maxLen, ffDim, true), // Causal=true
    goneuron.SequenceUnroller(goneuron.Dense(embedDim, vocabSize, goneuron.LogSoftmax), maxLen, true),
)
```
*   **Design**: The `TransformerBlock` is configured in causal mode, making it suitable for autoregressive generation. The `SequenceUnroller` at the output layer predicts the next token for every position in the input sequence.

### 2. Advanced Inference
The implementation includes a generation loop that uses the model's logits to sample new tokens. By adjusting the `Temperature`, users can control the "creativity" of the output, while `Top-K` sampling prevents the model from choosing highly improbable tokens.

## Execution

```bash
go run main.go
```
