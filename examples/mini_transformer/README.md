# Compact Transformer Implementation

This example demonstrates a streamlined implementation of the **Transformer** architecture for sentiment analysis. It specifically showcases the `TransformerBlockExt` component, which offers enhanced configurability compared to standard blocks.

## Technical Objectives

1.  **Architectural Flexibility**: Comparing different normalization techniques (`LayerNorm` vs. `RMSNorm`) and activation functions within the Transformer block.
2.  **CLS Pooling**: Utilizing a dedicated [CLS] token to represent the entire sequence's sentiment.

## Key Implementation Details

### 1. Configurable Transformer Block
```go
model := goneuron.NewSequential(
    goneuron.Embedding(vocabSize, embedDim),
    goneuron.TransformerBlockExt(embedDim, numHeads, ffDim, goneuron.GELU, true), // Enhanced config
    goneuron.CLSPooling(maxLen, embedDim),
    goneuron.Dense(embedDim, 1, goneuron.Sigmoid),
)
```
*   **Design**: The `TransformerBlockExt` allows for fine-grained control over the internal feed-forward network's activation (e.g., using `GELU` for better gradient flow).

### 2. Classification via CLS
Instead of averaging all word vectors, this model learns a specific representation in the first token position (the CLS token) that summarizes the sentiment of the entire input.

## Execution

```bash
go run main.go
```
