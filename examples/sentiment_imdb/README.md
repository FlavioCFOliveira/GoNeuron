# IMDB Sentiment Analysis with Attention

This example implements a sophisticated Natural Language Processing (NLP) pipeline for binary sentiment classification using the IMDB movie reviews dataset. It combines **Bidirectional LSTMs** with a **Global Attention** mechanism to identify critical sentiment-bearing words.

## Technical Objectives

1.  **Sequence Modeling**: Utilizing `Bidirectional` LSTMs to capture context from both past and future words.
2.  **Attention Mechanism**: Implementing `GlobalAttention` to weight the importance of different timesteps in a sequence.
3.  **Word Embeddings**: Demonstrating how discrete tokens are mapped to continuous vector spaces for semantic processing.

## Key Implementation Details

### 1. Attention-Based Architecture
```go
model := goneuron.NewSequential(
    goneuron.Embedding(vocabSize, embedDim),
    goneuron.Bidirectional(goneuron.LSTM(embedDim, hiddenDim)),
    goneuron.GlobalAttention(hiddenDim*2),
    goneuron.Dense(hiddenDim*2, 1, goneuron.Sigmoid),
)
```
*   **Design**: The `Bidirectional` wrapper doubles the hidden state size (processing forward and backward). The `GlobalAttention` layer then aggregates these states by learning which parts of the review are most relevant to the sentiment, effectively acting as a learned pooling operation.

### 2. Global Context
Unlike simple LSTMs that rely on the last hidden state, the addition of Global Attention allows the network to "look back" at all words in the review simultaneously, significantly improving performance on long texts where the terminal state might suffer from information loss.

## Execution

```bash
go run main.go
```
