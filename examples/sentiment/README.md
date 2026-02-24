# Basic Sentiment Analysis

A fundamental example of sentiment classification using a standard Feed-Forward Neural Network (FNN) and word-level representations.

## Technical Objectives

1.  **Vocabulary Mapping**: Converting text to fixed-size input vectors.
2.  **Dense Sentiment Mapping**: Using deep layers to identify polarity in short text snippets.

## Key Implementation Details

### 1. Feature Vectorization
The example demonstrates a simple approach to turning text into model-ready tensors, serving as a baseline for more complex recurrent or Transformer-based models.

### 2. Binary Classification head
```go
model := goneuron.NewSequential(
    goneuron.Dense(inputDim, 64, goneuron.ReLU),
    goneuron.Dense(64, 1, goneuron.Sigmoid),
)
```
*   **Design**: A classic architecture for binary sentiment tasks, providing a baseline for accuracy and speed comparisons.

## Execution

```bash
go run main.go
```
