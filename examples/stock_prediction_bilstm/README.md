# Bidirectional LSTM for Stock Prediction

This example utilizes a **Bidirectional LSTM** to process stock price sequences. By processing the history in both forward and backward directions, the model can capture more complex temporal patterns.

## Technical Objectives

1.  **Bidirectional Context**: Learning representations that consider the sequence from both temporal ends.
2.  **Complex Trend Analysis**: Enhancing the feature set for the final regression layer.

## Key Implementation Details

### 1. Bidirectional Architecture
```go
model := goneuron.NewSequential(
    goneuron.SequenceUnroller(goneuron.Bidirectional(goneuron.LSTM(1, 64)), lookback, false),
    goneuron.Dense(128, 1, goneuron.Linear), // Bidirectional doubles the hidden size
)
```
*   **Mechanism**: The `Bidirectional` wrapper processes the input sequence twice. The outputs are concatenated, providing the classification head with a richer 128-dimensional summary of the 64-unit forward and 64-unit backward states.

## Execution

```bash
go run main.go
```
