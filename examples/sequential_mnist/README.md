# Sequential MNIST with LSTMs

This example presents a unique approach to MNIST classification by treating each 28x28 image as a sequence of 28 rows, where each row contains 28 pixel values. It demonstrates the versatility of **Long Short-Term Memory (LSTM)** units in capturing temporal dependencies within spatial data.

## Technical Objectives

1.  **Recurrent Spatial Modeling**: Using LSTMs to process image rows sequentially.
2.  **Sequence Unrolling**: Demonstrating the `SequenceUnroller` utility to adapt recurrent cells for classification tasks.
3.  **Temporal Feature Integration**: Capturing the relationship between consecutive image segments.

## Key Implementation Details

### 1. Recurrent Architecture
```go
model := goneuron.NewSequential(
    goneuron.SequenceUnroller(goneuron.LSTM(28, 128), 28, false),
    goneuron.Dense(128, 10, goneuron.LogSoftmax),
)
```
*   **Mechanism**: Each image is transformed into a $(28 \times 28)$ tensor. The `LSTM` processes the 28 rows one by one. The final hidden state (representing the summarized features of all rows) is then passed to a `Dense` layer for 10-class classification.

### 2. Sequence Handling
The `SequenceUnroller` manages the internal state propagation across the 28 timesteps, ensuring the gradients are correctly backpropagated through time (BPTT).

## Execution

```bash
go run main.go
```
