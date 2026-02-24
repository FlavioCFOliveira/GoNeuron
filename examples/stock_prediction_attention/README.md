# Stock Prediction with Attention Mechanism

This example enhances a standard LSTM model with a **Global Attention** layer. This allows the network to dynamically weight the importance of different days in the lookback window when making a prediction.

## Technical Objectives

1.  **Temporal Attention**: Identifying which past price points have the highest predictive value for the current target.
2.  **Interpretability**: Providing a mechanism (attention weights) that can be inspected to understand model focus.

## Key Implementation Details

### 1. Attention-Augmented Architecture
```go
model := goneuron.NewSequential(
    goneuron.SequenceUnroller(goneuron.LSTM(1, 64), lookback, true), // Return sequences=true
    goneuron.GlobalAttention(64),
    goneuron.Dense(64, 1, goneuron.Linear),
)
```
*   **Design**: By setting the LSTM to return the full sequence of hidden states, the `GlobalAttention` layer can compute a weighted sum across the entire history, rather than relying solely on the final state.

## Execution

```bash
go run main.go
```
