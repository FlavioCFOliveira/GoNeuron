# Stock Prediction with Gated Recurrent Units (GRU)

An alternative financial forecasting model using **Gated Recurrent Units (GRU)**. GRUs offer a more computationally efficient architecture compared to LSTMs while maintaining competitive performance on time-series tasks.

## Technical Objectives

1.  **GRU Efficiency**: Comparing the performance and training speed of GRU vs. LSTM cells.
2.  **Sequential Feature Mapping**: Capturing volatility and trends using gated recurrent mechanisms.

## Key Implementation Details

### 1. GRU Architecture
```go
model := goneuron.NewSequential(
    goneuron.SequenceUnroller(goneuron.GRU(1, 64), lookback, false),
    goneuron.Dense(64, 1, goneuron.Linear),
)
```
*   **Design**: The GRU cell uses a simplified gating mechanism (reset and update gates), which often leads to faster convergence on smaller datasets compared to the three-gate LSTM.

## Execution

```bash
go run main.go
```
