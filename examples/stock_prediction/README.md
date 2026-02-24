# Stock Price Prediction with LSTMs

This example demonstrates the application of **Long Short-Term Memory (LSTM)** networks to financial time-series forecasting. It uses historical closing prices of a specific stock (e.g., GOOG) to predict future price movements.

## Technical Objectives

1.  **Time-Series Windows**: Transforming raw sequential data into overlapping lookback windows.
2.  **Min-Max Scaling**: Normalizing financial data to the [0, 1] range for improved gradient stability.
3.  **Trend Regression**: Learning to map past price sequences to a single future point.

## Key Implementation Details

### 1. Data Pipeline
The model processes a sliding window of past closing prices (e.g., 60 days) as a single input sequence. The targets are the normalized closing prices of the following day.

### 2. Recurrent Architecture
```go
model := goneuron.NewSequential(
    goneuron.SequenceUnroller(goneuron.LSTM(1, 64), lookback, false),
    goneuron.Dense(64, 1, goneuron.Linear),
)
```
*   **Rationale**: LSTMs are used to mitigate the vanishing gradient problem inherent in standard RNNs, allowing the model to capture longer-term dependencies in the stock's price history.

## Execution

Ensure the historical CSV data is present, then run:

```bash
go run main.go
```
