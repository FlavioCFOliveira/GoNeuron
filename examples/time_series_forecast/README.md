# Multivariate Multi-Step Time Series Forecasting

This advanced example demonstrates forecasting multiple variables simultaneously over multiple future timesteps. It showcases the library's ability to handle complex, high-dimensional time-series data.

## Technical Objectives

1.  **Multivariate Inputs**: Processing multiple parallel time-series (e.g., Open, High, Low, Close, Volume).
2.  **Multi-Step Outputs**: Predicting a vector of future values rather than a single point.
3.  **Vector Regression**: Mapping complex history to a future "horizon".

## Key Implementation Details

### 1. Forecasting Architecture
```go
model := goneuron.NewSequential(
    goneuron.SequenceUnroller(goneuron.LSTM(numFeatures, 128), lookback, false),
    goneuron.Dense(128, forecastStep * numFeatures, goneuron.Linear),
)
```
*   **Mechanism**: The output layer is sized to provide all predicted features for all future steps in a single flat vector. This vector is then reshaped for evaluation and visualization.

### 2. Data Preparation
The implementation includes sophisticated scaling and windowing logic that ensures both input features and output targets are properly normalized and aligned across the forecast horizon.

## Execution

```bash
go run main.go
```
