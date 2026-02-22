# Deep FNN for Synthetic Housing Regression

This example demonstrates a deep Feedforward Neural Network (FNN) for regression using synthetic data modeled after housing price problems.

## Architecture
- **Layer 1**: Dense(8 -> 64) with ReLU.
- **Layer 2**: Dense(64 -> 32) with ReLU.
- **Layer 3**: Dense(32 -> 1) with Linear activation.
- **Loss**: Huber Loss (robust to outliers).

## Running the example
```bash
go run examples/regression_synthetic/main.go
```
