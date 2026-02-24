# Synthetic Regression with Huber Loss

This example explores high-dimensional regression on synthetic data. It specifically highlights the use of **Huber Loss**, which provides robustness against outliers in the training data.

## Technical Objectives

1.  **Robust Regression**: Implementing loss functions that balance Mean Absolute Error (MAE) and Mean Squared Error (MSE).
2.  **Deep FNN Architecture**: Constructing a multi-layer deep network to capture complex non-linear relationships.
3.  **Noise Handling**: Demonstrating the network's ability to generalize despite synthetic noise.

## Key Implementation Details

### 1. Robust Loss Configuration
```go
model.Compile(goneuron.Adam(0.001), goneuron.Huber)
```
*   **Mechanism**: `Huber Loss` acts like MSE for small errors and like MAE for large errors. This prevents outliers from disproportionately influencing the gradient updates, leading to more stable training in noisy environments.

### 2. Multi-Layer Topology
```go
model := goneuron.NewSequential(
    goneuron.Dense(8, 64, goneuron.ReLU),
    goneuron.Dense(64, 32, goneuron.ReLU),
    goneuron.Dense(32, 1, goneuron.Linear),
)
```
*   **Rationale**: The 8-64-32 architecture provides sufficient capacity to learn the underlying function of the 8-feature synthetic input.

## Execution

Run the regression tutorial:

```bash
go run main.go
```
