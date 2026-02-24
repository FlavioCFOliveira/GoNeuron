# XOR Classification Example

This example demonstrates the solution to the classical XOR (Exclusive OR) problem using the GoNeuron high-level Sequential API. The XOR problem is a fundamental benchmark for neural networks as it requires a non-linear decision boundary that a single-layer perceptron cannot solve.

## Technical Objectives

1.  **Non-Linear Modeling**: Utilizing hidden layers with non-linear activation functions to solve non-linearly separable problems.
2.  **API Proficiency**: Demonstrating the `goneuron` package for rapid model prototyping and training.
3.  **Inference and Evaluation**: Performing post-training predictions and accuracy assessment.

## Key Implementation Details

### 1. Architectural Configuration
```go
model := goneuron.NewSequential(
    goneuron.Dense(2, 8, goneuron.Tanh),
    goneuron.Dense(8, 1, goneuron.Sigmoid),
)
```
*   **Rationale**: The `Tanh` activation in the hidden layer is selected to avoid the vanishing gradient or "dead neuron" issues sometimes encountered with ReLU in very small-scale networks. The `Sigmoid` output layer maps the result to a probability range [0, 1].

### 2. Training Orchestration
```go
model.Compile(goneuron.Adam(0.01), goneuron.MSE)
model.Fit(inputs, targets, 500, 4, goneuron.Logger(100))
```
*   **Mechanism**: The `Adam` optimizer provides adaptive learning rates, ensuring stable convergence. The training loop utilizes the `Logger` callback to provide periodic feedback on the Mean Squared Error (MSE).

## Execution

To run the XOR classification:

```bash
go run main.go
```
