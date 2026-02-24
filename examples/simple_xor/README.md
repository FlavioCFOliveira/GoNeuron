# Simple XOR Implementation

A minimal implementation of the XOR problem, serving as a clean entry point for understanding the `goneuron` Sequential API.

## Technical Objectives

1.  **API Simplicity**: Showcasing the minimal code required to build and train a functional network.
2.  **Stochastic Gradient Descent**: Demonstrating basic training parameters and model compilation.

## Key Implementation Details

### 1. Model Definition
```go
model := goneuron.NewSequential(
    goneuron.Dense(2, 4, goneuron.Tanh),
    goneuron.Dense(4, 1, goneuron.Sigmoid),
)
```
*   **Design**: A compact 2-4-1 architecture that demonstrates the ability of a single hidden layer to solve the XOR logic gate problem.

### 2. Compilation and Training
```go
model.Compile(goneuron.Adam(0.1), goneuron.MSE)
model.Fit(inputs, targets, 200, 1)
```
*   **Mechanism**: Shows high-level model lifecycle management from construction to convergence.

## Execution

```bash
go run main.go
```
