# Iris Species Classification

This example implements a neural network classifier for the Iris flower dataset, a classic multi-class classification problem. It demonstrates end-to-end data handling, from automated downloading to model evaluation.

## Technical Objectives

1.  **Multi-Class Classification**: Implementing a Softmax output layer for 3-class identification.
2.  **Data Preprocessing**: Handling CSV data, performing normalization, and one-hot encoding for targets.
3.  **Cross-Entropy Optimization**: Utilizing categorical cross-entropy loss for probabilistic alignment.

## Key Implementation Details

### 1. Softmax Classification
```go
model := goneuron.NewSequential(
    goneuron.Dense(4, 16, goneuron.ReLU),
    goneuron.Dense(16, 3, goneuron.Softmax),
)
```
*   **Rationale**: The output layer uses 3 units with `Softmax` activation to produce a probability distribution across the three Iris species (Setosa, Versicolor, Virginica).

### 2. Data Pipeline
The example uses a custom utility to:
- Download the Iris dataset if not present.
- Normalize input features (sepal length/width, petal length/width) to improve convergence.
- Shuffle the dataset to ensure stochastic gradient updates are representative.

## Execution

Execute the classification script:

```bash
go run main.go
```
