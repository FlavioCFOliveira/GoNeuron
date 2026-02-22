# SLP for Iris Classification

This example demonstrates a Single-Layer Perceptron (SLP) for multi-class classification on the Iris dataset.

## Architecture
- **Dense Layer**: 4 inputs (sepal length, sepal width, petal length, petal width), 3 outputs (Iris-setosa, Iris-versicolor, Iris-virginica).
- **Activation**: Softmax.
- **Loss**: CrossEntropy.

## Running the example
```bash
go run examples/slp_iris/main.go
```
The example will download the Iris dataset if it's not present in the current directory.
