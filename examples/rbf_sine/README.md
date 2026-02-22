# RBF Network for Sine Wave Approximation

This example demonstrates using a Radial Basis Function (RBF) layer followed by a Dense layer to approximate a sine wave function.

## Architecture
- **RBF Layer**: 1 input, 50 centers, 50 outputs, Gaussian kernel (gamma=1.0).
- **Dense Layer**: 50 inputs, 1 output, Linear activation.

## How it works
RBF networks use radial basis functions as activation functions. Each unit in the RBF layer computes the distance from the input to its center. The output of the RBF layer is a non-linear transformation of these distances, which is then linearly combined by the subsequent Dense layer to produce the final prediction.

## Running the example
```bash
go run examples/rbf_sine/main.go
```
