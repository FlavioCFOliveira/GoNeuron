# XOR Neural Network Example

This example demonstrates a simple 2-layer neural network solving the non-linearly separable XOR problem.

## Architecture

- **Input Layer**: 2 neurons (representing binary inputs A and B)
- **Hidden Layer**: 4 neurons with **Tanh** activation
- **Output Layer**: 1 neuron with **Sigmoid** activation

The network uses **Mean Squared Error (MSE)** loss and the **Adam** optimizer for efficient training.

## Why Tanh?

The XOR problem is non-linearly separable. Using a hidden layer with a non-linear activation function like Tanh allows the network to learn the required representation. Tanh is preferred over ReLU for this specific problem in our library to avoid the "dying ReLU" problem on such a small dataset.

## Running the example

From the project root:

```bash
go run examples/xor/main.go
```

## Expected Output

The network should converge within a few hundred epochs, resulting in predictions very close to the targets:

- `[0, 0] -> ~0.0`
- `[0, 1] -> ~1.0`
- `[1, 0] -> ~1.0`
- `[1, 1] -> ~0.0`
