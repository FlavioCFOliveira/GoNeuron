# Neural Network Examples

This directory contains several examples demonstrating how to train neural networks with different architectures and configurations.

## Running Examples

Each example is a standalone Go program. Run them with:

```bash
go run ./cmd/<example-name>/main.go
```

## Available Examples

### 1. XOR (`cmd/xor/`)
Trains a simple network to solve the XOR problem.
- Architecture: 2 inputs → 3 hidden → 1 output
- Activation: ReLU (hidden), Sigmoid (output)
- Loss: MSE
- Optimizer: SGD

### 2. Iris Classification (`cmd/iris/`)
Multi-class classification on a simplified Iris dataset.
- Architecture: 4 inputs → 8 hidden → 6 hidden → 3 outputs
- Activation: ReLU (hidden), Sigmoid (output)
- Loss: MSE
- Optimizer: Adam
- Target: Classify 3 iris species

### 3. MLP Examples (`cmd/mlp/`)
Demonstrates different MLP configurations:
- Example 1: Small XOR network (2-4-1)
- Example 2: Larger XOR network (2-8-8-1)
- Example 3: Binary classification (4-8-4-1)

### 4. MNIST-style Classification (`cmd/mnist/`)
Simulated MNIST digit classification with 784 inputs (28x28 pixels).
- Architecture: 784 → 128 → 64 → 10 outputs
- Activation: ReLU (hidden), Sigmoid (output)
- Loss: MSE
- Optimizer: Adam

### 5. Regression Examples (`cmd/regression/`)
Demonstrates regression with different loss functions:
- Example 1: Linear regression with normalized data
- Example 2: Non-linear regression (y = x²)
- Example 3: Multi-input regression (z = x + y)
- Example 4: Robust regression with Huber loss (outlier-resistant)

### 6. LSTM Examples (`cmd/lstm/`)
Demonstrates recurrent neural networks for sequence modeling:
- Example 1: Sequential prediction (y_t = x_t + 0.5*y_{t-1})
- Example 2: XOR sequence (requires memory)
- Example 3: Sinusoidal sequence prediction

### 7. CNN Examples (`cmd/cnn/`)
Demonstrates convolutional neural networks for spatial data:
- Example 1: 1D convolution for sequence features
- Example 2: 2D convolution for 8x8 images
- Example 3: Simple CNN for binary classification (100% accuracy)

### 8. Combined Network Examples (`cmd/combined/`)
Demonstrates networks combining multiple layer types:
- Example 1: Conv + LSTM for sequence of images
- Example 2: CNN for feature extraction + Dense for classification

## Common Patterns

### Creating a Network with Dense Layers
```go
l1 := layer.NewDense(in, hidden, activations.ReLU{})
l2 := layer.NewDense(hidden, out, activations.Sigmoid{})

network := net.New(
    []layer.Layer{l1, l2},
    loss.MSE{},
    opt.NewAdam(0.01),
)
```

### Creating a Network with LSTM
```go
lstm := layer.NewLSTM(inSize, outSize)
output := layer.NewDense(outSize, finalOut, activations.Sigmoid{})

network := net.New(
    []layer.Layer{lstm, output},
    loss.MSE{},
    opt.NewAdam(0.05),
)

// For each sequence:
lstm.Reset() // Reset state before new sequence
for t := 0; t < len(sequence); t++ {
    lstm.Forward(input[t])
}
```

### Training
```go
for epoch := 0; epoch < 5000; epoch++ {
    totalLoss := 0.0
    for i := range trainX {
        loss := network.Train(trainX[i], trainY[i])
        totalLoss += loss
    }
    if epoch%500 == 0 {
        fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(trainX)))
    }
}
```

### Saving and Loading
```go
// Save
network.Save("model.bin")

// Load
loadedNetwork, _, err := net.Load("model.bin")
```

### LSTM State Management
```go
// Reset LSTM state before processing a new sequence
lstm.Reset()

// Forward pass through sequence
for t := 0; t < len(sequence); t++ {
    output := lstm.Forward(sequence[t])
}
```

## Architecture Guidelines

| Task Type | Hidden Layers | Activation |
|-----------|---------------|------------|
| Simple XOR | 1 layer | ReLU/Sigmoid |
| Multi-class | 2+ layers | ReLU (hidden), Sigmoid (output) |
| Regression | 2+ layers | ReLU/Tanh (hidden), Sigmoid (output) |
| Robust Reg. | 2+ layers | ReLU/Tanh (hidden), Sigmoid (output) |

### Output Layer Selection
- **Binary classification**: Sigmoid + BinaryCrossEntropy
- **Multi-class**: Sigmoid + MSE (or CrossEntropy)
- **Regression**: Sigmoid/Tanh (for bounded outputs) or ReLU (for unbounded)

### Optimizer Selection
- **SGD**: Simple, good for basic problems
- **Adam**: Faster convergence, better for complex problems

### LSTM Layer Selection
| Task Type | Hidden Size | Optimizer |
|-----------|-------------|-----------|
| Simple sequence | 8-16 | Adam (0.01-0.05) |
| Complex sequence | 32-128 | Adam (0.001-0.01) |
| Long sequences | 16-64 | Adam with gradient clipping |

### CNN Layer Selection
| Task Type | Filters | Kernel Size | Optimizer |
|-----------|---------|-------------|-----------|
| Simple 1D | 4-16 | 3-5 | Adam (0.01-0.05) |
| 2D images | 8-64 | 3-5 | Adam (0.001-0.01) |
| Complex | 32-128 | 3 | Adam + pooling |
