# GoNeuron

[![Go Version](https://img.shields.io/badge/go-1.23+-blue.svg)](https://golang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A pure Go neural network library with no external dependencies. Built for performance, simplicity, and extensibility.

## Features

- **Pure Go** - No external BLAS or C dependencies
- **Multiple Layer Types** - Dense, Conv2D, LSTM
- **Activation Functions** - ReLU, Sigmoid, Tanh, LeakyReLU, Softmax
- **Loss Functions** - MSE, CrossEntropy, Huber
- **Optimizers** - SGD, Adam
- **Performance Optimized** - Zero-allocation patterns, contiguous memory, parallel batch processing
- **Model Persistence** - Save and load trained models

## Installation

```bash
go get github.com/FlavioCFOliveira/GoNeuron
```

## Quick Start

### Training a Neural Network

```go
package main

import (
    "fmt"

    "github.com/FlavioCFOliveira/GoNeuron/internal/activations"
    "github.com/FlavioCFOliveira/GoNeuron/internal/layer"
    "github.com/FlavioCFOliveira/GoNeuron/internal/loss"
    "github.com/FlavioCFOliveira/GoNeuron/internal/net"
    "github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

func main() {
    // Create a simple network: 2 inputs -> 3 hidden -> 1 output
    l1 := layer.NewDense(2, 3, activations.Tanh{})  // Hidden layer
    l2 := layer.NewDense(3, 1, activations.Sigmoid{}) // Output layer

    network := net.New(
        []layer.Layer{l1, l2},
        loss.MSE{},
        opt.SGD{LearningRate: 0.1},
    )

    // Training data (XOR problem)
    trainX := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    trainY := [][]float64{{0}, {1}, {1}, {0}}

    // Train for 5000 epochs
    for epoch := 0; epoch < 5000; epoch++ {
        for i := range trainX {
            network.Train(trainX[i], trainY[i])
        }
        if epoch%1000 == 0 {
            // Calculate average loss
            totalLoss := 0.0
            for i := range trainX {
                pred := network.Forward(trainX[i])
                totalLoss += network.Loss.Forward(pred, trainY[i])
            }
            fmt.Printf("Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(len(trainX)))
        }
    }

    // Test the network
    fmt.Println("\nTesting:")
    for i := range trainX {
        pred := network.Forward(trainX[i])
        fmt.Printf("Input: %v, Predicted: %.4f, Target: %v\n",
            trainX[i], pred[0], trainY[i][0])
    }
}
```

## Running Examples

```bash
# XOR training example
go run cmd/xor/main.go

# MLP example
go run cmd/mlp/main.go

# CNN example
go run cmd/cnn/main.go

# LSTM example
go run cmd/lstm/main.go

# All inference examples
go run cmd/inference/main.go
```

## Usage Guides

### 1. Training a Network

```go
// Create layers
inputLayer := layer.NewDense(inputSize, hiddenSize, activations.ReLU{})
hiddenLayer := layer.NewDense(hiddenSize, hiddenSize, activations.ReLU{})
outputLayer := layer.NewDense(hiddenSize, outputSize, activations.Sigmoid{})

// Create network
network := net.New(
    []layer.Layer{inputLayer, hiddenLayer, outputLayer},
    loss.MSE{},
    opt.NewAdam(0.001),  // Learning rate 0.001
)

// Train on batches
for epoch := 0; epoch < 100; epoch++ {
    // For single samples
    for i := range trainX {
        network.Train(trainX[i], trainY[i])
    }

    // Or for batch training (auto-parallelizes for batches >= 16)
    network.TrainBatch(trainX, trainY)
}
```

### 2. Making Inferences

```go
// Single sample inference
output := network.Forward(input)

// Batch inference
for _, sample := range samples {
    output := network.Forward(sample)
    // Process output
}
```

### 3. Saving and Loading Models

```go
// Save the network
err := network.Save("mymodel.bin")
if err != nil {
    log.Fatal("Error saving:", err)
}

// Load the network
loadedNetwork, loadedLoss, err := net.Load("mymodel.bin")
if err != nil {
    log.Fatal("Error loading:", err)
}

// Use the loaded network
output := loadedNetwork.Forward(input)
```

### 4. Different Activation Functions

```go
// ReLU - Good for hidden layers
layer.NewDense(10, 20, activations.ReLU{})

// Sigmoid - Good for output (probabilities 0-1)
layer.NewDense(20, 1, activations.Sigmoid{})

// Tanh - Good for hidden layers (outputs -1 to 1)
layer.NewDense(10, 20, activations.Tanh{})

// LeakyReLU - Prevents dying neurons
layer.NewDense(10, 20, activations.NewLeakyReLU(0.01))

// Softmax - For multi-class output (must use ActivateBatch)
softmax := activations.Softmax{}
output := softmax.ActivateBatch(rawScores)
```

### 5. Different Loss Functions

```go
// MSE - Mean Squared Error (regression)
network := net.New(layers, loss.MSE{}, opt.NewAdam(0.001))

// CrossEntropy - Classification (use with Softmax)
network := net.New(layers, loss.CrossEntropy{}, opt.NewAdam(0.001))

// Huber - Robust to outliers
network := net.New(layers, loss.Huber{Delta: 1.0}, opt.NewAdam(0.001))
```

### 6. Using LSTM for Sequence Data

```go
lstm := layer.NewLSTM(inputSize, hiddenSize)
outputLayer := layer.NewDense(hiddenSize, outputSize, activations.Sigmoid{})

// Process sequence
for _, input := range sequence {
    hidden := lstm.Forward(input)
    output := outputLayer.Forward(hidden)
    // Process output
}

// Reset state for new sequence
lstm.Reset()
```

### 7. Using Conv2D for Image Data

```go
// Create CNN for 8x8 grayscale images
conv := layer.NewConv2D(
    inChannels:  1,
    outChannels: 4,
    kernelSize: 3,
    stride:     1,
    padding:    1,
    activation: activations.ReLU{},
)

// After conv: 1x8x8 -> 4x8x8 = 256 values
// Flatten and feed to dense layer
dense := layer.NewDense(256, 10, activations.Sigmoid{})

// Combine in network
network := net.New(
    []layer.Layer{conv, dense},
    loss.CrossEntropy{},
    opt.NewAdam(0.001),
)
```

## Architecture

```
internal/
├── net/           # Core Network interface and training loop
├── layer/         # Layer implementations (Dense, Conv2D, LSTM)
├── activations/   # Activation functions
├── loss/          # Loss functions
└── opt/           # Optimizers (SGD, Adam)
```

## Performance

- **Zero-allocation patterns** - Pre-allocated buffers in layer constructors
- **Contiguous memory** - Weights stored as row-major []float64 for cache efficiency
- **Parallel processing** - Automatic parallelization for batch training (>= 16 samples)

## API Reference

### Network

```go
net.New(layers []Layer, loss Loss, optimizer Optimizer) *Network
network.Forward(x []float64) []float64
network.Train(x []float64, y []float64) float64  // Returns loss
network.TrainBatch(X [][]float64, Y [][]float64) float64
network.Save(filename string) error
network.Load(filename string) (*Network, Loss, error)
```

### Layers

```go
layer.NewDense(in, out int, activation Activation) *Dense
layer.NewConv2D(inC, outC, kernel, stride, padding int, activation Activation) *Conv2D
layer.NewLSTM(in, out int) *LSTM
```

### Optimizers

```go
opt.SGD{LearningRate: 0.1}
opt.NewAdam(lr float64) *Adam  // Default: beta1=0.9, beta2=0.999, eps=1e-8
```

## Limitations

- No pooling layers (MaxPool/AvgPool)
- No dropout layer
- No batch normalization
- No GPU acceleration (CPU only)
- Deterministic initialization (fixed seed=42)

## License

MIT License
