# GoNeuron

[![Go Version](https://img.shields.io/badge/go-1.23+-blue.svg)](https://golang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A pure Go neural network library with no external dependencies. Built for performance, simplicity, and extensibility.

## Features

- **Pure Go** - No external BLAS or C dependencies
- **Multiple Layer Types** - Dense, Conv2D, LSTM, GRU, MaxPool2D, AvgPool2D, Dropout, BatchNorm2D, LayerNorm, Embedding, Flatten, PositionalEncoding, MultiHeadAttention, TransformerBlock, GlobalAveragePooling1D
- **Activation Functions** - ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, LogSoftmax
- **Loss Functions** - MSE, CrossEntropy, Huber, NLLLoss, BCELoss
- **Performance Optimized** - Zero-allocation patterns, contiguous memory, parallel batch processing, Metal/MPS hardware acceleration (macOS)
- **Model Persistence** - Save and load trained models
- **Advanced Examples** - CIFAR-10 CNN, MNIST GAN, Mini-Transformer for NLP

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
    trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    trainY := [][]float32{{0}, {1}, {1}, {0}}

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
                totalLoss += float64(network.Loss.Forward(pred, trainY[i]))
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

### 8. Using Transformer for NLP

```go
const (
    dModel   = 32
    numHeads = 4
    ffDim    = 64
    maxLen   = 10
    vocabSize = 1000
)

layers := []layer.Layer{
    layer.NewEmbedding(vocabSize, dModel),
    layer.NewPositionalEncoding(maxLen, dModel),
    layer.NewTransformerBlock(dModel, numHeads, maxLen, ffDim),
    layer.NewGlobalAveragePooling1D(maxLen, dModel),
    layer.NewDense(dModel, 2, activations.LogSoftmax{}),
}

network := net.New(layers, loss.NLLLoss{}, opt.NewAdam(0.001))
```

## Examples Directory

The `examples/` directory contains complete, runnable implementations:

- **cifar10**: Deep CNN for 10-class image classification.
- **gan_mnist**: Generative Adversarial Network for handwritten digit generation.
- **mini_transformer**: Transformer-based sentiment analysis.
- **mnist**: Basic handwritten digit classification.
- **stock_prediction**: Time-series forecasting using LSTM/GRU.
- **xor**: Simple neural network solving the XOR problem.

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
- **Contiguous memory** - Weights stored as row-major []float32 for cache efficiency
- **Parallel processing** - Automatic parallelization for batch training (>= 16 samples)
- **Hardware Acceleration** - Metal/MPS support for Apple Silicon (macOS)

## API Reference

### Network

```go
net.New(layers []Layer, loss Loss, optimizer Optimizer) *Network
network.Forward(x []float32) []float32
network.Train(x []float32, y []float32) float32  // Returns loss
network.TrainBatch(X [][]float32, Y [][]float32) float32
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

- Deterministic initialization (fixed seed=42)

## License

MIT License
