# GoNeuron

[![Go Version](https://img.shields.io/badge/go-1.23+-blue.svg)](https://golang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A pure Go neural network library with no external dependencies. Built for high performance, human-friendly API, and hardware acceleration.

## Features

- **Human-Friendly API** - High-level Keras-style `Sequential` model via the `goneuron` package.
- **Pure Go** - No external BLAS or C dependencies.
- **Hardware Acceleration** - Native Metal/MPS support for Apple Silicon (macOS).
- **Multiple Layer Types** - Dense, Conv2D, LSTM, GRU, Bidirectional, Transformer, Attention, and more.
- **Zero-Allocation Core** - Performance optimized with pre-allocated contiguous memory and arena-based state saving.
- **Metal 2.0 Acceleration** - Full backend support for Conv2D, Pooling, Normalization, and Transformer blocks on Apple Silicon.
- **High-Performance Parallelism** - Built-in worker pool with thread pinning (LockOSThread) for stable GPU batch training.

## Installation

```bash
go get github.com/FlavioCFOliveira/GoNeuron
```

## Quick Start

The `goneuron` package provides a simplified entry point for all library features.

```go
package main

import (
    "fmt"
    "github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

func main() {
    // 1. Create a Sequential model
    model := goneuron.NewSequential(
        goneuron.Dense(2, 4, goneuron.Tanh),
        goneuron.Dense(4, 1, goneuron.Sigmoid),
    )

    // 2. Configure training
    model.Compile(goneuron.Adam(0.05), goneuron.MSE)

    // 3. Train the model
    inputs := [][]float32{{0,0}, {0,1}, {1,0}, {1,1}}
    targets := [][]float32{{0}, {1}, {1}, {0}}

    fmt.Println("Training...")
    model.Fit(inputs, targets, 500, 1, goneuron.Logger(100))

    // 4. Inference
    prediction := model.Predict([]float32{0, 1})
    fmt.Printf("Input: [0, 1] -> Prediction: %.4f\n", prediction[0])
}
```

---

## Component Usage

### 1. Model Architectures

#### Multi-Layer Perceptron (MLP)
Standard feed-forward network for classification or regression.
```go
model := goneuron.NewSequential(
    goneuron.Dense(784, 128, goneuron.ReLU),
    goneuron.Dropout(0.2, 128),
    goneuron.Dense(128, 10, goneuron.Softmax),
)
```

#### Convolutional Neural Network (CNN)
Optimized for image processing.
```go
model := goneuron.NewSequential(
    goneuron.Conv2D(3, 16, 3, 1, 1, goneuron.ReLU), // InChannels, OutChannels, Kernel, Stride, Padding
    goneuron.MaxPool2D(16, 2, 2, 0),
    goneuron.Flatten(),
    goneuron.Dense(16*14*14, 10, goneuron.Softmax),
)
```

#### Recurrent Neural Networks (LSTM / GRU)
For time-series and sequential data.
```go
model := goneuron.NewSequential(
    goneuron.SequenceUnroller(goneuron.LSTM(inputDim, hiddenDim), seqLen, false),
    goneuron.Dense(hiddenDim, outputDim, goneuron.Linear),
)
```

#### Mixture of Experts (MoE)
Dynamic routing to specialized sub-networks.
```go
model := goneuron.NewSequential(
    goneuron.Dense(1, 16, goneuron.ReLU),
    goneuron.MoE(16, 16, 4, 2), // 4 experts, activate top-2
    goneuron.Dense(16, 1, goneuron.Linear),
)
```

#### Transformers & Attention
State-of-the-art NLP architectures.
```go
model := goneuron.NewSequential(
    goneuron.Embedding(vocabSize, embedDim),
    goneuron.PositionalEncoding(maxLen, embedDim),
    goneuron.TransformerBlock(embedDim, numHeads, maxLen, ffDim, true), // causal=true for GPT
    goneuron.CLSPooling(maxLen, embedDim),
    goneuron.Dense(embedDim, vocabSize, goneuron.LogSoftmax),
)
```

### 2. Training and Evaluation

#### Compile and Fit
Connect an optimizer and loss function, then run the automated training loop.
```go
model.Compile(goneuron.Adam(0.001), goneuron.CrossEntropy)

// Fit supports batch size and callbacks
model.Fit(xTrain, yTrain, epochs, batchSize,
    goneuron.Logger(1),
    goneuron.EarlyStopping(3, 0.001),
    goneuron.ModelCheckpoint("best_model.gob"),
)
```

#### Prediction and Metrics
```go
// Single sample prediction
output := model.Predict(sample)

// Evaluate on test set
loss := model.Evaluate(xTest, yTest)
```

### 3. Model Persistence
Save and reload models including weights and architecture.

```go
// Save
err := model.Save("model.gob")

// Load
loadedModel, lossFn, err := goneuron.Load("model.gob")
```

#### GGUF Serialization
GoNeuron supports GGUF v3 serialization for interoperability with the broader LLM ecosystem (e.g., llama.cpp).

```go
import "github.com/FlavioCFOliveira/GoNeuron/internal/net"

// Save as standard F32 GGUF
err := model.SaveGGUF("model.gguf")

// Save as compressed F16 GGUF
err := model.SaveGGUFExt("model_f16.gguf", net.GGMLTypeF16)
```

### 4. Advanced Components

| Component | Description |
|-----------|-------------|
| `goneuron.Bidirectional(layer)` | Wraps an RNN to process sequences in both directions. |
| `goneuron.GlobalAttention(size)` | Global context attention mechanism. |
| `goneuron.ReduceLROnPlateau(...)` | Callback to reduce learning rate when progress stalls. |
| `goneuron.LeakyReLU(alpha)` | Activation that prevents dying neurons. |
| `goneuron.BatchNorm2D(channels)`| Normalization for faster CNN training. |

---

## Performance & Efficiency

GoNeuron is engineered for maximum throughput by minimizing Go's runtime overhead:

- **Zero-Allocation Training**: Sequential training loops achieve `0 allocs/op`. All intermediate states (activations, gradients, pre-activations) are managed via a contiguous **Arena Buffer** using offsets instead of slice allocations.
- **Persistent Worker Pool**: Parallel training reuses a pool of worker networks. This avoids the heavy cost of cloning the model architecture for every batch, reducing garbage collection pressure by over 99%.
- **Contiguous Memory Layout**: Model parameters and gradients are stored in unified flat buffers, improving cache locality and enabling faster hardware transfers.

---

## Hardware Acceleration
GoNeuron automatically detects and uses Apple Silicon GPU (Metal/MPS) if available on macOS. To manually set the device:

```go
import "github.com/FlavioCFOliveira/GoNeuron/internal/layer"

device := layer.GetDefaultDevice() // Usually GPU on Mac, CPU elsewhere
model.SetDevice(device)
```

## Documentation & PDS
For a detailed technical overview of the implementation and roadmap, see the [Software Development Plan (PDS)](PDS.md).

## Examples
Explore the `examples/` directory for full implementations:
- `mnist`: CNN for digit classification.
- `mini_transformer`: Sentiment analysis.
- `gan_mnist`: Image generation.
- `stock_prediction`: Financial time-series forecasting.
- `portuguese_embeddings`: NLP specific to PT-PT.
- `GGUF_EXAMPLES.md`: Tutorials for model export and interoperability.

## License
MIT License
