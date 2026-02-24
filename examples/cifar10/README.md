# CIFAR-10 Image Classification

This example demonstrates the classification of 32x32 color images into 10 distinct classes (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck). It highlights the use of **Batch Normalization** to stabilize training in deep convolutional networks.

## Technical Objectives

1.  **Color Feature Extraction**: Processing 3-channel (RGB) input data through multiple convolutional blocks.
2.  **Internal Normalization**: Utilizing `BatchNorm2D` to mitigate internal covariate shift and accelerate convergence.
3.  **Deep Spatial Learning**: Managing deeper architectures with multiple pooling stages.

## Key Implementation Details

### 1. Batch-Normalized Convolutional Blocks
```go
model := goneuron.NewSequential(
    goneuron.Conv2D(3, 32, 3, 1, 1, goneuron.ReLU),
    goneuron.BatchNorm2D(32),
    goneuron.MaxPool2D(32, 2, 2, 0),
    // ... additional blocks ...
    goneuron.Flatten(),
    goneuron.Dense(..., 10, goneuron.LogSoftmax),
)
```
*   **Rationale**: `BatchNorm2D` normalizes the activations of each channel, allowing for higher learning rates and reducing sensitivity to weight initialization.

### 2. GPU Acceleration
Due to the computational intensity of CIFAR-10, this example is optimized for **Metal/GPU** execution on Apple Silicon, significantly reducing the per-epoch training time compared to CPU execution.

## Execution

```bash
go run main.go
```
