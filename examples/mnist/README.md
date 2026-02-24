# MNIST Handwritten Digit Classification

This example implements a high-performance Convolutional Neural Network (CNN) to classify the MNIST dataset (70,000 grayscale images of digits 0-9). It serves as a benchmark for computer vision capabilities and hardware acceleration within the GoNeuron library.

## Technical Objectives

1.  **Convolutional Feature Extraction**: Utilizing `Conv2D` and `MaxPool2D` layers for spatial hierarchy learning.
2.  **Hardware Acceleration**: Demonstrating the performance gains of **Apple Silicon (Metal/MPS)** integration.
3.  **Advanced Training Callbacks**: Implementing `ReduceLROnPlateau`, `EarlyStopping`, and `ModelCheckpoint` for production-grade training stability.

## Key Implementation Details

### 1. CNN Architecture
```go
model := goneuron.NewSequential(
    goneuron.Conv2D(1, 32, 3, 1, 1, goneuron.ReLU),
    goneuron.MaxPool2D(32, 2, 2, 0),
    goneuron.Conv2D(32, 64, 3, 1, 1, goneuron.ReLU),
    goneuron.MaxPool2D(64, 2, 2, 0),
    goneuron.Flatten(),
    goneuron.Dense(64*7*7, 128, goneuron.ReLU),
    goneuron.Dropout(0.5, 128),
    goneuron.Dense(128, 10, goneuron.LogSoftmax),
)
```
*   **Design**: A deep feature extractor followed by a classification head. The `LogSoftmax` output combined with `NLLLoss` provides numerically stable probability estimation for multi-class classification.

### 2. Hardware and Optimization
The implementation automatically detects macOS environments to enable Metal acceleration. It also uses a sophisticated training loop with:
- **EarlyStopping**: Halts training when validation loss stops improving, preventing overfitting.
- **ModelCheckpoint**: Periodically saves the best model state to disk.

## Execution

Ensure the MNIST dataset is downloaded (the script provides automated fetching), then run:

```bash
go run main.go
```
