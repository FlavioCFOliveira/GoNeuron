# Hybrid CNN-LSTM for Stock Prediction

This model implements a hybrid architecture that combines **1D Convolutions (Conv1D)** with **LSTMs**. The CNN acts as a local feature extractor, while the LSTM handles long-term temporal dependencies.

## Technical Objectives

1.  **Multi-Scale Feature Extraction**: Using convolutions to identify local trends (e.g., weekly patterns) before sequential processing.
2.  **Hybrid Modeling**: Demonstrating the integration of spatial (1D) and temporal layers.

## Key Implementation Details

### 1. Hybrid Architecture
The input window is first processed by a `Conv2D` layer (operating as a 1D convolution over the time dimension) to extract filtered features. These features are then unrolled and fed into the LSTM.

### 2. Dimensionality Reduction
Pooling layers may be used after the convolution to reduce the sequence length, allowing the LSTM to focus on higher-level abstractions of the price movement.

## Execution

```bash
go run main.go
```
