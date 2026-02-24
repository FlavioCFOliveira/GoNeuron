# Hardware Backend Validation

This utility ensures mathematical consistency and numerical stability across different hardware backends. It verifies that a model produces identical (or epsilon-equivalent) results when running on the **CPU** versus the **GPU**.

## Technical Objectives

1.  **Backend Parity**: Verifying that Metal kernels match the mathematical output of their Go/CPU counterparts.
2.  **Convergence Verification**: Ensuring that models converge at similar rates regardless of the selected execution device.

## Key Implementation Details

### 1. Epsilon-Consistency Checking
The tool runs identical inputs through the network on both backends and compares the resulting output tensors. It calculates the maximum absolute deviation, ensuring that floating-point variations stay within acceptable numerical bounds.

### 2. Gradient Correctness
By comparing the results of a training step, the tool validates that backpropagation and weight updates are correctly synchronized across hardware boundaries.

## Execution

```bash
go run main.go
```
