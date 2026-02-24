# Hardware Performance Benchmarking

A rigorous benchmarking suite designed to measure and compare the computational throughput of the **CPU** and **GPU (Metal)** backends on various network architectures.

## Technical Objectives

1.  **Latency Measurement**: Calculating the time per inference/training pass.
2.  **Throughput Analysis**: Measuring "tokens per second" or "samples per second" across different hardware configurations.
3.  **Scale Testing**: Evaluating how performance scales with batch size and model depth.

## Key Implementation Details

### 1. Backend Comparison
The benchmark iteratively runs the same workload on the `CPUDevice` and the `MetalDevice` (on Apple Silicon). It reports the speedup factor, identifying the specific batch sizes where GPU acceleration becomes most efficient.

### 2. Operational Profiling
Provides detailed timing for:
- Forward pass latency.
- Backward pass latency (gradient computation).
- Weight update (optimization) speed.

## Execution

```bash
go run main.go
```
