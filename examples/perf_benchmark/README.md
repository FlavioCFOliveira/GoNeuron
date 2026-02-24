# Core Engine Performance Benchmark

This example provides high-level performance metrics for the GoNeuron engine, focusing on real-world training scenarios across different architectures (CNNs, RNNs, and MLPs).

## Technical Objectives

1.  **Architecture-Specific Benchmarks**: Analyzing how different layer types (e.g., `Conv2D` vs `LSTM`) impact overall training speed.
2.  **Memory Efficiency**: Monitoring allocation patterns and garbage collection pressure during high-throughput training.

## Key Implementation Details

### 1. Throughput Metrics
The benchmark reports samples per second (SPS), allowing developers to quantify the impact of architectural changes or optimizations on the library's core.

### 2. Multi-Core Utilization
Demonstrates the effectiveness of the library's internal worker pool and parallel batch training implementation on modern multi-core CPUs.

## Execution

```bash
go run main.go
```
