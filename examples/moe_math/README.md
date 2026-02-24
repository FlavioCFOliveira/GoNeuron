# Mixture of Experts (MoE) for Piecewise Functions

This example demonstrates the power of **Conditional Computation** using a **Mixture of Experts (MoE)** layer. It is trained to approximate complex, piecewise-defined mathematical functions by routing different inputs to specialized "expert" sub-networks.

## Technical Objectives

1.  **Sparse Gating**: Implementing a routing mechanism that only activates a subset of neurons for any given input.
2.  **Specialization**: Learning expert-specific weights that handle different regions of the input space.
3.  **Computational Efficiency**: Demonstrating how large-capacity models can remain efficient by keeping most parameters inactive during inference.

## Key Implementation Details

### 1. MoE Architecture
```go
model := goneuron.NewSequential(
    goneuron.Dense(1, 16, goneuron.ReLU),
    goneuron.MoE(16, 16, 4, 2), // 16 input/output, 4 experts, top-2 active
    goneuron.Dense(16, 1, goneuron.Linear),
)
```
*   **Mechanism**: The `MoE` layer contains 4 independent experts. For each input, a gating network calculates a probability distribution over these experts and activates only the top 2. This allows the model to partition the task, with different experts learning different segments of the target function.

### 2. Piecewise Learning
The example target function varies significantly across different input ranges, making it a perfect candidate for MoE where experts can "specialize" in specific input regimes.

## Execution

```bash
go run main.go
```
