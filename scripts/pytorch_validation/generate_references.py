#!/usr/bin/env python3
"""
PyTorch Reference Generator for GoNeuron Validation

This script generates reference outputs from PyTorch for various neural network
operations. These references are used by Go tests to validate GoNeuron's
implementations match PyTorch's behavior.

Usage:
    python generate_references.py [--output-dir ./test_data]

Requirements:
    pip install torch numpy
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Any, Callable

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
except ImportError:
    print("Error: PyTorch and NumPy are required.")
    print("Install with: pip install torch numpy")
    exit(1)


def float32_to_bytes(data: list[float]) -> bytes:
    """Convert list of floats to bytes."""
    return struct.pack(f'{len(data)}f', *data)


def save_test_case(output_dir: Path, name: str, inputs: dict[str, list], outputs: dict[str, list], metadata: dict[str, Any] = None):
    """Save a test case with inputs, outputs, and metadata."""
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs as binary float32
    for key, value in inputs.items():
        with open(case_dir / f"{key}.bin", "wb") as f:
            f.write(float32_to_bytes(value))

    # Save outputs as binary float32
    for key, value in outputs.items():
        with open(case_dir / f"{key}.bin", "wb") as f:
            f.write(float32_to_bytes(value))

    # Save metadata as JSON
    meta = {
        "name": name,
        "inputs": {k: {"shape": metadata.get(f"{k}_shape", [len(v)]), "type": "float32"}
                   for k, v in inputs.items()},
        "outputs": {k: {"shape": metadata.get(f"{k}_shape", [len(v)]), "type": "float32"}
                    for k, v in outputs.items()},
    }
    if metadata:
        meta.update({k: v for k, v in metadata.items() if not k.endswith("_shape")})

    with open(case_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Generated: {name}")


def generate_dense_layer_tests(output_dir: Path):
    """Generate reference outputs for Dense/FullyConnected layer."""
    print("\n=== Dense Layer Tests ===")

    # Test 1: Simple linear transformation
    torch.manual_seed(42)
    in_features, out_features = 4, 3

    linear = torch.nn.Linear(in_features, out_features, bias=True)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    y = linear(x)

    # Extract weights and biases
    weights = linear.weight.detach().numpy().flatten().tolist()  # [out, in] -> flatten
    biases = linear.bias.detach().numpy().tolist()

    save_test_case(
        output_dir, "dense_simple",
        inputs={"input": x.flatten().tolist(), "weights": weights, "biases": biases},
        outputs={"output": y.flatten().tolist()},
        metadata={
            "input_shape": [1, 4],
            "weights_shape": [3, 4],
            "biases_shape": [3],
            "output_shape": [1, 3],
            "operation": "dense_forward"
        }
    )

    # Test 2: Batch of samples
    batch_size = 3
    x2 = torch.randn(batch_size, in_features, dtype=torch.float32)
    y2 = linear(x2)

    save_test_case(
        output_dir, "dense_batch",
        inputs={"input": x2.flatten().tolist(), "weights": weights, "biases": biases},
        outputs={"output": y2.flatten().tolist()},
        metadata={
            "input_shape": [3, 4],
            "weights_shape": [3, 4],
            "biases_shape": [3],
            "output_shape": [3, 3],
            "operation": "dense_forward"
        }
    )


def generate_activation_tests(output_dir: Path):
    """Generate reference outputs for activation functions."""
    print("\n=== Activation Function Tests ===")

    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32)

    activations = {
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "leaky_relu": lambda t: F.leaky_relu(t, negative_slope=0.01),
        "elu": F.elu,
    }

    for name, func in activations.items():
        y = func(x)

        save_test_case(
            output_dir, f"activation_{name}",
            inputs={"input": x.tolist()},
            outputs={"output": y.tolist()},
            metadata={
                "input_shape": [7],
                "output_shape": [7],
                "operation": f"activation_{name}"
            }
        )


def generate_loss_tests(output_dir: Path):
    """Generate reference outputs for loss functions."""
    print("\n=== Loss Function Tests ===")

    # MSE Loss
    y_pred = torch.tensor([0.5, 1.2, 2.8, 3.1], dtype=torch.float32)
    y_true = torch.tensor([0.0, 1.0, 3.0, 3.0], dtype=torch.float32)
    mse_loss = F.mse_loss(y_pred, y_true)

    save_test_case(
        output_dir, "loss_mse",
        inputs={"y_pred": y_pred.tolist(), "y_true": y_true.tolist()},
        outputs={"loss": [mse_loss.item()]},
        metadata={
            "y_pred_shape": [4],
            "y_true_shape": [4],
            "operation": "loss_mse"
        }
    )

    # Cross Entropy Loss
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    ce_loss = F.cross_entropy(logits, targets)

    save_test_case(
        output_dir, "loss_cross_entropy",
        inputs={"logits": logits.flatten().tolist(), "targets": targets.tolist()},
        outputs={"loss": [ce_loss.item()]},
        metadata={
            "logits_shape": [2, 3],
            "targets_shape": [2],
            "operation": "loss_cross_entropy"
        }
    )

    # Softmax (for cross-entropy components)
    softmax_out = F.softmax(logits, dim=1)
    log_softmax_out = F.log_softmax(logits, dim=1)

    save_test_case(
        output_dir, "activation_softmax",
        inputs={"input": logits.flatten().tolist()},
        outputs={"output": softmax_out.flatten().tolist()},
        metadata={
            "input_shape": [2, 3],
            "output_shape": [2, 3],
            "operation": "activation_softmax"
        }
    )

    save_test_case(
        output_dir, "activation_log_softmax",
        inputs={"input": logits.flatten().tolist()},
        outputs={"output": log_softmax_out.flatten().tolist()},
        metadata={
            "input_shape": [2, 3],
            "output_shape": [2, 3],
            "operation": "activation_log_softmax"
        }
    )


def generate_conv2d_tests(output_dir: Path):
    """Generate reference outputs for Conv2D layer."""
    print("\n=== Conv2D Layer Tests ===")

    torch.manual_seed(42)

    # Simple 3x3 input, 1 channel
    # Input: 1x1x3x3 (NCHW)
    x = torch.tensor([[[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]]]], dtype=torch.float32)

    # Conv2d: 1 input channel, 1 output channel, 2x2 kernel
    conv = torch.nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
    conv.weight.data.fill_(1.0)  # All ones
    conv.bias.data.fill_(0.0)

    y = conv(x)

    save_test_case(
        output_dir, "conv2d_simple",
        inputs={"input": x.flatten().tolist()},
        outputs={"output": y.flatten().tolist()},
        metadata={
            "input_shape": [1, 1, 3, 3],  # NCHW
            "output_shape": list(y.shape),
            "kernel_size": 2,
            "stride": 1,
            "padding": 0,
            "operation": "conv2d_forward"
        }
    )

    # Test with stride and padding
    conv2 = torch.nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1, bias=False)
    y2 = conv2(x)

    save_test_case(
        output_dir, "conv2d_stride_pad",
        inputs={"input": x.flatten().tolist()},
        outputs={"output": y2.flatten().tolist()},
        metadata={
            "input_shape": [1, 1, 3, 3],
            "output_shape": list(y2.shape),
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "operation": "conv2d_forward"
        }
    )


def generate_batchnorm_tests(output_dir: Path):
    """Generate reference outputs for BatchNorm."""
    print("\n=== BatchNorm Tests ===")

    torch.manual_seed(42)

    # Input: 2x3x4 (NCHW) - 2 samples, 3 channels, 4x4 spatial
    x = torch.randn(2, 3, 4, 4, dtype=torch.float32)

    bn = torch.nn.BatchNorm2d(3, affine=True, track_running_stats=False)
    bn.eval()  # Use eval mode for deterministic output

    y = bn(x)

    save_test_case(
        output_dir, "batchnorm2d_eval",
        inputs={"input": x.flatten().tolist()},
        outputs={"output": y.flatten().tolist()},
        metadata={
            "input_shape": [2, 3, 4, 4],
            "output_shape": list(y.shape),
            "num_features": 3,
            "operation": "batchnorm2d_forward"
        }
    )


def generate_optimizer_tests(output_dir: Path):
    """Generate reference outputs for optimizer steps."""
    print("\n=== Optimizer Tests ===")

    torch.manual_seed(42)

    # SGD test
    param = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
    grad = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    optimizer = torch.optim.SGD([param], lr=0.1, momentum=0.9)
    param.grad = grad
    optimizer.step()

    save_test_case(
        output_dir, "optimizer_sgd_step",
        inputs={
            "param": [1.0, 2.0, 3.0],
            "grad": [0.1, 0.2, 0.3],
            "lr": [0.1],
            "momentum": [0.9]
        },
        outputs={"param_updated": param.detach().tolist()},
        metadata={
            "param_shape": [3],
            "grad_shape": [3],
            "operation": "optimizer_sgd_step"
        }
    )

    # Adam test
    param2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
    grad2 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    optimizer2 = torch.optim.Adam([param2], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    param2.grad = grad2
    optimizer2.step()

    save_test_case(
        output_dir, "optimizer_adam_step",
        inputs={
            "param": [1.0, 2.0, 3.0],
            "grad": [0.1, 0.2, 0.3],
            "lr": [0.001],
            "beta1": [0.9],
            "beta2": [0.999],
            "eps": [1e-8]
        },
        outputs={"param_updated": param2.detach().tolist()},
        metadata={
            "param_shape": [3],
            "grad_shape": [3],
            "operation": "optimizer_adam_step"
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch reference outputs for GoNeuron validation")
    parser.add_argument("--output-dir", type=str, default="./test_data/pytorch_refs",
                        help="Directory to save reference outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating PyTorch reference outputs to: {output_dir.absolute()}")
    print(f"PyTorch version: {torch.__version__}")

    generate_dense_layer_tests(output_dir)
    generate_activation_tests(output_dir)
    generate_loss_tests(output_dir)
    generate_conv2d_tests(output_dir)
    generate_batchnorm_tests(output_dir)
    generate_optimizer_tests(output_dir)

    print(f"\n✓ Reference data generated successfully in: {output_dir.absolute()}")
    print("\nTo run Go validation tests:")
    print(f"  go test ./... -run TestPyTorchValidation")


if __name__ == "__main__":
    main()
