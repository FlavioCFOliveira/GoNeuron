# PyTorch Validation Framework for GoNeuron

This framework validates GoNeuron's implementations against PyTorch reference outputs to ensure mathematical correctness and compatibility.

## Overview

The validation framework consists of:
- **Python script** (`generate_references.py`): Generates reference outputs from PyTorch
- **Go tests** (`pytorch_validation_test.go`): Validates GoNeuron against references
- **Shell script** (`validate_against_pytorch.sh`): Runs the full validation pipeline

## Supported Operations

### Layers
- Dense (Linear)
- Conv2D
- BatchNorm2D

### Activations
- ReLU
- Sigmoid
- Tanh
- LeakyReLU
- Softmax
- LogSoftmax

### Loss Functions
- MSE (Mean Squared Error)
- CrossEntropy

### Optimizers
- SGD
- Adam

## Requirements

```bash
# Python dependencies
pip install torch numpy

# Go dependencies (already in go.mod)
go mod download
```

## Usage

### Quick Start

Run the complete validation pipeline:

```bash
cd scripts/pytorch_validation
./validate_against_pytorch.sh
```

### Manual Steps

1. **Generate reference data:**
```bash
python generate_references.py --output-dir ./test_data/pytorch_refs
```

2. **Run Go validation tests:**
```bash
go test ./internal/validation/... -v -run TestPyTorchValidation
```

### Run Specific Tests

```bash
# Test only activation functions
go test ./internal/validation/... -v -run TestPyTorchValidation_Activations

# Test only loss functions
go test ./internal/validation/... -v -run TestPyTorchValidation_Loss

# Test a specific activation
go test ./internal/validation/... -v -run "TestPyTorchValidation_Activations/relu"
```

## Adding New Validation Tests

### 1. Python Side (Reference Generator)

Add a new test case in `generate_references.py`:

```python
def generate_my_operation_tests(output_dir: Path):
    """Generate reference outputs for MyOperation."""
    print("\n=== MyOperation Tests ===")

    # Create test inputs
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    # Run PyTorch operation
    y = my_torch_operation(x)

    # Save test case
    save_test_case(
        output_dir, "my_operation_test",
        inputs={"input": x.tolist()},
        outputs={"output": y.tolist()},
        metadata={
            "input_shape": [3],
            "output_shape": [3],
            "operation": "my_operation"
        }
    )
```

Then add your generator to `main()`:

```python
def main():
    # ... existing code ...
    generate_my_operation_tests(output_dir)
```

### 2. Go Side (Validation Test)

Add a test in `pytorch_validation_test.go`:

```go
func TestPyTorchValidation_MyOperation(t *testing.T) {
    t.Run("TestName", func(t *testing.T) {
        tc := skipIfNoTestData(t, "my_operation_test")

        input := tc.Inputs["input"].([]float32)
        expected := tc.Outputs["output"].([]float32)

        // Run GoNeuron operation
        output := myGoneuronOperation(input)

        // Compare
        if !slicesEqual(output, expected, 1e-5) {
            t.Errorf("MyOperation mismatch.\nGot:      %v\nExpected: %v", output, expected)
        }
    })
}
```

## Test Data Format

Each test case is stored in a directory structure:

```
test_data/pytorch_refs/
├── test_name/
│   ├── input.bin           # Binary float32 data
│   ├── weights.bin         # Binary float32 data
│   ├── output.bin          # Binary float32 data
│   └── metadata.json       # Test configuration
```

### Binary Format

Float32 values are stored as little-endian IEEE 754 binary32:
- 4 bytes per float
- Little-endian byte order

### Metadata Format

```json
{
  "name": "test_name",
  "inputs": {
    "input": {"shape": [1, 4], "type": "float32"}
  },
  "outputs": {
    "output": {"shape": [1, 3], "type": "float32"}
  },
  "operation": "dense_forward"
}
```

## Tolerance Values

Different operations use different tolerance values based on numerical precision:

| Operation | Tolerance | Reason |
|-----------|-----------|--------|
| Dense | 1e-4 | Matrix multiplication precision |
| Activations | 1e-5 | Direct computation |
| Loss functions | 1e-5 | Direct computation |
| Optimizers | 1e-3 - 1e-5 | State-dependent updates |

## CI Integration

To add validation to CI/CD:

```yaml
# .github/workflows/validation.yml
- name: PyTorch Validation
  run: |
    pip install torch numpy
    ./scripts/pytorch_validation/validate_against_pytorch.sh
```

## Troubleshooting

### "PyTorch reference data not found"

Run the reference generator first:
```bash
python scripts/pytorch_validation/generate_references.py
```

### "No module named 'torch'"

Install PyTorch:
```bash
pip install torch numpy
```

### Tolerance failures

If tests fail with small differences:
1. Check if PyTorch version changed (different implementations)
2. Increase tolerance slightly if mathematically equivalent
3. Investigate numerical differences

## Contributing

When adding new features to GoNeuron:
1. Add corresponding PyTorch reference generation
2. Add Go validation test
3. Update this README with new supported operations
