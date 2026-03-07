#!/usr/bin/env python3
"""
Script to update layer method signatures to return errors.
This updates the function signatures from:
    func (l *Layer) Method(...) []float32 {
to:
    func (l *Layer) Method(...) ([]float32, error) {
"""

import re
import sys

FILES = [
    "internal/layer/attention.go",
    "internal/layer/avgpool2d.go",
    "internal/layer/batchnorm2d.go",
    "internal/layer/bidirectional.go",
    "internal/layer/dropout.go",
    "internal/layer/embedding.go",
    "internal/layer/gru.go",
    "internal/layer/layernorm.go",
    "internal/layer/lstm.go",
    "internal/layer/maxpool2d.go",
    "internal/layer/moe.go",
    "internal/layer/rbf.go",
    "internal/layer/reshape.go",
    "internal/layer/rmsnorm.go",
    "internal/layer/swiglu.go",
    "internal/layer/transformer.go",
    "internal/layer/unroller.go",
    "internal/layer/conv1d.go",
    "internal/layer/conv2d.go",
]

# Patterns to match function signatures
PATTERNS = [
    # Forward
    (r'func \((\w+) \*(\w+)\) Forward\(x \[\]float32\) \[\]float32 \{',
     r'func (\1 *\2) Forward(x []float32) ([]float32, error) {'),

    # Backward
    (r'func \((\w+) \*(\w+)\) Backward\(grad \[\]float32\) \[\]float32 \{',
     r'func (\1 *\2) Backward(grad []float32) ([]float32, error) {'),

    # AccumulateBackward
    (r'func \((\w+) \*(\w+)\) AccumulateBackward\(grad \[\]float32\) \[\]float32 \{',
     r'func (\1 *\2) AccumulateBackward(grad []float32) ([]float32, error) {'),

    # ForwardBatch
    (r'func \((\w+) \*(\w+)\) ForwardBatch\(x \[\]float32, batchSize int\) \[\]float32 \{',
     r'func (\1 *\2) ForwardBatch(x []float32, batchSize int) ([]float32, error) {'),

    # BackwardBatch
    (r'func \((\w+) \*(\w+)\) BackwardBatch\(grad \[\]float32, batchSize int\) \[\]float32 \{',
     r'func (\1 *\2) BackwardBatch(grad []float32, batchSize int) ([]float32, error) {'),

    # AccumulateBackwardBatch
    (r'func \((\w+) \*(\w+)\) AccumulateBackwardBatch\(grad \[\]float32, batchSize int\) \[\]float32 \{',
     r'func (\1 *\2) AccumulateBackwardBatch(grad []float32, batchSize int) ([]float32, error) {'),

    # ForwardWithArena
    (r'func \((\w+) \*(\w+)\) ForwardWithArena\(x \[\]float32, arena \*\[\]float32, offset \*int\) \[\]float32 \{',
     r'func (\1 *\2) ForwardWithArena(x []float32, arena *[]float32, offset *int) ([]float32, error) {'),

    # ForwardBatchWithArena
    (r'func \((\w+) \*(\w+)\) ForwardBatchWithArena\(x \[\]float32, batchSize int, arena \*\[\]float32, offset \*int\) \[\]float32 \{',
     r'func (\1 *\2) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) ([]float32, error) {'),
]

def process_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original = content

        for pattern, replacement in PATTERNS:
            content = re.sub(pattern, replacement, content)

        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Updated: {filepath}")
        else:
            print(f"No changes: {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    for filepath in FILES:
        process_file(filepath)
    print("\nSignature updates complete!")
    print("Note: You still need to:")
    print("1. Add input validation and return errors")
    print("2. Update return statements to return values with nil error")
    print("3. Handle error propagation in internal calls")
