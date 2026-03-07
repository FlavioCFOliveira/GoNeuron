#!/usr/bin/env python3
"""Fix remaining Conv2D and other layer return value issues."""

import re

def fix_conv2d(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix 1: ForwardWithArena signature
    content = re.sub(
        r'func \(c \*Conv2D\) ForwardWithArena\(input \[\]float32, arena \*\[\]float32, offset \*int\) \[\]float32 \{',
        r'func (c *Conv2D) ForwardWithArena(input []float32, arena *[]float32, offset *int) ([]float32, error) {',
        content
    )

    # Fix 2: Forward signature
    content = re.sub(
        r'func \(c \*Conv2D\) Forward\(input \[\]float32\) \[\]float32 \{',
        r'func (c *Conv2D) Forward(input []float32) ([]float32, error) {',
        content
    )

    # Fix 3: ForwardBatch signature
    content = re.sub(
        r'func \(c \*Conv2D\) ForwardBatch\(input \[\]float32, batchSize int\) \[\]float32 \{',
        r'func (c *Conv2D) ForwardBatch(input []float32, batchSize int) ([]float32, error) {',
        content
    )

    # Fix 4: ForwardBatchWithArena signature
    content = re.sub(
        r'func \(c \*Conv2D\) ForwardBatchWithArena\(input \[\]float32, batchSize int, arena \*\[\]float32, offset \*int\) \[\]float32 \{',
        r'func (c *Conv2D) ForwardBatchWithArena(input []float32, batchSize int, arena *[]float32, offset *int) ([]float32, error) {',
        content
    )

    # Fix 5: return c.outputBuf[:requiredOutput], nil in ForwardWithArena
    # (already correct in Metal path, check CPU path)

    # Fix 6: Fix Forward calling ForwardWithArena
    content = re.sub(
        r'return c\.ForwardWithArena\(input, nil, nil\)',
        r'return c.ForwardWithArena(input, nil, nil)',
        content
    )

    # Fix 7: Fix ForwardBatch calling ForwardBatchWithArena
    content = re.sub(
        r'return c\.ForwardBatchWithArena\(input, batchSize, nil, nil\)',
        r'return c.ForwardBatchWithArena(input, batchSize, nil, nil)',
        content
    )

    # Fix 8: Fix duplicate err in ForwardBatchWithArena loop
    content = re.sub(
        r'out, err, err := c\.ForwardWithArena',
        r'out, err := c.ForwardWithArena',
        content
    )

    # Fix 9: Fix return at end of ForwardBatchWithArena
    content = re.sub(
        r'return c\.outputBuf\[:batchSize\*outSize\], nil\n\}\n\n\n// Backward',
        r'return c.outputBuf[:batchSize*outSize], nil\n}\n\n\n// Backward',
        content
    )

    # Fix 10: Fix NewConv2D return
    content = re.sub(
        r'return c, nil\n\}\n\n// Build',
        r'return c\n}\n\n// Build',
        content
    )

    # Fix 11: Fix NamedParams return
    content = re.sub(
        r'return named, nil\n\}\n\n// OutputBuf',
        r'return named\n}\n\n// OutputBuf',
        content
    )

    # Fix 12: Fix Clone and LightweightClone returns
    content = re.sub(
        r'return newC, nil\n\}\n\nfunc \(c \*Conv2D\) LightweightClone',
        r'return newC\n}\n\nfunc (c *Conv2D) LightweightClone',
        content
    )

    # Fix 13: Fix second LightweightClone return
    content = re.sub(
        r'return newC, nil\n\}\n\n// InSize',
        r'return newC\n}\n\n// InSize',
        content
    )

    # Fix 14: Fix return nil, nil in ForwardBatchWithArena
    content = re.sub(
        r'return nil, nil\n\}\n\nfunc \(c \*Conv2D\) Backward',
        r'return nil, fmt.Errorf("no saved input for backward")\n}\n\nfunc (c *Conv2D) Backward',
        content
    )

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed: {filepath}")

def fix_conv1d(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix return statements that need , nil
    # Forward returns ([]float32, error)
    content = re.sub(
        r'return c\.ForwardWithArena\(x, nil, nil\)\n\}\n\n// ForwardWithArena',
        r'return c.ForwardWithArena(x, nil, nil)\n}\n\n// ForwardWithArena',
        content
    )

    # Fix return at end of ForwardWithArena
    content = re.sub(
        r'return c\.outputBuf\[:outputSize\]\n\}\n\n// conv1DForward',
        r'return c.outputBuf[:outputSize], nil\n}\n\n// conv1DForward',
        content
    )

    # Fix return at end of BackwardInPlace
    content = re.sub(
        r'return gradIn\n\}\n\n// conv1DBackward',
        r'return gradIn, nil\n}\n\n// conv1DBackward',
        content
    )

    # Fix AccumulateBackward
    content = re.sub(
        r'return c\.BackwardInPlace\(grad, nil\)\n\}\n\n// SetTraining',
        r'return c.BackwardInPlace(grad, nil)\n}\n\n// SetTraining',
        content
    )

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed: {filepath}")

def fix_batchnorm2d(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix NamedParams return
    content = re.sub(
        r'return named, nil\n\}\n\n// OutputBuf',
        r'return named\n}\n\n// OutputBuf',
        content
    )

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed: {filepath}")

if __name__ == "__main__":
    fix_conv2d("internal/layer/conv2d.go")
    fix_conv1d("internal/layer/conv1d.go")
    fix_batchnorm2d("internal/layer/batchnorm2d.go")
    print("\nDone!")
