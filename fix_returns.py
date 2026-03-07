#!/usr/bin/env python3
"""
Fix return statements in layer files to return errors properly.
"""

import re
import os

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Fix 1: return nil -> return nil, nil (in error-returning functions)
    # This pattern matches return nil that should return nil, nil
    content = re.sub(
        r'(func \(.*?\) \w+\(.*\) \(\[\]float32, error\) \{[\s\S]*?\breturn\b) nil\s*\n',
        r'\1 nil, nil\n',
        content
    )

    # Fix 2: return gradIn -> return gradIn, nil (for []float32 returns)
    content = re.sub(
        r'\breturn\s+(\w+)\s*\n\s*\}',
        r'return \1, nil\n}',
        content
    )

    # Fix 3: return a\.outputBuf -> return a.outputBuf, nil
    content = re.sub(
        r'\breturn\s+(a\.\w+Buf\[.*?\])\s*\n',
        r'return \1, nil\n',
        content
    )

    # Fix 4: return output -> return output, nil
    content = re.sub(
        r'\breturn\s+(output\[.*?\])\s*\n',
        r'return \1, nil\n',
        content
    )

    # Fix 5: return c.outputBuf -> return c.outputBuf, nil
    content = re.sub(
        r'\breturn\s+(c\.\w+Buf\[.*?\])\s*\n',
        r'return \1, nil\n',
        content
    )

    # Fix 6: return g.outputBuf -> return g.outputBuf, nil
    content = re.sub(
        r'\breturn\s+(g\.\w+Buf\[.*?\])\s*\n',
        r'return \1, nil\n',
        content
    )

    # Fix 7: return l.outputBuf -> return l.outputBuf, nil
    content = re.sub(
        r'\breturn\s+(l\.\w+Buf\[.*?\])\s*\n',
        r'return \1, nil\n',
        content
    )

    # Fix 8: return m.outputBuf -> return m.outputBuf, nil
    content = re.sub(
        r'\breturn\s+(m\.\w+Buf\[.*?\])\s*\n',
        r'return \1, nil\n',
        content
    )

    # Fix 9: return b.outputBuf -> return b.outputBuf, nil
    content = re.sub(
        r'\breturn\s+(b\.\w+Buf\[.*?\])\s*\n',
        r'return \1, nil\n',
        content
    )

    # Fix 10: return r.outputBuf -> return r.outputBuf, nil
    content = re.sub(
        r'\breturn\s+(r\.\w+Buf\[.*?\])\s*\n',
        r'return \1, nil\n',
        content
    )

    # Fix 11: Fix assignment mismatches like dx := a.Backward(...) -> dx, err := a.Backward(...)
    content = re.sub(
        r'(\s+)(\w+) := (\w+)\.Backward\(',
        r'\1\2, err := \3.Backward(',
        content
    )

    # Fix 12: Fix assignment mismatches like out := a.Forward(...) -> out, err := a.Forward(...)
    content = re.sub(
        r'(\s+)(\w+) := (\w+)\.Forward\(',
        r'\1\2, err := \3.Forward(',
        content
    )

    # Fix 13: Fix assignment mismatches like out := a.ForwardWithArena(...) -> out, err := a.ForwardWithArena(...)
    content = re.sub(
        r'(\s+)(\w+) := (\w+)\.ForwardWithArena\(',
        r'\1\2, err := \3.ForwardWithArena(',
        content
    )

    # Fix 14: Fix assignment mismatches like fOut = arenaLayer.ForwardWithArena -> fOut, err = arenaLayer.ForwardWithArena
    content = re.sub(
        r'(\s+)(\w+) = (\w+)\.ForwardWithArena\(',
        r'\1\2, err = \3.ForwardWithArena(',
        content
    )

    # Fix 15: Fix assignment mismatches like fOut = b.forward.Forward -> fOut, err = b.forward.Forward
    content = re.sub(
        r'(\s+)(\w+) = (\w+)\.Forward\(',
        r'\1\2, err = \3.Forward(',
        content
    )

    # Fix 16: Fix assignment mismatches like dx = b.backward.Backward -> dx, err = b.backward.Backward
    content = re.sub(
        r'(\s+)(\w+) = (\w+)\.Backward\(',
        r'\1\2, err = \3.Backward(',
        content
    )

    # Fix 17: return gradIn[:something] -> return gradIn[:something], nil
    content = re.sub(
        r'\breturn\s+(\w+\[[:\w]+\])\s*\n',
        r'return \1, nil\n',
        content
    )

    # Fix 18: Fix fmt.Errorf undefined by changing to errors.New or just nil, fmt.Errorf
    # But first let's check if fmt is imported

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
    else:
        print(f"No changes: {filepath}")

FILES = [
    "internal/layer/avgpool2d.go",
    "internal/layer/batchnorm2d.go",
    "internal/layer/bidirectional.go",
    "internal/layer/conv2d.go",
    "internal/layer/maxpool2d.go",
]

for filepath in FILES:
    if os.path.exists(filepath):
        fix_file(filepath)
    else:
        print(f"File not found: {filepath}")

print("\nDone! Please check for remaining issues manually.")
