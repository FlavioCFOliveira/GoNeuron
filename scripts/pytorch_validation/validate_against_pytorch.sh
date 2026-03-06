#!/bin/bash
#
# PyTorch Validation Runner
#
# This script:
# 1. Generates PyTorch reference outputs
# 2. Runs Go validation tests against those references
#
# Usage: ./validate_against_pytorch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TEST_DATA_DIR="$SCRIPT_DIR/test_data/pytorch_refs"

echo "=================================="
echo "PyTorch Validation for GoNeuron"
echo "=================================="
echo ""

# Check if Python and PyTorch are available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

if ! python3 -c "import torch" 2> /dev/null; then
    echo "❌ Error: PyTorch not installed"
    echo "   Install with: pip install torch numpy"
    exit 1
fi

echo "✓ Python and PyTorch found"
echo ""

# Step 1: Generate reference data
echo "Step 1: Generating PyTorch reference data..."
cd "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/generate_references.py" --output-dir "$TEST_DATA_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate reference data"
    exit 1
fi

echo ""

# Step 2: Run Go validation tests
echo "Step 2: Running Go validation tests..."
cd "$PROJECT_ROOT"
go test ./internal/validation/... -v -run TestPyTorchValidation

echo ""
echo "=================================="
echo "Validation complete!"
echo "=================================="
