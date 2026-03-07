# ML Training Validator - Memory

## Critical Issues Found

### 1. MoE Layer Crash (CRITICAL)
- **Location:** `examples/moe_math/main.go`, `internal/layer/moe.go`
- **Error:** `panic: runtime error: index out of range [2] with length 2`
- **Cause:** Dimension mismatch in Dense.Backward when using MoE layer in Sequential model
- **Status:** Needs investigation and fix

### 2. Autoencoder Not Converging (HIGH)
- **Location:** `examples/autoencoder_synthetic/main.go`
- **Issue:** Loss stuck at 0.098, encoder outputs all zeros
- **Cause:** Sigmoid output with unnormalized data, shallow architecture
- **Fix:** Change output activation to Tanh/Linear, add normalization

## Convergence Patterns Observed

| Problem Type | Typical Epochs to Converge | Notes |
|--------------|---------------------------|-------|
| XOR (2-4-1) | 500-1000 | Tanh + Sigmoid works well |
| Iris SLP | 1000+ | Single layer has limited capacity |
| Synthetic Regression | 500-1000 | Huber loss helps with outliers |
| RBF Sine | 1000 | Good for function approximation |

## Key Findings

1. **Good Examples:**
   - `simple_xor/` - Fast, reliable convergence
   - `xor/` - Full validation with tolerance check
   - `regression_synthetic/` - Good regression example
   - `rbf_sine/` - RBF networks work well
   - `csv_training/` - CSV loading works

2. **Dataset-Dependent Examples:**
   - MNIST examples need dataset download
   - Stock prediction needs CSV files
   - Portuguese NLP examples need text corpora

3. **Performance:**
   - Small networks converge in milliseconds
   - Batch training significantly faster
   - Adam optimizer most reliable

## Testing Notes

- Always test with `go run examples/<name>/main.go`
- Some examples take >2 minutes (sentiment, mnist)
- Metal/GPU examples require Apple Silicon
