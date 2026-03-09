# Metal Performance Optimizations

## Synchronization Reduction Pattern (PERF-016)

**Problem**: Multiple consecutive `Read()` calls on Metal buffers cause redundant CPU-GPU synchronizations. Each `Read()`:
- Acquires device mutex
- Calls `waitForMetal()` (sync point)
- Copies data from GPU to CPU

**Solution**: `ReadMultiple()` function that consolidates N reads into 1 sync point.

**Implementation**:
- Location: `internal/layer/metal.go:167-206`
- Key: Single `C.waitForMetal()` call before all buffer reads
- Safety: Validates all buffers belong to same device, handles nil/empty checks

**Usage Pattern**:
```go
// Before (3 sync points):
buffer1.Read(dst1)
buffer2.Read(dst2)
buffer3.Read(dst3)

// After (1 sync point):
ReadMultiple(
    []*MetalBuffer{buffer1, buffer2, buffer3},
    [][]float32{dst1, dst2, dst3},
)
```

**Files Modified**:
- `internal/layer/metal.go` - New `ReadMultiple()` function
- `internal/layer/layer.go` - Dense Backward (3→1 reads)
- `internal/layer/lstm.go` - LSTM Forward (7→1 reads)
- `internal/layer/dropout.go` - Dropout Forward (2→1 reads)
- `internal/layer/batchnorm2d.go` - BatchNorm2D Backward (3→1 reads)
- `internal/layer/conv2d.go` - Conv2D Backward (2→1 reads)

**Impact**: 50-86% reduction in CPU-GPU synchronization points during backward pass.

## Metal Buffer Management

**Key Insight**: On Apple Silicon's Unified Memory Architecture, the synchronization overhead (waiting for GPU to finish work) often dominates over the actual memory copy time. Minimizing sync points is crucial for performance.

**Best Practice**: When reading multiple buffers from GPU, always use `ReadMultiple()` instead of consecutive `Read()` calls.

## Thread Safety

Metal device uses a `sync.Mutex` to protect access. The `ReadMultiple()` function:
1. Acquires lock once for all operations
2. Validates device consistency across buffers
3. Releases lock after all reads complete

This reduces lock contention in multi-worker training scenarios.
