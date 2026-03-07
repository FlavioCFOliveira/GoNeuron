#!/bin/bash
# Script to update layer method signatures to return errors

# List of files to update
FILES=(
    "internal/layer/attention.go"
    "internal/layer/avgpool2d.go"
    "internal/layer/batchnorm2d.go"
    "internal/layer/bidirectional.go"
    "internal/layer/dropout.go"
    "internal/layer/embedding.go"
    "internal/layer/gru.go"
    "internal/layer/layernorm.go"
    "internal/layer/lstm.go"
    "internal/layer/maxpool2d.go"
    "internal/layer/moe.go"
    "internal/layer/rbf.go"
    "internal/layer/reshape.go"
    "internal/layer/rmsnorm.go"
    "internal/layer/swiglu.go"
    "internal/layer/transformer.go"
    "internal/layer/unroller.go"
    "internal/layer/conv1d.go"
    "internal/layer/conv2d.go"
)

for file in "${FILES[@]}"; do
    echo "Processing $file..."

    # Replace method signatures that return []float32 to return ([]float32, error)
    # Forward method
    gofmt -r 'func (l *LayerType) Forward(x []float32) []float32 { -> func (l *LayerType) Forward(x []float32) ([]float32, error) {' -w "$file" 2>/dev/null || true

    # Backward method
    gofmt -r 'func (l *LayerType) Backward(grad []float32) []float32 { -> func (l *LayerType) Backward(grad []float32) ([]float32, error) {' -w "$file" 2>/dev/null || true

    # AccumulateBackward method
    gofmt -r 'func (l *LayerType) AccumulateBackward(grad []float32) []float32 { -> func (l *LayerType) AccumulateBackward(grad []float32) ([]float32, error) {' -w "$file" 2>/dev/null || true

    # ForwardBatch method
    gofmt -r 'func (l *LayerType) ForwardBatch(x []float32, batchSize int) []float32 { -> func (l *LayerType) ForwardBatch(x []float32, batchSize int) ([]float32, error) {' -w "$file" 2>/dev/null || true

    # BackwardBatch method
    gofmt -r 'func (l *LayerType) BackwardBatch(grad []float32, batchSize int) []float32 { -> func (l *LayerType) BackwardBatch(grad []float32, batchSize int) ([]float32, error) {' -w "$file" 2>/dev/null || true

    # AccumulateBackwardBatch method
    gofmt -r 'func (l *LayerType) AccumulateBackwardBatch(grad []float32, batchSize int) []float32 { -> func (l *LayerType) AccumulateBackwardBatch(grad []float32, batchSize int) ([]float32, error) {' -w "$file" 2>/dev/null || true

    # ForwardWithArena method (for ArenaLayer interface)
    gofmt -r 'func (l *LayerType) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 { -> func (l *LayerType) ForwardWithArena(x []float32, arena *[]float32, offset *int) ([]float32, error) {' -w "$file" 2>/dev/null || true

    # ForwardBatchWithArena method
    gofmt -r 'func (l *LayerType) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) []float32 { -> func (l *LayerType) ForwardBatchWithArena(x []float32, batchSize int, arena *[]float32, offset *int) ([]float32, error) {' -w "$file" 2>/dev/null || true

done

echo "Signature updates complete. Manual fixes may still be needed for:"
echo "1. Return statements need to return nil, err or output, nil"
echo "2. Input validation should be added to methods"
echo "3. Internal calls need to handle error returns"
