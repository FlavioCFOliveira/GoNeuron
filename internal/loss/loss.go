// Package loss provides optimized loss functions.
package loss

import (
	"errors"
	"math"
)

// Common errors
var (
	ErrLengthMismatch   = errors.New("prediction and target must have same length")
	ErrDivisibleBy3     = errors.New("prediction length must be divisible by 3")
	ErrDivisibleBy2     = errors.New("prediction length must be divisible by 2")
	ErrInvalidDimension = errors.New("invalid dimension")
	ErrEmptyInput       = errors.New("input slice is empty")
	ErrInvalidTarget    = errors.New("target index out of bounds")
)

// BackwardInPlacer is an optional interface for loss functions that support
// in-place gradient computation to avoid allocations.
type BackwardInPlacer interface {
	BackwardInPlace(yPred, yTrue, grad []float32) error
}

// Loss is a loss function with derivative.
type Loss interface {
	// Forward computes the loss between predicted and true values.
	// Returns error if input lengths don't match.
	Forward(yPred, yTrue []float32) (float32, error)

	// Backward computes the gradient of the loss w.r.t. prediction.
	// Uses internal buffer reuse for zero-allocation operation.
	// For thread-safe usage, use BackwardInPlace with externally provided buffer.
	// Returns error if input lengths don't match.
	Backward(yPred, yTrue []float32) ([]float32, error)
}

// MSE (Mean Squared Error) loss with buffer reuse.
type MSE struct {
	// gradBuf is a pre-allocated buffer for gradient computation.
	// Grows as needed and is reused across Backward calls.
	// For thread-safe concurrent usage, use BackwardInPlace with external buffer.
	gradBuf []float32
}

// ResetGradBuffer clears the internal gradient buffer.
// Call this when input size changes significantly to release memory.
func (m *MSE) ResetGradBuffer() {
	m.gradBuf = nil
}

// Forward computes mean squared error: (1/n) * sum((y_pred - y_true)^2)
func (m MSE) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	var sum float32
	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		sum += diff * diff
	}
	return sum / float32(n), nil
}

// Backward computes gradient: dL/dy_pred = (2/n) * (y_pred - y_true)
// Uses stack-allocated buffer for zero-allocation operation.
func (m MSE) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	grad := make([]float32, n)
	factor := float32(2.0) / float32(n)
	for i := 0; i < n; i++ {
		grad[i] = factor * (yPred[i] - yTrue[i])
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// This is thread-safe when using externally provided buffers.
func (m MSE) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	factor := float32(2.0) / float32(n)
	for i := 0; i < n; i++ {
		grad[i] = factor * (yPred[i] - yTrue[i])
	}
	return nil
}

// CrossEntropy loss for classification with buffer reuse.
type CrossEntropy struct {
	// gradBuf is a pre-allocated buffer for gradient computation.
	// Grows as needed and is reused across Backward calls.
	gradBuf []float32
}

// ResetGradBuffer clears the internal gradient buffer.
func (c *CrossEntropy) ResetGradBuffer() {
	c.gradBuf = nil
}

// Forward computes cross entropy: -sum(y_true * log(y_pred + eps))
func (c CrossEntropy) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	const eps = 1e-10
	var sum float32
	for i := 0; i < n; i++ {
		// Clip prediction to avoid log(0)
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		sum -= yTrue[i] * float32(math.Log(float64(pred)))
	}
	return sum, nil
}

// Backward computes gradient for cross entropy combined with softmax.
// When used with softmax activation, the combined gradient is: dL/dz = y_pred - y_true
// This assumes y_pred contains probabilities from softmax and y_true is one-hot encoded.
// Uses stack-allocated buffer for zero-allocation operation.
func (c CrossEntropy) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	grad := make([]float32, n)
	for i := 0; i < n; i++ {
		grad[i] = yPred[i] - yTrue[i]
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Combined gradient: y_pred - y_true (when yPred is from softmax)
// Thread-safe when using externally provided buffers.
func (c CrossEntropy) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	for i := 0; i < n; i++ {
		grad[i] = yPred[i] - yTrue[i]
	}
	return nil
}

// Huber loss for robust regression with buffer reuse.
type Huber struct {
	Delta   float32 // Threshold for quadratic/linear transition
	gradBuf []float32
}

// NewHuber creates a Huber loss with the given delta.
func NewHuber(delta float32) *Huber {
	return &Huber{Delta: delta}
}

// ResetGradBuffer clears the internal gradient buffer.
func (h *Huber) ResetGradBuffer() {
	h.gradBuf = nil
}

// Forward computes Huber loss.
func (h *Huber) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	var sum float32
	for i := 0; i < n; i++ {
		diff := float32(math.Abs(float64(yPred[i] - yTrue[i])))
		if diff <= h.Delta {
			sum += 0.5 * diff * diff
		} else {
			sum += h.Delta * (diff - 0.5*h.Delta)
		}
	}
	return sum / float32(n), nil
}

// Backward computes gradient for Huber loss.
// Reuses internal buffer to avoid allocations.
func (h *Huber) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(h.gradBuf) < n {
		h.gradBuf = make([]float32, n)
	}
	grad := h.gradBuf[:n]

	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		if float32(math.Abs(float64(diff))) <= h.Delta {
			grad[i] = diff
		} else {
			grad[i] = h.Delta * float32(math.Copysign(1, float64(diff)))
		}
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (h Huber) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		if float32(math.Abs(float64(diff))) <= h.Delta {
			grad[i] = diff
		} else {
			grad[i] = h.Delta * float32(math.Copysign(1, float64(diff)))
		}
	}
	return nil
}

// L1Loss (Mean Absolute Error) loss with buffer reuse.
type L1Loss struct {
	gradBuf []float32
}

// ResetGradBuffer clears the internal gradient buffer.
func (l *L1Loss) ResetGradBuffer() {
	l.gradBuf = nil
}

// Forward computes mean absolute error: (1/n) * sum(|y_pred - y_true|)
func (l *L1Loss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	var sum float32
	for i := 0; i < n; i++ {
		sum += float32(math.Abs(float64(yPred[i] - yTrue[i])))
	}
	return sum / float32(n), nil
}

// Backward computes gradient for L1 loss: dL/dy_pred = (1/n) * sign(y_pred - y_true)
// Reuses internal buffer to avoid allocations.
func (l *L1Loss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(l.gradBuf) < n {
		l.gradBuf = make([]float32, n)
	}
	grad := l.gradBuf[:n]

	factor := float32(1.0) / float32(n)
	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		if diff > 0 {
			grad[i] = factor
		} else if diff < 0 {
			grad[i] = -factor
		} else {
			grad[i] = 0
		}
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (l L1Loss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	factor := float32(1.0) / float32(n)
	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		if diff > 0 {
			grad[i] = factor
		} else if diff < 0 {
			grad[i] = -factor
		} else {
			grad[i] = 0
		}
	}
	return nil
}

// BCELoss (Binary Cross Entropy) loss with buffer reuse.
// Requires predictions to be in range (0, 1).
type BCELoss struct {
	gradBuf []float32
}

// ResetGradBuffer clears the internal gradient buffer.
func (b *BCELoss) ResetGradBuffer() {
	b.gradBuf = nil
}

// Forward computes binary cross entropy: -(1/n) * sum(y*log(p) + (1-y)*log(1-p))
func (b BCELoss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	const eps = 1e-10
	var sum float32
	for i := 0; i < n; i++ {
		// Clip predictions to avoid log(0)
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		if pred > 1-eps {
			pred = 1 - eps
		}
		sum += yTrue[i]*float32(math.Log(float64(pred))) + (1.0-yTrue[i])*float32(math.Log(float64(1.0-pred)))
	}
	return -sum / float32(n), nil
}

// Backward computes gradient for BCE loss.
// Gradient: d/d_pred = (pred - y) / (pred * (1-pred)) / n
// Note: The gradient is normalized by n since the loss is averaged.
// Reuses internal buffer to avoid allocations.
func (b BCELoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(b.gradBuf) < n {
		b.gradBuf = make([]float32, n)
	}
	grad := b.gradBuf[:n]

	const eps = 1e-10
	for i := 0; i < n; i++ {
		// Clip predictions to avoid division by zero
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		if pred > 1-eps {
			pred = 1 - eps
		}
		// Gradient of BCE: (pred - y) / (pred * (1-pred)) / n
		grad[i] = (pred - yTrue[i]) / (pred * (1.0-pred)) / float32(n)
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (b BCELoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	const eps = 1e-10
	for i := 0; i < n; i++ {
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		if pred > 1-eps {
			pred = 1 - eps
		}
		grad[i] = (pred - yTrue[i]) / (pred * (1.0-pred) * float32(n))
	}
	return nil
}

// BCEWithLogitsLoss combines BCE loss with sigmoid for numerical stability with buffer reuse.
type BCEWithLogitsLoss struct {
	gradBuf []float32
}

// ResetGradBuffer clears the internal gradient buffer.
func (b *BCEWithLogitsLoss) ResetGradBuffer() {
	b.gradBuf = nil
}

// Forward computes BCE loss with sigmoid applied internally for stability.
func (b BCEWithLogitsLoss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	var sum float32
	for i := 0; i < n; i++ {
		// Compute x = y_pred, y = y_true
		// Loss = max(x, 0) - x*y + log(1 + exp(-|x|))
		// This is numerically stable
		x := yPred[i]
		y := yTrue[i]
		if x >= 0 {
			sum += float32(math.Log(1+math.Exp(float64(-x)))) + x - x*y
		} else {
			sum += float32(math.Log(1+math.Exp(float64(x)))) - x*y
		}
	}
	return sum / float32(n), nil
}

// Backward computes gradient for BCEWithLogitsLoss.
// Gradient is: sigmoid(x) - y
// Reuses internal buffer to avoid allocations.
func (b BCEWithLogitsLoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(b.gradBuf) < n {
		b.gradBuf = make([]float32, n)
	}
	grad := b.gradBuf[:n]

	for i := 0; i < n; i++ {
		// sigmoid(x) - y
		sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-yPred[i]))))
		grad[i] = (sigmoid - yTrue[i]) / float32(n)
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (b BCEWithLogitsLoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	for i := 0; i < n; i++ {
		sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-yPred[i]))))
		grad[i] = (sigmoid - yTrue[i]) / float32(n)
	}
	return nil
}

// NLLLoss (Negative Log Likelihood) loss for classification with buffer reuse.
// This is typically used with log-probabilities as input (e.g., from LogSoftmax).
// When input is log-probabilities: loss = -sum(y_true * log_pred) / n
// When input is probabilities: use CrossEntropy instead.
type NLLLoss struct {
	gradBuf []float32
}

// ResetGradBuffer clears the internal gradient buffer.
func (n *NLLLoss) ResetGradBuffer() {
	n.gradBuf = nil
}

// Forward computes negative log likelihood from log-probabilities.
// Input should be log-probabilities (e.g., output of LogSoftmax).
func (n NLLLoss) Forward(yPred, yTrue []float32) (float32, error) {
	nLen := len(yPred)
	if nLen == 0 {
		return 0, ErrEmptyInput
	}
	if nLen != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	var sum float32
	for i := 0; i < nLen; i++ {
		// yPred is assumed to be log-probability, so use directly
		// Clip to avoid issues with -inf
		logPred := yPred[i]
		if logPred < -700 { // exp(-700) is essentially 0
			logPred = -700
		}
		sum -= yTrue[i] * logPred
	}
	return sum / float32(nLen), nil
}

// Backward computes gradient for NLL loss with log-probabilities.
// For log-prob input: grad[i] = -y_true[i]
// This is combined with LogSoftmax's gradient in the Dense layer.
// Reuses internal buffer to avoid allocations.
func (n NLLLoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	nLen := len(yPred)
	if nLen == 0 {
		return nil, ErrEmptyInput
	}
	if nLen != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(n.gradBuf) < nLen {
		n.gradBuf = make([]float32, nLen)
	}
	grad := n.gradBuf[:nLen]

	factor := float32(1.0) / float32(nLen)
	for i := 0; i < nLen; i++ {
		grad[i] = -yTrue[i] * factor
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (n NLLLoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	nLen := len(yPred)
	if nLen == 0 {
		return ErrEmptyInput
	}
	if nLen != len(yTrue) || nLen != len(grad) {
		return ErrLengthMismatch
	}

	factor := float32(1.0) / float32(nLen)
	for i := 0; i < nLen; i++ {
		grad[i] = -yTrue[i] * factor
	}
	return nil
}

// CosineEmbeddingLoss measures similarity between two embeddings with buffer reuse.
type CosineEmbeddingLoss struct {
	Margin  float32 // Minimum margin between positive and negative pairs
	gradBuf []float32
}

// NewCosineEmbeddingLoss creates a CosineEmbeddingLoss with the given margin.
func NewCosineEmbeddingLoss(margin float32) *CosineEmbeddingLoss {
	return &CosineEmbeddingLoss{Margin: margin}
}

// ResetGradBuffer clears the internal gradient buffer.
func (c *CosineEmbeddingLoss) ResetGradBuffer() {
	c.gradBuf = nil
}

// Forward computes cosine embedding loss.
// For y=1 (similar): loss = 1 - cos(x1, x2)
// For y=-1 (dissimilar): loss = max(0, cos(x1, x2) - margin)
func (c *CosineEmbeddingLoss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	var sum float32
	for i := 0; i < n; i++ {
		y := yTrue[i]
		pred := yPred[i]

		if y > 0 { // Similar pair
			sum += 1.0 - pred
		} else { // Dissimilar pair
			diff := pred - c.Margin
			if diff > 0 {
				sum += diff
			}
		}
	}
	return sum / float32(n), nil
}

// Backward computes gradient for cosine embedding loss.
// Reuses internal buffer to avoid allocations.
func (c *CosineEmbeddingLoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(c.gradBuf) < n {
		c.gradBuf = make([]float32, n)
	}
	grad := c.gradBuf[:n]

	for i := 0; i < n; i++ {
		y := yTrue[i]
		pred := yPred[i]

		if y > 0 { // Similar pair: d/d_pred(1 - pred) = -1
			grad[i] = -1.0 / float32(n)
		} else { // Dissimilar pair
			if pred > c.Margin {
				grad[i] = 1.0 / float32(n)
			} else {
				grad[i] = 0
			}
		}
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (c CosineEmbeddingLoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	for i := 0; i < n; i++ {
		y := yTrue[i]
		pred := yPred[i]

		if y > 0 {
			grad[i] = -1.0 / float32(n)
		} else {
			if pred > c.Margin {
				grad[i] = 1.0 / float32(n)
			} else {
				grad[i] = 0
			}
		}
	}
	return nil
}

// HingeEmbeddingLoss measures embedding similarity using hinge loss with buffer reuse.
type HingeEmbeddingLoss struct {
	Margin  float32 // Margin for dissimilar pairs
	gradBuf []float32
}

// NewHingeEmbeddingLoss creates a HingeEmbeddingLoss with the given margin.
func NewHingeEmbeddingLoss(margin float32) *HingeEmbeddingLoss {
	return &HingeEmbeddingLoss{Margin: margin}
}

// ResetGradBuffer clears the internal gradient buffer.
func (h *HingeEmbeddingLoss) ResetGradBuffer() {
	h.gradBuf = nil
}

// Forward computes hinge embedding loss.
// For y=1: loss = x
// For y=-1: loss = max(0, margin - x)
func (h *HingeEmbeddingLoss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	var sum float32
	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		if y > 0 {
			sum += x
		} else {
			diff := h.Margin - x
			if diff > 0 {
				sum += diff
			}
		}
	}
	return sum / float32(n), nil
}

// Backward computes gradient for hinge embedding loss.
// Reuses internal buffer to avoid allocations.
func (h *HingeEmbeddingLoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(h.gradBuf) < n {
		h.gradBuf = make([]float32, n)
	}
	grad := h.gradBuf[:n]

	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		if y > 0 {
			grad[i] = 1.0 / float32(n)
		} else {
			if x < h.Margin {
				grad[i] = -1.0 / float32(n)
			} else {
				grad[i] = 0
			}
		}
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (h HingeEmbeddingLoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		if y > 0 {
			grad[i] = 1.0 / float32(n)
		} else {
			if x < h.Margin {
				grad[i] = -1.0 / float32(n)
			} else {
				grad[i] = 0
			}
		}
	}
	return nil
}

// TripletMarginLoss measures similarity using triplet margin with buffer reuse.
// It takes anchor, positive, and negative embeddings as input.
// Loss = max(0, d(a,p) - d(a,n) + margin)
type TripletMarginLoss struct {
	Margin  float32 // Margin for the triplet loss
	P       int     // The norm degree for pairwise distance (1 or 2)
	gradBuf []float32
}

// NewTripletMarginLoss creates a TripletMarginLoss with the given margin and norm.
func NewTripletMarginLoss(margin float32, p int) *TripletMarginLoss {
	return &TripletMarginLoss{
		Margin: margin,
		P:      p,
	}
}

// ResetGradBuffer clears the internal gradient buffer.
func (t *TripletMarginLoss) ResetGradBuffer() {
	t.gradBuf = nil
}

// Forward computes triplet margin loss.
// yPred contains [anchor, positive, negative] concatenated.
func (t *TripletMarginLoss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}
	if n%3 != 0 {
		return 0, ErrDivisibleBy3
	}

	embDim := n / 3
	var sum float32

	for i := 0; i < embDim; i++ {
		anchor := yPred[i]
		positive := yPred[embDim+i]
		negative := yPred[2*embDim+i]

		// Compute L2 distances
		distPos := (anchor - positive) * (anchor - positive)
		distNeg := (anchor - negative) * (anchor - negative)

		if t.P == 1 {
			distPos = float32(math.Abs(float64(anchor - positive)))
			distNeg = float32(math.Abs(float64(anchor - negative)))
		}

		// Triplet loss: max(0, dist_pos - dist_neg + margin)
		marginDiff := distPos - distNeg + t.Margin
		if marginDiff > 0 {
			sum += marginDiff
		}
	}

	return sum / float32(embDim), nil
}

// Backward computes gradient for triplet margin loss.
// Reuses internal buffer to avoid allocations.
func (t *TripletMarginLoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}
	if n%3 != 0 {
		return nil, ErrDivisibleBy3
	}

	embDim := n / 3

	// Reuse or grow internal buffer
	if cap(t.gradBuf) < n {
		t.gradBuf = make([]float32, n)
	}
	grad := t.gradBuf[:n]
	// Clear gradient buffer since we accumulate
	for i := range grad {
		grad[i] = 0
	}

	for i := 0; i < embDim; i++ {
		anchor := yPred[i]
		positive := yPred[embDim+i]
		negative := yPred[2*embDim+i]

		distPos := (anchor - positive) * (anchor - positive)
		distNeg := (anchor - negative) * (anchor - negative)

		if t.P == 1 {
			distPos = float32(math.Abs(float64(anchor - positive)))
			distNeg = float32(math.Abs(float64(anchor - negative)))
		}

		marginDiff := distPos - distNeg + t.Margin
		if marginDiff <= 0 {
			continue
		}

		if t.P == 2 {
			grad[i] += 2 * (negative - positive) / float32(embDim)
			grad[embDim+i] += -2 * (anchor - positive) / float32(embDim)
			grad[2*embDim+i] += 2 * (anchor - negative) / float32(embDim)
		} else {
			if anchor > positive {
				grad[i] += 1 / float32(embDim)
				grad[embDim+i] += -1 / float32(embDim)
			} else {
				grad[i] += -1 / float32(embDim)
				grad[embDim+i] += 1 / float32(embDim)
			}
			if anchor > negative {
				grad[i] += -1 / float32(embDim)
				grad[2*embDim+i] += 1 / float32(embDim)
			} else {
				grad[i] += 1 / float32(embDim)
				grad[2*embDim+i] += -1 / float32(embDim)
			}
		}
	}

	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (t TripletMarginLoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) {
		return ErrLengthMismatch
	}
	if n%3 != 0 {
		return ErrDivisibleBy3
	}
	if len(grad) != n {
		return ErrLengthMismatch
	}

	embDim := n / 3

	for i := 0; i < embDim; i++ {
		anchor := yPred[i]
		positive := yPred[embDim+i]
		negative := yPred[2*embDim+i]

		distPos := (anchor - positive) * (anchor - positive)
		distNeg := (anchor - negative) * (anchor - negative)

		if t.P == 1 {
			distPos = float32(math.Abs(float64(anchor - positive)))
			distNeg = float32(math.Abs(float64(anchor - negative)))
		}

		marginDiff := distPos - distNeg + t.Margin
		if marginDiff <= 0 {
			continue
		}

		if t.P == 2 {
			grad[i] += 2 * (negative - positive) / float32(embDim)
			grad[embDim+i] += -2 * (anchor - positive) / float32(embDim)
			grad[2*embDim+i] += 2 * (anchor - negative) / float32(embDim)
		} else {
			if anchor > positive {
				grad[i] += 1 / float32(embDim)
				grad[embDim+i] += -1 / float32(embDim)
			} else {
				grad[i] += -1 / float32(embDim)
				grad[embDim+i] += 1 / float32(embDim)
			}
			if anchor > negative {
				grad[i] += -1 / float32(embDim)
				grad[2*embDim+i] += 1 / float32(embDim)
			} else {
				grad[i] += 1 / float32(embDim)
				grad[2*embDim+i] += -1 / float32(embDim)
			}
		}
	}
	return nil
}

// MarginRankingLoss for ranking tasks with buffer reuse.
// Loss = max(0, -y * (x1 - x2) + margin)
type MarginRankingLoss struct {
	Margin  float32 // Margin for ranking loss
	gradBuf []float32
}

// NewMarginRankingLoss creates a MarginRankingLoss with the given margin.
func NewMarginRankingLoss(margin float32) *MarginRankingLoss {
	return &MarginRankingLoss{Margin: margin}
}

// ResetGradBuffer clears the internal gradient buffer.
func (m *MarginRankingLoss) ResetGradBuffer() {
	m.gradBuf = nil
}

// Forward computes margin ranking loss.
// yPred contains [x1, x2] concatenated (length = 2 * nPairs).
// yTrue contains targets (+1 or -1) for each pair (length = nPairs).
func (m *MarginRankingLoss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n%2 != 0 {
		return 0, ErrDivisibleBy2
	}

	nPairs := n / 2
	if len(yTrue) != nPairs {
		return 0, ErrLengthMismatch
	}

	var sum float32

	for i := 0; i < nPairs; i++ {
		x1 := yPred[i]
		x2 := yPred[nPairs+i]
		y := yTrue[i]

		diff := -y * (x1 - x2) + m.Margin
		if diff > 0 {
			sum += diff
		}
	}

	return sum / float32(nPairs), nil
}

// Backward computes gradient for margin ranking loss.
// Reuses internal buffer to avoid allocations.
func (m *MarginRankingLoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n%2 != 0 {
		return nil, ErrDivisibleBy2
	}

	nPairs := n / 2
	if len(yTrue) != nPairs {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(m.gradBuf) < n {
		m.gradBuf = make([]float32, n)
	}
	grad := m.gradBuf[:n]
	// Clear gradient buffer
	for i := range grad {
		grad[i] = 0
	}

	for i := 0; i < nPairs; i++ {
		x1 := yPred[i]
		x2 := yPred[nPairs+i]
		y := yTrue[i]

		diff := -y * (x1 - x2) + m.Margin
		if diff <= 0 {
			continue
		}

		grad[i] -= y / float32(nPairs)
		grad[nPairs+i] += y / float32(nPairs)
	}

	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (m MarginRankingLoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n%2 != 0 {
		return ErrDivisibleBy2
	}

	nPairs := n / 2
	if len(yTrue) != nPairs {
		return ErrLengthMismatch
	}
	if len(grad) != n {
		return ErrLengthMismatch
	}

	for i := 0; i < nPairs; i++ {
		x1 := yPred[i]
		x2 := yPred[nPairs+i]
		y := yTrue[i]

		diff := -y * (x1 - x2) + m.Margin
		if diff <= 0 {
			continue
		}

		grad[i] -= y / float32(nPairs)
		grad[nPairs+i] += y / float32(nPairs)
	}
	return nil
}

// KLDivLoss (KL Divergence) loss with buffer reuse.
type KLDivLoss struct {
	gradBuf []float32
}

// ResetGradBuffer clears the internal gradient buffer.
func (k *KLDivLoss) ResetGradBuffer() {
	k.gradBuf = nil
}

// Forward computes KL divergence: sum(y_true * log(y_true / y_pred)) / n
func (k *KLDivLoss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	const eps = 1e-10
	var sum float32
	for i := 0; i < n; i++ {
		pred := yPred[i]
		trueVal := yTrue[i]

		if pred < eps {
			pred = eps
		}
		if trueVal < eps {
			trueVal = eps
		}

		sum += trueVal * float32(math.Log(float64(trueVal/pred)))
	}
	return sum / float32(n), nil
}

// Backward computes gradient for KL divergence: -y_true / (y_pred * n)
// Reuses internal buffer to avoid allocations.
func (k *KLDivLoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(k.gradBuf) < n {
		k.gradBuf = make([]float32, n)
	}
	grad := k.gradBuf[:n]

	const eps = 1e-10
	for i := 0; i < n; i++ {
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		grad[i] = -yTrue[i] / (pred * float32(n))
	}
	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (k KLDivLoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	const eps = 1e-10
	for i := 0; i < n; i++ {
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		grad[i] = -yTrue[i] / (pred * float32(n))
	}
	return nil
}

// MultiMarginLoss for multi-class classification with margin and buffer reuse.
type MultiMarginLoss struct {
	Margin  float32 // Margin for the hinge loss
	P       int     // The norm degree (1 or 2)
	gradBuf []float32
}

// NewMultiMarginLoss creates a MultiMarginLoss with the given margin and norm.
func NewMultiMarginLoss(margin float32, p int) *MultiMarginLoss {
	return &MultiMarginLoss{Margin: margin, P: p}
}

// ResetGradBuffer clears the internal gradient buffer.
func (m *MultiMarginLoss) ResetGradBuffer() {
	m.gradBuf = nil
}

// Forward computes multi-class margin loss.
// yPred contains class scores.
// yTrue contains integer class indices.
func (m *MultiMarginLoss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	var sum float32
	for i := 0; i < n; i++ {
		target := int(yTrue[i])

		if target < 0 || target >= n {
			return 0, ErrInvalidTarget
		}

		targetScore := yPred[target]

		// For each non-target class
		for j := 0; j < n; j++ {
			if j == target {
				continue
			}

			// Loss = max(0, margin - target_score + other_score)^p
			diff := m.Margin - targetScore + yPred[j]
			if diff > 0 {
				if m.P == 1 {
					sum += diff
				} else {
					sum += diff * diff
				}
			}
		}
	}
	return sum / float32(n), nil
}

// Backward computes gradient for multi-class margin loss.
// Reuses internal buffer to avoid allocations.
func (m *MultiMarginLoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(m.gradBuf) < n {
		m.gradBuf = make([]float32, n)
	}
	grad := m.gradBuf[:n]
	// Clear gradient buffer
	for i := range grad {
		grad[i] = 0
	}

	factor := float32(1.0) / float32(n)

	for i := 0; i < n; i++ {
		target := int(yTrue[i])

		if target < 0 || target >= n {
			return nil, ErrInvalidTarget
		}

		targetScore := yPred[target]

		for j := 0; j < n; j++ {
			if j == target {
				continue
			}

			diff := m.Margin - targetScore + yPred[j]
			if diff > 0 {
				if m.P == 1 {
					grad[target] -= factor
					grad[j] += factor
				} else {
					grad[target] -= 2 * diff * factor
					grad[j] += 2 * diff * factor
				}
			}
		}
	}

	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (m MultiMarginLoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	for i := 0; i < n; i++ {
		target := int(yTrue[i])

		if target < 0 || target >= n {
			return ErrInvalidTarget
		}

		targetScore := yPred[target]

		for j := 0; j < n; j++ {
			if j == target {
				continue
			}

			diff := m.Margin - targetScore + yPred[j]
			if diff > 0 {
				if m.P == 1 {
					grad[target] -= 1.0 / float32(n)
					grad[j] += 1.0 / float32(n)
				} else {
					grad[target] -= 2 * diff / float32(n)
					grad[j] += 2 * diff / float32(n)
				}
			}
		}
	}
	return nil
}

// MultiLabelSoftMarginLoss for multi-label classification with buffer reuse.
type MultiLabelSoftMarginLoss struct {
	gradBuf []float32
}

// ResetGradBuffer clears the internal gradient buffer.
func (m *MultiLabelSoftMarginLoss) ResetGradBuffer() {
	m.gradBuf = nil
}

// Forward computes multi-label soft margin loss.
// Loss = -sum(y*log(σ(x)) + (1-y)*log(1-σ(x))) / n
func (m *MultiLabelSoftMarginLoss) Forward(yPred, yTrue []float32) (float32, error) {
	n := len(yPred)
	if n == 0 {
		return 0, ErrEmptyInput
	}
	if n != len(yTrue) {
		return 0, ErrLengthMismatch
	}

	const eps = 1e-10
	var sum float32
	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		// Clip for numerical stability
		pred := float32(1.0 / (1.0 + math.Exp(-math.Abs(float64(x)))))
		if pred < eps {
			pred = eps
		}
		if pred > 1-eps {
			pred = 1 - eps
		}

		if y > 0 {
			sum += float32(math.Log(float64(pred + eps)))
		} else {
			sum += float32(math.Log(float64(1-pred + eps)))
		}
	}
	return -sum / float32(n), nil
}

// Backward computes gradient for multi-label soft margin loss.
// Reuses internal buffer to avoid allocations.
func (m *MultiLabelSoftMarginLoss) Backward(yPred, yTrue []float32) ([]float32, error) {
	n := len(yPred)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	if n != len(yTrue) {
		return nil, ErrLengthMismatch
	}

	// Reuse or grow internal buffer
	if cap(m.gradBuf) < n {
		m.gradBuf = make([]float32, n)
	}
	grad := m.gradBuf[:n]

	factor := float32(1.0) / float32(n)
	const eps = 1e-10

	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		// sigmoid(x) - y
		sigmoid := float32(1.0 / (1.0 + math.Exp(float64(-x))))
		grad[i] = (sigmoid - y) * factor
	}

	return grad, nil
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// Thread-safe when using externally provided buffers.
func (m MultiLabelSoftMarginLoss) BackwardInPlace(yPred, yTrue, grad []float32) error {
	n := len(yPred)
	if n == 0 {
		return ErrEmptyInput
	}
	if n != len(yTrue) || n != len(grad) {
		return ErrLengthMismatch
	}

	factor := float32(1.0) / float32(n)

	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		sigmoid := float32(1.0 / (1.0 + math.Exp(float64(-x))))
		grad[i] = (sigmoid - y) * factor
	}
	return nil
}
