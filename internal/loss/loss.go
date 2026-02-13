// Package loss provides optimized loss functions.
package loss

import "math"

// BackwardInPlacer is an optional interface for loss functions that support
// in-place gradient computation to avoid allocations.
type BackwardInPlacer interface {
	BackwardInPlace(yPred, yTrue, grad []float64)
}

// Loss is a loss function with derivative.
type Loss interface {
	// Forward computes the loss between predicted and true values.
	Forward(yPred, yTrue []float64) float64

	// Backward computes the gradient of the loss w.r.t. prediction.
	// This creates a new slice and should be avoided in hot loops.
	Backward(yPred, yTrue []float64) []float64
}

// MSE (Mean Squared Error) loss.
type MSE struct{}

// Forward computes mean squared error: (1/n) * sum((y_pred - y_true)^2)
func (m MSE) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("MSE: prediction and target must have same length")
	}

	var sum float64
	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		sum += diff * diff
	}
	return sum / float64(n)
}

// Backward computes gradient: dL/dy_pred = (2/n) * (y_pred - y_true)
// Note: Returned slice is newly allocated for safety.
func (m MSE) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("MSE: prediction and target must have same length")
	}

	grad := make([]float64, n)
	factor := 2.0 / float64(n)
	for i := 0; i < n; i++ {
		grad[i] = factor * (yPred[i] - yTrue[i])
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
// This avoids allocation when grad slice is pre-allocated.
func (m MSE) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("MSE: slices must have same length")
	}

	factor := 2.0 / float64(n)
	for i := 0; i < n; i++ {
		grad[i] = factor * (yPred[i] - yTrue[i])
	}
}

// CrossEntropy loss for classification.
type CrossEntropy struct{}

// Forward computes cross entropy: -sum(y_true * log(y_pred + eps))
func (c CrossEntropy) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("CrossEntropy: prediction and target must have same length")
	}

	const eps = 1e-10
	var sum float64
	for i := 0; i < n; i++ {
		// Clip prediction to avoid log(0)
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		sum -= yTrue[i] * math.Log(pred)
	}
	return sum / float64(n)
}

// Backward computes gradient for cross entropy with softmax.
// For cross entropy + softmax, gradient simplifies to (y_pred - y_true).
func (c CrossEntropy) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("CrossEntropy: prediction and target must have same length")
	}

	grad := make([]float64, n)
	for i := 0; i < n; i++ {
		grad[i] = yPred[i] - yTrue[i]
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (c CrossEntropy) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("CrossEntropy: slices must have same length")
	}

	for i := 0; i < n; i++ {
		grad[i] = yPred[i] - yTrue[i]
	}
}

// Huber loss for robust regression.
type Huber struct {
	Delta float64 // Threshold for quadratic/linear transition
}

// NewHuber creates a Huber loss with the given delta.
func NewHuber(delta float64) *Huber {
	return &Huber{Delta: delta}
}

// Forward computes Huber loss.
func (h Huber) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("Huber: prediction and target must have same length")
	}

	var sum float64
	for i := 0; i < n; i++ {
		diff := math.Abs(yPred[i] - yTrue[i])
		if diff <= h.Delta {
			sum += 0.5 * diff * diff
		} else {
			sum += h.Delta * (diff - 0.5*h.Delta)
		}
	}
	return sum / float64(n)
}

// Backward computes gradient for Huber loss.
func (h Huber) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("Huber: prediction and target must have same length")
	}

	grad := make([]float64, n)
	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		if math.Abs(diff) <= h.Delta {
			grad[i] = diff
		} else {
			grad[i] = h.Delta * math.Copysign(1, diff)
		}
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (h Huber) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("Huber: slices must have same length")
	}

	for i := 0; i < n; i++ {
		diff := yPred[i] - yTrue[i]
		if math.Abs(diff) <= h.Delta {
			grad[i] = diff
		} else {
			grad[i] = h.Delta * math.Copysign(1, diff)
		}
	}
}

// L1Loss (Mean Absolute Error) loss.
type L1Loss struct{}

// Forward computes mean absolute error: (1/n) * sum(|y_pred - y_true|)
func (l L1Loss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("L1Loss: prediction and target must have same length")
	}

	var sum float64
	for i := 0; i < n; i++ {
		sum += math.Abs(yPred[i] - yTrue[i])
	}
	return sum / float64(n)
}

// Backward computes gradient for L1 loss: dL/dy_pred = (1/n) * sign(y_pred - y_true)
func (l L1Loss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("L1Loss: prediction and target must have same length")
	}

	grad := make([]float64, n)
	factor := 1.0 / float64(n)
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
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (l L1Loss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("L1Loss: slices must have same length")
	}

	factor := 1.0 / float64(n)
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
}

// BCELoss (Binary Cross Entropy) loss.
// Requires predictions to be in range (0, 1).
type BCELoss struct{}

// Forward computes binary cross entropy: -(1/n) * sum(y*log(p) + (1-y)*log(1-p))
func (b BCELoss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("BCELoss: prediction and target must have same length")
	}

	const eps = 1e-10
	var sum float64
	for i := 0; i < n; i++ {
		// Clip predictions to avoid log(0)
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		if pred > 1-eps {
			pred = 1 - eps
		}
		sum += yTrue[i]*math.Log(pred) + (1.0-yTrue[i])*math.Log(1.0-pred)
	}
	return -sum / float64(n)
}

// Backward computes gradient for BCE loss.
// Gradient: d/d_pred = (pred - y) / (pred * (1-pred)) / n
// Note: The gradient is normalized by n since the loss is averaged.
func (b BCELoss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("BCELoss: prediction and target must have same length")
	}

	grad := make([]float64, n)
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
		grad[i] = (pred - yTrue[i]) / (pred * (1.0-pred)) / float64(n)
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (b BCELoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("BCELoss: slices must have same length")
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
		grad[i] = (pred - yTrue[i]) / (pred * (1.0-pred) * float64(n))
	}
}

// BCEWithLogitsLoss combines BCE loss with sigmoid for numerical stability.
type BCEWithLogitsLoss struct{}

// Forward computes BCE loss with sigmoid applied internally for stability.
func (b BCEWithLogitsLoss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("BCEWithLogitsLoss: prediction and target must have same length")
	}

	var sum float64
	for i := 0; i < n; i++ {
		// Compute x = y_pred, y = y_true
		// Loss = max(x, 0) - x*y + log(1 + exp(-|x|))
		// This is numerically stable
		x := yPred[i]
		y := yTrue[i]
		if x >= 0 {
			sum += math.Log(1+math.Exp(-x)) + x - x*y
		} else {
			sum += -x + math.Log(1+math.Exp(x)) - x*y
		}
	}
	return sum / float64(n)
}

// Backward computes gradient for BCEWithLogitsLoss.
// Gradient is: sigmoid(x) - y
func (b BCEWithLogitsLoss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("BCEWithLogitsLoss: prediction and target must have same length")
	}

	grad := make([]float64, n)
	for i := 0; i < n; i++ {
		// sigmoid(x) - y
		sigmoid := 1.0 / (1.0 + math.Exp(-yPred[i]))
		grad[i] = (sigmoid - yTrue[i]) / float64(n)
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (b BCEWithLogitsLoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("BCEWithLogitsLoss: slices must have same length")
	}

	for i := 0; i < n; i++ {
		sigmoid := 1.0 / (1.0 + math.Exp(-yPred[i]))
		grad[i] = (sigmoid - yTrue[i]) / float64(n)
	}
}

// NLLLoss (Negative Log Likelihood) loss for classification.
// This is typically used with log-probabilities as input.
type NLLLoss struct{}

// Forward computes negative log likelihood: -sum(y_true * log(y_pred)) / n
func (n NLLLoss) Forward(yPred, yTrue []float64) float64 {
	nLen := len(yPred)
	if nLen != len(yTrue) {
		panic("NLLLoss: prediction and target must have same length")
	}

	var sum float64
	const eps = 1e-10
	for i := 0; i < nLen; i++ {
		// Clip to avoid log(0)
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		sum -= yTrue[i] * math.Log(pred)
	}
	return sum / float64(nLen)
}

// Backward computes gradient for NLL loss.
// For one-hot encoded targets: grad[i] = -y_true[i] / (y_pred[i] * n)
func (n NLLLoss) Backward(yPred, yTrue []float64) []float64 {
	nLen := len(yPred)
	if nLen != len(yTrue) {
		panic("NLLLoss: prediction and target must have same length")
	}

	grad := make([]float64, nLen)
	const eps = 1e-10
	for i := 0; i < nLen; i++ {
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		grad[i] = -yTrue[i] / (pred * float64(nLen))
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (n NLLLoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	nLen := len(yPred)
	if nLen != len(yTrue) || nLen != len(grad) {
		panic("NLLLoss: slices must have same length")
	}

	const eps = 1e-10
	for i := 0; i < nLen; i++ {
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		grad[i] = -yTrue[i] / (pred * float64(nLen))
	}
}

// CosineEmbeddingLoss measures similarity between two embeddings.
type CosineEmbeddingLoss struct {
	Margin float64 // Minimum margin between positive and negative pairs
}

// NewCosineEmbeddingLoss creates a CosineEmbeddingLoss with the given margin.
func NewCosineEmbeddingLoss(margin float64) *CosineEmbeddingLoss {
	return &CosineEmbeddingLoss{Margin: margin}
}

// Forward computes cosine embedding loss.
// For y=1 (similar): loss = 1 - cos(x1, x2)
// For y=-1 (dissimilar): loss = max(0, cos(x1, x2) - margin)
func (c CosineEmbeddingLoss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("CosineEmbeddingLoss: prediction and target must have same length")
	}

	var sum float64
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
	return sum / float64(n)
}

// Backward computes gradient for cosine embedding loss.
func (c CosineEmbeddingLoss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("CosineEmbeddingLoss: prediction and target must have same length")
	}

	grad := make([]float64, n)
	for i := 0; i < n; i++ {
		y := yTrue[i]
		pred := yPred[i]

		if y > 0 { // Similar pair: d/d_pred(1 - pred) = -1
			grad[i] = -1.0 / float64(n)
		} else { // Dissimilar pair
			if pred > c.Margin {
				grad[i] = 1.0 / float64(n)
			} else {
				grad[i] = 0
			}
		}
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (c CosineEmbeddingLoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("CosineEmbeddingLoss: slices must have same length")
	}

	for i := 0; i < n; i++ {
		y := yTrue[i]
		pred := yPred[i]

		if y > 0 {
			grad[i] = -1.0 / float64(n)
		} else {
			if pred > c.Margin {
				grad[i] = 1.0 / float64(n)
			} else {
				grad[i] = 0
			}
		}
	}
}

// HingeEmbeddingLoss measures embedding similarity using hinge loss.
type HingeEmbeddingLoss struct {
	Margin float64 // Margin for dissimilar pairs
}

// NewHingeEmbeddingLoss creates a HingeEmbeddingLoss with the given margin.
func NewHingeEmbeddingLoss(margin float64) *HingeEmbeddingLoss {
	return &HingeEmbeddingLoss{Margin: margin}
}

// Forward computes hinge embedding loss.
// For y=1: loss = x
// For y=-1: loss = max(0, margin - x)
func (h HingeEmbeddingLoss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("HingeEmbeddingLoss: prediction and target must have same length")
	}

	var sum float64
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
	return sum / float64(n)
}

// Backward computes gradient for hinge embedding loss.
func (h HingeEmbeddingLoss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("HingeEmbeddingLoss: prediction and target must have same length")
	}

	grad := make([]float64, n)
	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		if y > 0 {
			grad[i] = 1.0 / float64(n)
		} else {
			if x < h.Margin {
				grad[i] = -1.0 / float64(n)
			} else {
				grad[i] = 0
			}
		}
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (h HingeEmbeddingLoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("HingeEmbeddingLoss: slices must have same length")
	}

	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		if y > 0 {
			grad[i] = 1.0 / float64(n)
		} else {
			if x < h.Margin {
				grad[i] = -1.0 / float64(n)
			} else {
				grad[i] = 0
			}
		}
	}
}

// TripletMarginLoss measures similarity using triplet margin.
// It takes anchor, positive, and negative embeddings as input.
// Loss = max(0, d(a,p) - d(a,n) + margin)
type TripletMarginLoss struct {
	Margin float64 // Margin for the triplet loss
	P      int     // The norm degree for pairwise distance (1 or 2)
}

// NewTripletMarginLoss creates a TripletMarginLoss with the given margin and norm.
func NewTripletMarginLoss(margin float64, p int) *TripletMarginLoss {
	return &TripletMarginLoss{
		Margin: margin,
		P:      p,
	}
}

// Forward computes triplet margin loss.
// yPred contains [anchor, positive, negative] concatenated.
func (t TripletMarginLoss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("TripletMarginLoss: prediction and target must have same length")
	}
	if n%3 != 0 {
		panic("TripletMarginLoss: prediction length must be divisible by 3")
	}

	embDim := n / 3
	var sum float64

	for i := 0; i < embDim; i++ {
		anchor := yPred[i]
		positive := yPred[embDim+i]
		negative := yPred[2*embDim+i]

		// Compute L2 distances
		distPos := (anchor - positive) * (anchor - positive)
		distNeg := (anchor - negative) * (anchor - negative)

		if t.P == 1 {
			distPos = math.Abs(anchor - positive)
			distNeg = math.Abs(anchor - negative)
		}

		// Triplet loss: max(0, dist_pos - dist_neg + margin)
		marginDiff := distPos - distNeg + t.Margin
		if marginDiff > 0 {
			sum += marginDiff
		}
	}

	return sum / float64(embDim)
}

// Backward computes gradient for triplet margin loss.
func (t TripletMarginLoss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("TripletMarginLoss: prediction and target must have same length")
	}
	if n%3 != 0 {
		panic("TripletMarginLoss: prediction length must be divisible by 3")
	}

	embDim := n / 3
	grad := make([]float64, n)

	for i := 0; i < embDim; i++ {
		anchor := yPred[i]
		positive := yPred[embDim+i]
		negative := yPred[2*embDim+i]

		distPos := (anchor - positive) * (anchor - positive)
		distNeg := (anchor - negative) * (anchor - negative)

		if t.P == 1 {
			distPos = math.Abs(anchor - positive)
			distNeg = math.Abs(anchor - negative)
		}

		marginDiff := distPos - distNeg + t.Margin
		if marginDiff <= 0 {
			continue
		}

		if t.P == 2 {
			grad[i] += 2 * (negative - positive) / float64(embDim)
			grad[embDim+i] += -2 * (anchor - positive) / float64(embDim)
			grad[2*embDim+i] += 2 * (anchor - negative) / float64(embDim)
		} else {
			if anchor > positive {
				grad[i] += 1 / float64(embDim)
				grad[embDim+i] += -1 / float64(embDim)
			} else {
				grad[i] += -1 / float64(embDim)
				grad[embDim+i] += 1 / float64(embDim)
			}
			if anchor > negative {
				grad[i] += -1 / float64(embDim)
				grad[2*embDim+i] += 1 / float64(embDim)
			} else {
				grad[i] += 1 / float64(embDim)
				grad[2*embDim+i] += -1 / float64(embDim)
			}
		}
	}

	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (t TripletMarginLoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) {
		panic("TripletMarginLoss: prediction and target must have same length")
	}
	if n%3 != 0 {
		panic("TripletMarginLoss: prediction length must be divisible by 3")
	}
	if len(grad) != n {
		panic("TripletMarginLoss: grad must have same length as prediction")
	}

	embDim := n / 3

	for i := 0; i < embDim; i++ {
		anchor := yPred[i]
		positive := yPred[embDim+i]
		negative := yPred[2*embDim+i]

		distPos := (anchor - positive) * (anchor - positive)
		distNeg := (anchor - negative) * (anchor - negative)

		if t.P == 1 {
			distPos = math.Abs(anchor - positive)
			distNeg = math.Abs(anchor - negative)
		}

		marginDiff := distPos - distNeg + t.Margin
		if marginDiff <= 0 {
			continue
		}

		if t.P == 2 {
			grad[i] += 2 * (negative - positive) / float64(embDim)
			grad[embDim+i] += -2 * (anchor - positive) / float64(embDim)
			grad[2*embDim+i] += 2 * (anchor - negative) / float64(embDim)
		} else {
			if anchor > positive {
				grad[i] += 1 / float64(embDim)
				grad[embDim+i] += -1 / float64(embDim)
			} else {
				grad[i] += -1 / float64(embDim)
				grad[embDim+i] += 1 / float64(embDim)
			}
			if anchor > negative {
				grad[i] += -1 / float64(embDim)
				grad[2*embDim+i] += 1 / float64(embDim)
			} else {
				grad[i] += 1 / float64(embDim)
				grad[2*embDim+i] += -1 / float64(embDim)
			}
		}
	}
}

// MarginRankingLoss for ranking tasks.
// Loss = max(0, -y * (x1 - x2) + margin)
type MarginRankingLoss struct {
	Margin float64 // Margin for ranking loss
}

// NewMarginRankingLoss creates a MarginRankingLoss with the given margin.
func NewMarginRankingLoss(margin float64) *MarginRankingLoss {
	return &MarginRankingLoss{Margin: margin}
}

// Forward computes margin ranking loss.
// yPred contains [x1, x2] concatenated (length = 2 * nPairs).
// yTrue contains targets (+1 or -1) for each pair (length = nPairs).
func (m MarginRankingLoss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n%2 != 0 {
		panic("MarginRankingLoss: prediction length must be divisible by 2")
	}

	nPairs := n / 2
	if len(yTrue) != nPairs {
		panic("MarginRankingLoss: target length must equal prediction length / 2")
	}

	var sum float64

	for i := 0; i < nPairs; i++ {
		x1 := yPred[i]
		x2 := yPred[nPairs+i]
		y := yTrue[i]

		diff := -y * (x1 - x2) + m.Margin
		if diff > 0 {
			sum += diff
		}
	}

	return sum / float64(nPairs)
}

// Backward computes gradient for margin ranking loss.
func (m MarginRankingLoss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n%2 != 0 {
		panic("MarginRankingLoss: prediction length must be divisible by 2")
	}

	nPairs := n / 2
	if len(yTrue) != nPairs {
		panic("MarginRankingLoss: target length must equal prediction length / 2")
	}

	grad := make([]float64, n)

	for i := 0; i < nPairs; i++ {
		x1 := yPred[i]
		x2 := yPred[nPairs+i]
		y := yTrue[i]

		diff := -y * (x1 - x2) + m.Margin
		if diff <= 0 {
			continue
		}

		grad[i] -= y / float64(nPairs)
		grad[nPairs+i] += y / float64(nPairs)
	}

	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (m MarginRankingLoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n%2 != 0 {
		panic("MarginRankingLoss: prediction length must be divisible by 2")
	}

	nPairs := n / 2
	if len(yTrue) != nPairs {
		panic("MarginRankingLoss: target length must equal prediction length / 2")
	}
	if len(grad) != n {
		panic("MarginRankingLoss: grad must have same length as prediction")
	}

	for i := 0; i < nPairs; i++ {
		x1 := yPred[i]
		x2 := yPred[nPairs+i]
		y := yTrue[i]

		diff := -y * (x1 - x2) + m.Margin
		if diff <= 0 {
			continue
		}

		grad[i] -= y / float64(nPairs)
		grad[nPairs+i] += y / float64(nPairs)
	}
}

// KLDivLoss (KL Divergence) loss.
type KLDivLoss struct{}

// Forward computes KL divergence: sum(y_true * log(y_true / y_pred)) / n
func (k KLDivLoss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("KLDivLoss: prediction and target must have same length")
	}

	const eps = 1e-10
	var sum float64
	for i := 0; i < n; i++ {
		pred := yPred[i]
		trueVal := yTrue[i]

		if pred < eps {
			pred = eps
		}
		if trueVal < eps {
			trueVal = eps
		}

		sum += trueVal * math.Log(trueVal/pred)
	}
	return sum / float64(n)
}

// Backward computes gradient for KL divergence: -y_true / (y_pred * n)
func (k KLDivLoss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("KLDivLoss: prediction and target must have same length")
	}

	grad := make([]float64, n)
	const eps = 1e-10
	for i := 0; i < n; i++ {
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		grad[i] = -yTrue[i] / (pred * float64(n))
	}
	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (k KLDivLoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("KLDivLoss: slices must have same length")
	}

	const eps = 1e-10
	for i := 0; i < n; i++ {
		pred := yPred[i]
		if pred < eps {
			pred = eps
		}
		grad[i] = -yTrue[i] / (pred * float64(n))
	}
}

// MultiMarginLoss for multi-class classification with margin.
type MultiMarginLoss struct {
	Margin float64 // Margin for the hinge loss
	P      int     // The norm degree (1 or 2)
}

// NewMultiMarginLoss creates a MultiMarginLoss with the given margin and norm.
func NewMultiMarginLoss(margin float64, p int) *MultiMarginLoss {
	return &MultiMarginLoss{Margin: margin, P: p}
}

// Forward computes multi-class margin loss.
// yPred contains class scores.
// yTrue contains integer class indices.
func (m MultiMarginLoss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("MultiMarginLoss: prediction and target must have same length")
	}

	var sum float64
	for i := 0; i < n; i++ {
		target := int(yTrue[i])

		if target < 0 || target >= n {
			continue // Skip invalid indices
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
	return sum / float64(n)
}

// Backward computes gradient for multi-class margin loss.
func (m MultiMarginLoss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("MultiMarginLoss: prediction and target must have same length")
	}

	grad := make([]float64, n)
	factor := 1.0 / float64(n)

	for i := 0; i < n; i++ {
		target := int(yTrue[i])

		if target < 0 || target >= n {
			continue
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

	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (m MultiMarginLoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("MultiMarginLoss: slices must have same length")
	}

	for i := 0; i < n; i++ {
		target := int(yTrue[i])

		if target < 0 || target >= n {
			continue
		}

		targetScore := yPred[target]

		for j := 0; j < n; j++ {
			if j == target {
				continue
			}

			diff := m.Margin - targetScore + yPred[j]
			if diff > 0 {
				if m.P == 1 {
					grad[target] -= 1.0 / float64(n)
					grad[j] += 1.0 / float64(n)
				} else {
					grad[target] -= 2 * diff / float64(n)
					grad[j] += 2 * diff / float64(n)
				}
			}
		}
	}
}

// MultiLabelSoftMarginLoss for multi-label classification.
type MultiLabelSoftMarginLoss struct{}

// Forward computes multi-label soft margin loss.
// Loss = -sum(y*log(σ(x)) + (1-y)*log(1-σ(x))) / n
func (m MultiLabelSoftMarginLoss) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("MultiLabelSoftMarginLoss: prediction and target must have same length")
	}

	const eps = 1e-10
	var sum float64
	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		// Clip for numerical stability
		pred := 1.0 / (1.0 + math.Exp(-math.Abs(x)))
		if pred < eps {
			pred = eps
		}
		if pred > 1-eps {
			pred = 1 - eps
		}

		if y > 0 {
			sum += math.Log(pred + eps)
		} else {
			sum += math.Log(1-pred + eps)
		}
	}
	return -sum / float64(n)
}

// Backward computes gradient for multi-label soft margin loss.
func (m MultiLabelSoftMarginLoss) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n != len(yTrue) {
		panic("MultiLabelSoftMarginLoss: prediction and target must have same length")
	}

	grad := make([]float64, n)
	factor := 1.0 / float64(n)
	const eps = 1e-10

	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		// sigmoid(x) - y
		sigmoid := 1.0 / (1.0 + math.Exp(-x))
		grad[i] = (sigmoid - y) * factor
	}

	return grad
}

// BackwardInPlace computes gradient and stores it in the grad slice.
func (m MultiLabelSoftMarginLoss) BackwardInPlace(yPred, yTrue, grad []float64) {
	n := len(yPred)
	if n != len(yTrue) || n != len(grad) {
		panic("MultiLabelSoftMarginLoss: slices must have same length")
	}

	factor := 1.0 / float64(n)

	for i := 0; i < n; i++ {
		x := yPred[i]
		y := yTrue[i]

		sigmoid := 1.0 / (1.0 + math.Exp(-x))
		grad[i] = (sigmoid - y) * factor
	}
}

