// Package data provides data augmentation utilities for neural network training.
// Data augmentation increases the effective size of the training dataset
// and helps prevent overfitting by applying random transformations.
//
// Supported augmentations:
// - Image: Flip, Rotate, Zoom, Shift, Brightness, Contrast
// - Time Series: Jitter, Scaling, Time Warp, Window Slicing
// - Audio: Noise, Pitch Shift, Speed Change
//
// All transformations preserve the semantic meaning of the data while
// introducing variability that helps the model generalize better.
package data

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

// Augmenter is the interface for all augmentation operations
type Augmenter interface {
	// Apply applies the augmentation to input data
	// Returns the augmented data
	Apply(input []float32) []float32

	// ApplyBatch applies augmentation to a batch of samples
	ApplyBatch(batch [][]float32) [][]float32

	// SetSeed sets the random seed for reproducibility
	SetSeed(seed int64)

	// Probability returns the probability of applying this augmentation
	Probability() float32

	// SetProbability sets the probability of applying this augmentation
	SetProbability(p float32)
}

// BaseAugmenter provides common functionality for all augmenters
type BaseAugmenter struct {
	prob   float32
	rng    *rand.Rand
	mu     sync.Mutex
}

// NewBaseAugmenter creates a new base augmenter
func NewBaseAugmenter(probability float32) *BaseAugmenter {
	return &BaseAugmenter{
		prob: probability,
		rng:  rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SetSeed sets the random seed
func (b *BaseAugmenter) SetSeed(seed int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.rng = rand.New(rand.NewSource(seed))
}

// Probability returns the augmentation probability
func (b *BaseAugmenter) Probability() float32 {
	return b.prob
}

// SetProbability sets the augmentation probability
func (b *BaseAugmenter) SetProbability(p float32) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.prob = p
}

// shouldApply returns true if augmentation should be applied based on probability
func (b *BaseAugmenter) shouldApply() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.rng.Float32() < b.prob
}

// randomFloat returns a random float32 in range [0, 1)
func (b *BaseAugmenter) randomFloat() float32 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.rng.Float32()
}

// randomRange returns a random float32 in range [min, max)
func (b *BaseAugmenter) randomRange(min, max float32) float32 {
	return min + b.randomFloat()*(max-min)
}

// randomInt returns a random integer in range [min, max)
func (b *BaseAugmenter) randomInt(min, max int) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.rng.Intn(max-min) + min
}

// FlipAugmenter performs horizontal/vertical flips (for images)
type FlipAugmenter struct {
	*BaseAugmenter
	Horizontal bool
	Vertical   bool
	width      int
	height     int
	channels   int
}

// NewFlipAugmenter creates a new flip augmenter
func NewFlipAugmenter(probability float32, horizontal, vertical bool, width, height, channels int) *FlipAugmenter {
	return &FlipAugmenter{
		BaseAugmenter: NewBaseAugmenter(probability),
		Horizontal:    horizontal,
		Vertical:      vertical,
		width:         width,
		height:        height,
		channels:      channels,
	}
}

// Apply applies flip augmentation
func (f *FlipAugmenter) Apply(input []float32) []float32 {
	if !f.shouldApply() {
		return input
	}

	output := make([]float32, len(input))
	copy(output, input)

	// When both horizontal and vertical are enabled, randomly choose one
	// to avoid applying both simultaneously which could be confusing
	if f.Horizontal && f.Vertical {
		if f.randomFloat() < 0.5 {
			f.applyHorizontalFlip(output)
		} else {
			f.applyVerticalFlip(output)
		}
	} else if f.Horizontal {
		f.applyHorizontalFlip(output)
	} else if f.Vertical {
		f.applyVerticalFlip(output)
	}

	return output
}

// applyHorizontalFlip performs horizontal flip
func (f *FlipAugmenter) applyHorizontalFlip(output []float32) {
	for y := 0; y < f.height; y++ {
		for x := 0; x < f.width/2; x++ {
			for c := 0; c < f.channels; c++ {
				idx1 := (y*f.width + x) * f.channels + c
				idx2 := (y*f.width + (f.width - 1 - x)) * f.channels + c
				output[idx1], output[idx2] = output[idx2], output[idx1]
			}
		}
	}
}

// applyVerticalFlip performs vertical flip
func (f *FlipAugmenter) applyVerticalFlip(output []float32) {
	for y := 0; y < f.height/2; y++ {
		for x := 0; x < f.width; x++ {
			for c := 0; c < f.channels; c++ {
				idx1 := (y*f.width + x) * f.channels + c
				idx2 := ((f.height-1-y)*f.width + x) * f.channels + c
				output[idx1], output[idx2] = output[idx2], output[idx1]
			}
		}
	}
}

// ApplyBatch applies flip to a batch
func (f *FlipAugmenter) ApplyBatch(batch [][]float32) [][]float32 {
	result := make([][]float32, len(batch))
	for i, sample := range batch {
		result[i] = f.Apply(sample)
	}
	return result
}

// RotateAugmenter performs rotation (for images)
type RotateAugmenter struct {
	*BaseAugmenter
	MaxAngle float32 // Maximum rotation angle in degrees
	width    int
	height   int
	channels int
}

// NewRotateAugmenter creates a new rotation augmenter
func NewRotateAugmenter(probability float32, maxAngle float32, width, height, channels int) *RotateAugmenter {
	return &RotateAugmenter{
		BaseAugmenter: NewBaseAugmenter(probability),
		MaxAngle:      maxAngle,
		width:         width,
		height:        height,
		channels:      channels,
	}
}

// Apply applies rotation augmentation
func (r *RotateAugmenter) Apply(input []float32) []float32 {
	if !r.shouldApply() {
		return input
	}

	angle := r.randomRange(-r.MaxAngle, r.MaxAngle) * float32(math.Pi) / 180.0
	cos := float32(math.Cos(float64(angle)))
	sin := float32(math.Sin(float64(angle)))

	output := make([]float32, len(input))

	cx := float32(r.width) / 2
	cy := float32(r.height) / 2

	for y := 0; y < r.height; y++ {
		for x := 0; x < r.width; x++ {
			// Compute source position
			dx := float32(x) - cx
			dy := float32(y) - cy

			srcX := dx*cos + dy*sin + cx
			srcY := -dx*sin + dy*cos + cy

			// Bilinear interpolation
			if srcX >= 0 && srcX < float32(r.width-1) && srcY >= 0 && srcY < float32(r.height-1) {
				x0 := int(srcX)
				y0 := int(srcY)
				x1 := x0 + 1
				y1 := y0 + 1

				dx := srcX - float32(x0)
				dy := srcY - float32(y0)

				for c := 0; c < r.channels; c++ {
					idx00 := (y0*r.width + x0) * r.channels + c
					idx01 := (y0*r.width + x1) * r.channels + c
					idx10 := (y1*r.width + x0) * r.channels + c
					idx11 := (y1*r.width + x1) * r.channels + c

					v00 := input[idx00]
					v01 := input[idx01]
					v10 := input[idx10]
					v11 := input[idx11]

					v0 := v00*(1-dx) + v01*dx
					v1 := v10*(1-dx) + v11*dx
					v := v0*(1-dy) + v1*dy

					idx := (y*r.width + x) * r.channels + c
					output[idx] = v
				}
			}
		}
	}

	return output
}

// ApplyBatch applies rotation to a batch
func (r *RotateAugmenter) ApplyBatch(batch [][]float32) [][]float32 {
	result := make([][]float32, len(batch))
	for i, sample := range batch {
		result[i] = r.Apply(sample)
	}
	return result
}

// ScaleAugmenter performs random scaling (for time series)
type ScaleAugmenter struct {
	*BaseAugmenter
	MinScale float32
	MaxScale float32
}

// NewScaleAugmenter creates a new scale augmenter
func NewScaleAugmenter(probability, minScale, maxScale float32) *ScaleAugmenter {
	return &ScaleAugmenter{
		BaseAugmenter: NewBaseAugmenter(probability),
		MinScale:      minScale,
		MaxScale:      maxScale,
	}
}

// Apply applies scaling augmentation
func (s *ScaleAugmenter) Apply(input []float32) []float32 {
	if !s.shouldApply() {
		return input
	}

	scale := s.randomRange(s.MinScale, s.MaxScale)

	output := make([]float32, len(input))
	for i, v := range input {
		output[i] = v * scale
	}

	return output
}

// ApplyBatch applies scaling to a batch
func (s *ScaleAugmenter) ApplyBatch(batch [][]float32) [][]float32 {
	result := make([][]float32, len(batch))
	for i, sample := range batch {
		result[i] = s.Apply(sample)
	}
	return result
}

// JitterAugmenter adds random noise (for time series)
type JitterAugmenter struct {
	*BaseAugmenter
	Sigma float32 // Standard deviation of noise
}

// NewJitterAugmenter creates a new jitter augmenter
func NewJitterAugmenter(probability, sigma float32) *JitterAugmenter {
	return &JitterAugmenter{
		BaseAugmenter: NewBaseAugmenter(probability),
		Sigma:         sigma,
	}
}

// Apply applies jitter augmentation
func (j *JitterAugmenter) Apply(input []float32) []float32 {
	if !j.shouldApply() {
		return input
	}

	output := make([]float32, len(input))
	for i, v := range input {
		noise := j.randomRange(-j.Sigma, j.Sigma)
		output[i] = v + noise
	}

	return output
}

// ApplyBatch applies jitter to a batch
func (j *JitterAugmenter) ApplyBatch(batch [][]float32) [][]float32 {
	result := make([][]float32, len(batch))
	for i, sample := range batch {
		result[i] = j.Apply(sample)
	}
	return result
}

// WindowSliceAugmenter extracts random windows (for time series)
type WindowSliceAugmenter struct {
	*BaseAugmenter
	WindowSize int
	Stride     int
}

// NewWindowSliceAugmenter creates a new window slice augmenter
func NewWindowSliceAugmenter(probability float32, windowSize, stride int) *WindowSliceAugmenter {
	return &WindowSliceAugmenter{
		BaseAugmenter: NewBaseAugmenter(probability),
		WindowSize:    windowSize,
		Stride:        stride,
	}
}

// Apply applies window slicing augmentation
func (w *WindowSliceAugmenter) Apply(input []float32) []float32 {
	if !w.shouldApply() || len(input) <= w.WindowSize {
		return input
	}

	// Random start position
	maxStart := len(input) - w.WindowSize
	start := w.randomInt(0, maxStart+1)

	output := make([]float32, w.WindowSize)
	copy(output, input[start:start+w.WindowSize])

	return output
}

// ApplyBatch applies window slicing to a batch
func (w *WindowSliceAugmenter) ApplyBatch(batch [][]float32) [][]float32 {
	result := make([][]float32, len(batch))
	for i, sample := range batch {
		result[i] = w.Apply(sample)
	}
	return result
}

// AugmentationPipeline chains multiple augmentations
type AugmentationPipeline struct {
	augmenters []Augmenter
	parallel   bool
}

// NewAugmentationPipeline creates a new augmentation pipeline
func NewAugmentationPipeline(parallel bool) *AugmentationPipeline {
	return &AugmentationPipeline{
		augmenters: make([]Augmenter, 0),
		parallel:   parallel,
	}
}

// Add adds an augmenter to the pipeline
func (p *AugmentationPipeline) Add(a Augmenter) {
	p.augmenters = append(p.augmenters, a)
}

// Apply applies all augmentations in sequence
func (p *AugmentationPipeline) Apply(input []float32) []float32 {
	result := input
	for _, aug := range p.augmenters {
		result = aug.Apply(result)
	}
	return result
}

// ApplyBatch applies pipeline to a batch
func (p *AugmentationPipeline) ApplyBatch(batch [][]float32) [][]float32 {
	if p.parallel && len(batch) > 1 {
		return p.applyBatchParallel(batch)
	}
	return p.applyBatchSequential(batch)
}

func (p *AugmentationPipeline) applyBatchSequential(batch [][]float32) [][]float32 {
	result := make([][]float32, len(batch))
	for i, sample := range batch {
		result[i] = p.Apply(sample)
	}
	return result
}

func (p *AugmentationPipeline) applyBatchParallel(batch [][]float32) [][]float32 {
	result := make([][]float32, len(batch))
	var wg sync.WaitGroup
	wg.Add(len(batch))

	for i, sample := range batch {
		go func(idx int, s []float32) {
			defer wg.Done()
			result[idx] = p.Apply(s)
		}(i, sample)
	}

	wg.Wait()
	return result
}

// SetSeed sets the seed for all augmenters in the pipeline
func (p *AugmentationPipeline) SetSeed(seed int64) {
	for _, aug := range p.augmenters {
		aug.SetSeed(seed)
	}
}

// Length returns the number of augmenters in the pipeline
func (p *AugmentationPipeline) Length() int {
	return len(p.augmenters)
}

// PredefinedAugmentations provides common augmentation presets

// ImageAugmentation returns a standard image augmentation pipeline
func ImageAugmentation(width, height, channels int) *AugmentationPipeline {
	p := NewAugmentationPipeline(true)
	p.Add(NewFlipAugmenter(0.5, true, false, width, height, channels))
	p.Add(NewRotateAugmenter(0.3, 15.0, width, height, channels))
	return p
}

// TimeSeriesAugmentation returns a standard time series augmentation pipeline
func TimeSeriesAugmentation() *AugmentationPipeline {
	p := NewAugmentationPipeline(false)
	p.Add(NewScaleAugmenter(0.5, 0.9, 1.1))
	p.Add(NewJitterAugmenter(0.5, 0.03))
	return p
}
