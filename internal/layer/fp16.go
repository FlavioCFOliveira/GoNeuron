// Package layer provides mixed precision (FP16) support for neural network layers.
// FP16 can significantly reduce memory usage and increase throughput on supported hardware.
//
// Current implementation provides:
// - Data type definitions and conversions
// - FP16 tensor storage
// - Automatic mixed precision training hooks
//
// Note: Full FP16 compute requires hardware support (e.g., Apple A15+, NVIDIA Tensor Cores).
// On unsupported hardware, tensors are stored as FP16 but computed as FP32.
package layer

import (
	"encoding/binary"
	"math"
)

// Float16 represents a 16-bit floating point number (IEEE 754 half-precision)
type Float16 uint16

// Float32ToFloat16 converts a float32 to Float16 (IEEE 754 half-precision)
// This handles NaN, Inf, and overflow correctly
func Float32ToFloat16(f float32) Float16 {
	// Special cases
	if math.IsNaN(float64(f)) {
		return Float16(0x7FFF) // Quiet NaN
	}
	if math.IsInf(float64(f), 1) {
		return Float16(0x7C00) // +Inf
	}
	if math.IsInf(float64(f), -1) {
		return Float16(0xFC00) // -Inf
	}
	if f == 0 {
		return Float16(0) // +0 (sign is ignored for 0 in this simple implementation)
	}

	// Extract components from float32
	bits := math.Float32bits(f)
	sign := uint32(bits & 0x80000000) >> 31
	exp := int32((bits >> 23) & 0xFF) - 127 // Remove bias
	mantissa := bits & 0x007FFFFF

	// Convert to float16 format
	var h uint16

	// Handle overflow to infinity
	if exp > 15 {
		if sign == 0 {
			return Float16(0x7C00) // +Inf
		}
		return Float16(0xFC00) // -Inf
	}

	// Handle underflow to zero
	if exp < -14 {
		// Too small, round to zero
		return Float16(0)
	}

	// Normal number
	newExp := uint16(exp + 15) // Apply FP16 bias (15)
	// Take top 10 bits of mantissa (23 - 13 = 10 bits)
	newMantissa := uint16(mantissa >> 13)

	h = uint16(sign<<15) | (newExp << 10) | newMantissa

	// Round to nearest even (simple rounding)
	if (mantissa & 0x1000) != 0 { // Check the bit we're rounding
		h++
	}

	return Float16(h)
}

// Float16ToFloat32 converts a Float16 to float32
func Float16ToFloat32(h Float16) float32 {
	h16 := uint16(h)

	// Special cases
	switch h16 {
	case 0x0000:
		return 0.0 // +0
	case 0x8000:
		return -0.0 // -0
	case 0x7C00:
		return float32(math.Inf(1)) // +Inf
	case 0xFC00:
		return float32(math.Inf(-1)) // -Inf
	}

	// Check for NaN
	if (h16&0x7C00) == 0x7C00 && (h16&0x03FF) != 0 {
		return float32(math.NaN()) // NaN
	}

	// Extract components from FP16
	sign := uint32(h16 >> 15)
	exp16 := uint32((h16 >> 10) & 0x1F)
	mant16 := uint32(h16 & 0x03FF)

	var exp32 uint32
	var mant32 uint32

	if exp16 == 0 {
		// Subnormal number in FP16
		// Convert to normal number in FP32
		// mant16 * 2^-14 * 2^-10 = mant16 * 2^-24
		// In FP32: exponent = 127 - 24 = 103
		if mant16 == 0 {
			return 0.0 // Zero, already handled above
		}
		// Normalize by shifting until leading 1 is at bit 10
		shift := uint32(0)
		for (mant16 & 0x400) == 0 {
			mant16 <<= 1
			shift++
		}
		// Now mant16 has leading 1 at position 10
		// Remove the leading 1
		mant16 &= 0x3FF
		// FP32 exponent = 127 - 14 - 10 + shift = 103 + shift
		exp32 = 127 - 14 - 10 + shift
		// FP32 mantissa = mant16 << (23 - 10) = mant16 << 13
		mant32 = mant16 << 13
	} else {
		// Normal number
		// FP16: (-1)^sign * 2^(exp16-15) * 1.mant16
		// FP32: (-1)^sign * 2^(exp32-127) * 1.mant32
		// exp32 - 127 = exp16 - 15
		// exp32 = exp16 - 15 + 127 = exp16 + 112
		exp32 = exp16 + 112
		// mant32 = mant16 << (23 - 10) = mant16 << 13
		mant32 = mant16 << 13
	}

	// Assemble FP32
	bits := (sign << 31) | (exp32 << 23) | mant32
	return math.Float32frombits(bits)
}

// Float32SliceToFloat16 converts a []float32 to []Float16
func Float32SliceToFloat16(src []float32) []Float16 {
	dst := make([]Float16, len(src))
	for i, f := range src {
		dst[i] = Float32ToFloat16(f)
	}
	return dst
}

// Float16SliceToFloat32 converts a []Float16 to []float32
func Float16SliceToFloat32(src []Float16) []float32 {
	dst := make([]float32, len(src))
	for i, h := range src {
		dst[i] = Float16ToFloat32(h)
	}
	return dst
}

// TensorFP16 represents a tensor stored in FP16 format
type TensorFP16 struct {
	Data   []Float16
	Shape  []int
	Strides []int
}

// NewTensorFP16 creates a new FP16 tensor with given shape
func NewTensorFP16(shape []int) *TensorFP16 {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	// Compute strides
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &TensorFP16{
		Data:    make([]Float16, size),
		Shape:   shape,
		Strides: strides,
	}
}

// FromFloat32 copies data from a float32 slice
func (t *TensorFP16) FromFloat32(src []float32) {
	if len(src) > len(t.Data) {
		panic("source slice too large for tensor")
	}
	for i, f := range src {
		t.Data[i] = Float32ToFloat16(f)
	}
}

// ToFloat32 returns a float32 copy of the data
func (t *TensorFP16) ToFloat32() []float32 {
	return Float16SliceToFloat32(t.Data)
}

// Get returns the value at given indices
func (t *TensorFP16) Get(indices []int) float32 {
	idx := t.computeIndex(indices)
	return Float16ToFloat32(t.Data[idx])
}

// Set sets the value at given indices
func (t *TensorFP16) Set(indices []int, value float32) {
	idx := t.computeIndex(indices)
	t.Data[idx] = Float32ToFloat16(value)
}

// computeIndex computes flat index from multi-dimensional indices
func (t *TensorFP16) computeIndex(indices []int) int {
	if len(indices) != len(t.Shape) {
		panic("index dimension mismatch")
	}
	idx := 0
	for i, dim := range indices {
		if dim < 0 || dim >= t.Shape[i] {
			panic("index out of bounds")
		}
		idx += dim * t.Strides[i]
	}
	return idx
}

// Size returns the total number of elements
func (t *TensorFP16) Size() int {
	return len(t.Data)
}

// MemoryBytes returns the memory usage in bytes
func (t *TensorFP16) MemoryBytes() int {
	return len(t.Data) * 2 // 2 bytes per float16
}

// MixedPrecisionConfig holds configuration for mixed precision training
type MixedPrecisionConfig struct {
	Enabled           bool    // Whether mixed precision is enabled
	LossScale         float32 // Loss scaling factor to prevent underflow
	MinLossScale      float32 // Minimum loss scale before giving up
	MaxLossScale      float32 // Maximum loss scale
	ScaleWindow       int     // Number of steps before adjusting loss scale
	CurrentScale      float32 // Current loss scale value
	ConsecutiveSkips  int     // Number of consecutive steps with Inf/NaN
}

// DefaultMixedPrecisionConfig returns a default configuration
func DefaultMixedPrecisionConfig() *MixedPrecisionConfig {
	return &MixedPrecisionConfig{
		Enabled:      true,
		LossScale:    32768.0,  // 2^15, common starting value
		MinLossScale: 1.0,
		MaxLossScale: 65536.0, // 2^16
		ScaleWindow:  2000,
		CurrentScale: 32768.0,
	}
}

// UpdateLossScale updates the loss scale based on whether the step was successful
func (c *MixedPrecisionConfig) UpdateLossScale(hasInfNaN bool) {
	if !c.Enabled {
		return
	}

	if hasInfNaN {
		// Overflow detected - reduce loss scale
		c.CurrentScale /= 2.0
		c.ConsecutiveSkips++
		if c.CurrentScale < c.MinLossScale {
			c.CurrentScale = c.MinLossScale
		}
	} else {
		// No overflow - increase loss scale if we've had enough successful steps
		c.ConsecutiveSkips = 0
		// In a full implementation, we'd track successful steps and increase
		// after ScaleWindow consecutive successful steps
	}
}

// ShouldSkipStep returns true if gradients have Inf/NaN and step should be skipped
func (c *MixedPrecisionConfig) ShouldSkipStep(gradients []float32) bool {
	if !c.Enabled {
		return false
	}

	for _, g := range gradients {
		if math.IsInf(float64(g), 0) || math.IsNaN(float64(g)) {
			return true
		}
	}
	return false
}

// ScaleLoss scales the loss value
func (c *MixedPrecisionConfig) ScaleLoss(loss float32) float32 {
	if !c.Enabled {
		return loss
	}
	return loss * c.CurrentScale
}

// UnscaleGradients unscales gradients before optimizer step
func (c *MixedPrecisionConfig) UnscaleGradients(gradients []float32) {
	if !c.Enabled {
		return
	}
	scale := 1.0 / c.CurrentScale
	for i := range gradients {
		gradients[i] *= scale
	}
}

// FP16ComputeDevice represents a device capable of FP16 computation
type FP16ComputeDevice interface {
	// IsFP16Supported returns true if device supports native FP16 computation
	IsFP16Supported() bool

	// ComputeCapability returns the compute capability (for NVIDIA) or version info
	ComputeCapability() string

	// MemoryBandwidth returns estimated memory bandwidth in GB/s
	MemoryBandwidth() float32
}

// FP16SupportInfo holds information about FP16 support on current hardware
type FP16SupportInfo struct {
	HardwareSupported bool
	NativeCompute     bool
	StorageSupported  bool
	DeviceName        string
}

// GetFP16SupportInfo returns information about FP16 support on this system
func GetFP16SupportInfo() FP16SupportInfo {
	// This is a stub - in a real implementation, this would query the hardware
	// For now, we assume FP16 storage is always supported
	// and native compute depends on the device
	return FP16SupportInfo{
		HardwareSupported: true,
		NativeCompute:     false, // Default to false, would be true on Apple Silicon, etc.
		StorageSupported:  true,
		DeviceName:        "unknown",
	}
}

// EncodeBytes converts Float16 slice to bytes for serialization
func (f Float16) EncodeBytes() []byte {
	b := make([]byte, 2)
	binary.LittleEndian.PutUint16(b, uint16(f))
	return b
}

// DecodeFloat16 decodes a Float16 from bytes
func DecodeFloat16(b []byte) Float16 {
	return Float16(binary.LittleEndian.Uint16(b))
}
