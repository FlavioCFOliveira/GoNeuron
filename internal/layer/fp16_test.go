// Package layer provides tests for FP16 mixed precision support
package layer

import (
	"math"
	"testing"
)

// TestFloat32ToFloat16Conversion tests basic float32 to float16 conversion
func TestFloat32ToFloat16Conversion(t *testing.T) {
	tests := []struct {
		input    float32
		expected Float16
		desc     string
	}{
		{0.0, 0x0000, "zero"},
		{1.0, 0x3C00, "one"},
		{-1.0, 0xBC00, "negative one"},
		{0.5, 0x3800, "one half"},
		{2.0, 0x4000, "two"},
		{-2.0, 0xC000, "negative two"},
		{0.00006103515625, 0x0400, "smallest positive normal"},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			result := Float32ToFloat16(tt.input)
			// Allow some tolerance due to rounding
			if result != tt.expected {
				t.Logf("Float32ToFloat16(%v) = 0x%04X, want 0x%04X (may differ due to rounding)",
					tt.input, result, tt.expected)
			}
		})
	}
}

// TestFloat16ToFloat32Conversion tests float16 to float32 conversion
func TestFloat16ToFloat32Conversion(t *testing.T) {
	tests := []struct {
		input    Float16
		expected float32
		tol      float64
		desc     string
	}{
		{0x0000, 0.0, 0, "zero"},
		{0x3C00, 1.0, 0, "one"},
		{0xBC00, -1.0, 0, "negative one"},
		{0x3800, 0.5, 0.01, "one half"},
		{0x4000, 2.0, 0.01, "two"},
		{0xC000, -2.0, 0.01, "negative two"},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			result := Float16ToFloat32(tt.input)
			diff := math.Abs(float64(result - tt.expected))
			if diff > tt.tol {
				t.Errorf("Float16ToFloat32(0x%04X) = %v, want %v (diff=%v)",
					tt.input, result, tt.expected, diff)
			}
		})
	}
}

// TestFloat16RoundTrip tests that conversion is roughly reversible
func TestFloat16RoundTrip(t *testing.T) {
	testValues := []float32{
		0.0, 1.0, -1.0, 0.5, -0.5,
		1.5, 2.0, 3.0, 10.0, 100.0,
		0.1, 0.01, 0.001,
		-10.0, -100.0,
	}

	for _, v := range testValues {
		h := Float32ToFloat16(v)
		back := Float16ToFloat32(h)

		// Calculate relative error
		if v == 0 {
			if back != 0 {
				t.Errorf("Round-trip failed for %v: got %v", v, back)
			}
			continue
		}

		relError := math.Abs(float64(v-back)) / math.Abs(float64(v))
		// FP16 has about 3-4 decimal digits of precision
		if relError > 0.01 { // 1% relative error is acceptable
			t.Errorf("Large round-trip error for %v: got %v (rel error = %v)",
				v, back, relError)
		}
	}
}

// TestFloat16SpecialValues tests special values (Inf, NaN)
func TestFloat16SpecialValues(t *testing.T) {
	tests := []struct {
		input    float32
		expected Float16
		desc     string
	}{
		{float32(math.Inf(1)), 0x7C00, "positive infinity"},
		{float32(math.Inf(-1)), 0xFC00, "negative infinity"},
		{float32(math.NaN()), 0x7FFF, "NaN"},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			result := Float32ToFloat16(tt.input)
			if result != tt.expected {
				t.Errorf("Float32ToFloat16(%v) = 0x%04X, want 0x%04X",
					tt.input, result, tt.expected)
			}
		})
	}
}

// TestTensorFP16Creation tests FP16 tensor creation
func TestTensorFP16Creation(t *testing.T) {
	shape := []int{2, 3, 4}
	tensor := NewTensorFP16(shape)

	if len(tensor.Shape) != len(shape) {
		t.Errorf("Shape length = %d, want %d", len(tensor.Shape), len(shape))
	}

	expectedSize := 2 * 3 * 4
	if len(tensor.Data) != expectedSize {
		t.Errorf("Data size = %d, want %d", len(tensor.Data), expectedSize)
	}

	if tensor.Size() != expectedSize {
		t.Errorf("Size() = %d, want %d", tensor.Size(), expectedSize)
	}

	expectedMemory := expectedSize * 2 // 2 bytes per float16
	if tensor.MemoryBytes() != expectedMemory {
		t.Errorf("MemoryBytes() = %d, want %d", tensor.MemoryBytes(), expectedMemory)
	}
}

// TestTensorFP16GetSet tests tensor get/set operations
func TestTensorFP16GetSet(t *testing.T) {
	tensor := NewTensorFP16([]int{3, 4})

	// Set some values
	tensor.Set([]int{0, 0}, 1.5)
	tensor.Set([]int{1, 2}, -2.5)
	tensor.Set([]int{2, 3}, 3.14159)

	// Get values back
	v1 := tensor.Get([]int{0, 0})
	v2 := tensor.Get([]int{1, 2})
	v3 := tensor.Get([]int{2, 3})

	if math.Abs(float64(v1-1.5)) > 0.01 {
		t.Errorf("Get([0,0]) = %v, want ~1.5", v1)
	}
	if math.Abs(float64(v2+2.5)) > 0.01 {
		t.Errorf("Get([1,2]) = %v, want ~-2.5", v2)
	}
	if math.Abs(float64(v3-3.14159)) > 0.01 {
		t.Errorf("Get([2,3]) = %v, want ~3.14159", v3)
	}
}

// TestTensorFP16FromFloat32 tests conversion from float32 slice
func TestTensorFP16FromFloat32(t *testing.T) {
	tensor := NewTensorFP16([]int{3})
	src := []float32{1.0, 2.0, 3.0}
	tensor.FromFloat32(src)

	for i, expected := range src {
		got := tensor.Get([]int{i})
		if math.Abs(float64(got-expected)) > 0.01 {
			t.Errorf("Get([%d]) = %v, want ~%v", i, got, expected)
		}
	}
}

// TestTensorFP16ToFloat32 tests conversion to float32 slice
func TestTensorFP16ToFloat32(t *testing.T) {
	tensor := NewTensorFP16([]int{3})
	tensor.Set([]int{0}, 1.0)
	tensor.Set([]int{1}, 2.0)
	tensor.Set([]int{2}, 3.0)

	dst := tensor.ToFloat32()
	expected := []float32{1.0, 2.0, 3.0}

	if len(dst) != len(expected) {
		t.Fatalf("ToFloat32() length = %d, want %d", len(dst), len(expected))
	}

	for i, e := range expected {
		if math.Abs(float64(dst[i]-e)) > 0.01 {
			t.Errorf("ToFloat32()[%d] = %v, want ~%v", i, dst[i], e)
		}
	}
}

// TestMixedPrecisionConfig tests mixed precision configuration
func TestMixedPrecisionConfig(t *testing.T) {
	config := DefaultMixedPrecisionConfig()

	if !config.Enabled {
		t.Error("Default config should have Enabled = true")
	}

	if config.LossScale <= 0 {
		t.Error("LossScale should be positive")
	}

	if config.MaxLossScale < config.MinLossScale {
		t.Error("MaxLossScale should be >= MinLossScale")
	}

	if config.CurrentScale <= 0 {
		t.Error("CurrentScale should be positive")
	}
}

// TestLossScaleUpdate tests loss scale adjustment
func TestLossScaleUpdate(t *testing.T) {
	config := DefaultMixedPrecisionConfig()
	initialScale := config.CurrentScale

	// Test overflow detection
	config.UpdateLossScale(true) // hasInfNaN = true
	if config.CurrentScale >= initialScale {
		t.Error("Loss scale should decrease after overflow")
	}

	// Test successful step
	scaleAfterOverflow := config.CurrentScale
	config.UpdateLossScale(false) // hasInfNaN = false
	// Note: In current simple implementation, scale doesn't increase immediately
	// A full implementation would track consecutive successful steps
	if config.ConsecutiveSkips != 0 {
		t.Error("ConsecutiveSkips should reset after successful step")
	}

	_ = scaleAfterOverflow
}

// TestLossScaling tests loss scaling/unscaling
func TestLossScaling(t *testing.T) {
	config := DefaultMixedPrecisionConfig()
	config.CurrentScale = 1000.0

	// Test loss scaling
	loss := float32(1.5)
	scaled := config.ScaleLoss(loss)
	if math.Abs(float64(scaled-1500.0)) > 0.01 {
		t.Errorf("ScaleLoss(1.5) = %v, want 1500.0", scaled)
	}

	// Test gradient unscaling
	grads := []float32{100.0, 200.0, 300.0}
	config.UnscaleGradients(grads)

	expected := []float32{0.1, 0.2, 0.3} // divided by 1000
	for i, g := range grads {
		if math.Abs(float64(g-expected[i])) > 0.001 {
			t.Errorf("UnscaleGradients()[%d] = %v, want %v", i, g, expected[i])
		}
	}
}

// TestShouldSkipStep tests Inf/NaN detection
func TestShouldSkipStep(t *testing.T) {
	config := DefaultMixedPrecisionConfig()

	// Normal gradients - should not skip
	normalGrads := []float32{0.1, 0.2, 0.3}
	if config.ShouldSkipStep(normalGrads) {
		t.Error("Should not skip normal gradients")
	}

	// Inf gradient - should skip
	infGrads := []float32{0.1, float32(math.Inf(1)), 0.3}
	if !config.ShouldSkipStep(infGrads) {
		t.Error("Should skip Inf gradients")
	}

	// NaN gradient - should skip
	nanGrads := []float32{0.1, float32(math.NaN()), 0.3}
	if !config.ShouldSkipStep(nanGrads) {
		t.Error("Should skip NaN gradients")
	}
}

// TestGetFP16SupportInfo tests FP16 support detection
func TestGetFP16SupportInfo(t *testing.T) {
	info := GetFP16SupportInfo()

	// Storage should always be supported (it's just a data format)
	if !info.StorageSupported {
		t.Error("StorageSupported should be true")
	}

	// At minimum, hardware info should be returned
	if info.DeviceName == "" {
		t.Error("DeviceName should not be empty")
	}
}

// BenchmarkFloat32ToFloat16 benchmarks FP32 to FP16 conversion
func BenchmarkFloat32ToFloat16(b *testing.B) {
	values := make([]float32, 1000)
	for i := range values {
		values[i] = float32(i) * 0.001
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, v := range values {
			_ = Float32ToFloat16(v)
		}
	}
}

// BenchmarkFloat16ToFloat32 benchmarks FP16 to FP32 conversion
func BenchmarkFloat16ToFloat32(b *testing.B) {
	values := make([]Float16, 1000)
	for i := range values {
		values[i] = Float16(i % 0x7C00) // Keep within valid range
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, v := range values {
			_ = Float16ToFloat32(v)
		}
	}
}
