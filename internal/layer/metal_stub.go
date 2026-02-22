// +build !darwin

package layer

// MetalDevice is a stub for non-macOS platforms.
type MetalDevice struct{}

func NewMetalDevice() *MetalDevice {
	return &MetalDevice{}
}

func (d *MetalDevice) Type() DeviceType { return GPU }

func (d *MetalDevice) IsAvailable() bool {
	return false
}

func (d *MetalDevice) MatMul(A, B, C []float32, M, N, K int) {
	// No-op on non-macOS
}
