// +build !darwin

package layer

// Activation types for fused kernels (stubs)
const (
	ActivationNone    = 0
	ActivationSigmoid = 1
	ActivationTanh    = 2
	ActivationReLU    = 3
)

// MetalDevice is a stub for non-macOS platforms.
type MetalDevice struct{}

func NewMetalDevice() *MetalDevice {
	return &MetalDevice{}
}

func (d *MetalDevice) Type() DeviceType { return GPU }

func (d *MetalDevice) IsAvailable() bool {
	return false
}

type MetalBuffer struct {
	ptr unsafe.Pointer
}

func (d *MetalDevice) CreateBuffer(data []float32) *MetalBuffer { return nil }
func (d *MetalDevice) CreateEmptyBuffer(length int) *MetalBuffer { return nil }
func (b *MetalBuffer) Update(data []float32) {}
func (b *MetalBuffer) Read(data []float32) {}
func (b *MetalBuffer) Free() {}
func (d *MetalDevice) MatMulPersistent(A, B, C *MetalBuffer, M, N, K int, beta float32) {}
func (d *MetalDevice) LSTMStepPersistent(preAct, cPrev, cNew, hNew, iGate, fGate, gGate, oGate *MetalBuffer, outSize int) {}
func (d *MetalDevice) Wait() {}

func (d *MetalDevice) MatMul(A, B, C []float32, M, N, K int) {
	// No-op on non-macOS
}

func (d *MetalDevice) MatMulFused(A, B, Bias, C []float32, M, N, K int, activation int) {
	// No-op on non-macOS
}

func (d *MetalDevice) ApplyActivation(data []float32, activation int) {
	// No-op on non-macOS
}

func (d *MetalDevice) LSTMStep(preAct, cPrev, cNew, hNew, iGate, fGate, gGate, oGate []float32, outSize int) {
	// No-op on non-macOS
}
