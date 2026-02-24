//go:build !darwin
// +build !darwin

package layer

import "unsafe"

// Activation types for fused kernels (stubs)
const (
	MetalActivationNone    = 0
	MetalActivationSigmoid = 1
	MetalActivationTanh    = 2
	MetalActivationReLU    = 3
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

func (d *MetalDevice) Close() {}

type MetalBuffer struct {
	ptr    unsafe.Pointer
	device *MetalDevice
	length int
}

func (d *MetalDevice) CreateBuffer(data []float32) *MetalBuffer                           { return nil }
func (d *MetalDevice) CreateEmptyBuffer(length int) *MetalBuffer                          { return nil }
func (b *MetalBuffer) Update(data []float32)                                              {}
func (b *MetalBuffer) Read(data []float32)                                                {}
func (b *MetalBuffer) Free()                                                              {}
func (d *MetalDevice) MatMulPersistent(A, B, C *MetalBuffer, M, N, K int, beta float32)   {}
func (d *MetalDevice) MatMulFusedPersistent(A, B, Bias, C *MetalBuffer, M, N, K, act int) {}
func (d *MetalDevice) LSTMStepPersistent(preAct, cPrev, cNew, hNew, i, f, g, o *MetalBuffer, outSize int) {
}
func (d *MetalDevice) AvgPool2DPersistent(input, output *MetalBuffer, batchSize, channels, inH, inW, outH, outW, kSize, stride, padding int) {
}
func (d *MetalDevice) AvgPool2DBackwardPersistent(gradOutput, gradInput *MetalBuffer, batchSize, channels, inH, inW, outH, outW, kSize, stride, padding int) {
}
func (d *MetalDevice) MaxPool2DPersistent(input, output, argmax *MetalBuffer, batchSize, channels, inH, inW, outH, outW, kSize, stride, padding int) {
}
func (d *MetalDevice) MaxPool2DBackwardPersistent(gradOutput, gradInput, argmax *MetalBuffer, batchSize, channels, inH, inW, outH, outW, kSize, stride, padding int) {
}
func (d *MetalDevice) DropoutPersistent(input, output, mask *MetalBuffer, length int, p float32, training bool, seed uint64) {
}
func (d *MetalDevice) DropoutBackwardPersistent(gradOutput, gradInput, mask *MetalBuffer, length int, p float32) {
}
func (d *MetalDevice) CreateConv2DState(inC, outC, kSize, stride, padding int, weights, biases []float32) unsafe.Pointer {
	return nil
}
func (d *MetalDevice) Conv2DForward(state unsafe.Pointer, input, output *MetalBuffer, batchSize, inH, inW, outH, outW int) {
}
func (d *MetalDevice) Conv2DBackward(state unsafe.Pointer, input, gradOutput, gradInput, gradWeights, gradBiases *MetalBuffer, batchSize, inH, inW, outH, outW int) {
}
func (d *MetalDevice) Conv2DReloadWeights(state unsafe.Pointer) {}
func (d *MetalDevice) FreeConv2DState(state unsafe.Pointer)   {}

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
