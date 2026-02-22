//go:build !darwin
// +build !darwin

package metal

import "unsafe"

type Device struct{}

func NewDevice() *Device {
	return nil
}

func IsAvailable() bool {
	return false
}

type Buffer struct {
	size int
}

func (d *Device) NewBuffer(size int) *Buffer {
	return nil
}

func (b *Buffer) Contents() unsafe.Pointer {
	return nil
}

func (b *Buffer) Float32Data() []float32 {
	return nil
}

func (b *Buffer) SyncFromFloat64(src []float64) {}

func (b *Buffer) SyncToFloat64(dst []float64) {}

func (d *Device) MatMul(A, B, C *Buffer, m, n, k int) {}
