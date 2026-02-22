//go:build darwin
// +build darwin

package metal

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#include "metal.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Device represents the Metal device and state
type Device struct {
	ctx C.MetalContext
}

// NewDevice initializes the Metal device
func NewDevice() *Device {
	ctx := C.InitMetal()
	if ctx == nil {
		return nil
	}
	dev := &Device{ctx: ctx}
	runtime.SetFinalizer(dev, func(d *Device) {
		C.FreeMetal(d.ctx)
	})
	return dev
}

// IsAvailable checks if Metal is available on the system
func IsAvailable() bool {
	return bool(C.IsMetalAvailable())
}

// Buffer wraps an MTLBuffer
type Buffer struct {
	ptr  C.MetalBuffer
	size int
}

// NewBuffer creates a shared memory buffer
func (d *Device) NewBuffer(size int) *Buffer {
	ptr := C.CreateSharedBuffer(d.ctx, C.size_t(size))
	if ptr == nil {
		return nil
	}
	buf := &Buffer{ptr: ptr, size: size}
	runtime.SetFinalizer(buf, func(b *Buffer) {
		C.FreeBuffer(b.ptr)
	})
	return buf
}

// Contents returns a pointer to the buffer's shared memory
func (b *Buffer) Contents() unsafe.Pointer {
	return C.GetBufferContents(b.ptr)
}

// Float32Data returns a slice of float32 pointing to the shared memory
func (b *Buffer) Float32Data() []float32 {
	const floatSize = 4
	count := b.size / floatSize
	return (*[1 << 30]float32)(b.Contents())[:count:count]
}

// SyncFromFloat64 copies data from a float64 slice to the Metal buffer as float32
func (b *Buffer) SyncFromFloat64(src []float64) {
	dst := b.Float32Data()
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	for i := 0; i < n; i++ {
		dst[i] = float32(src[i])
	}
}

// SyncToFloat64 copies data from the Metal buffer (float32) back to a float64 slice
func (b *Buffer) SyncToFloat64(dst []float64) {
	src := b.Float32Data()
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	for i := 0; i < n; i++ {
		dst[i] = float64(src[i])
	}
}

// MatMul performs matrix multiplication res = A * B using MPS
func (d *Device) MatMul(A, B, res *Buffer, m, n, k int) {
	C.MPSMatMul(d.ctx, A.ptr, B.ptr, res.ptr, C.int(m), C.int(n), C.int(k))
}
