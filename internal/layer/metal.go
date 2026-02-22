// +build darwin

package layer

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#include <stdbool.h>

// Forward declarations of C functions implemented in .m files
void* initMetalDevice();
bool isMetalAvailable(void* device);
void freeMetalDevice(void* device);
void matMulMPS(void* device, float* A, float* B, float* C, int M, int N, int K);
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// MetalDevice handles computations on Apple Silicon GPU via Metal Performance Shaders.
type MetalDevice struct {
	ptr unsafe.Pointer
}

func NewMetalDevice() *MetalDevice {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		return &MetalDevice{ptr: nil}
	}
	return &MetalDevice{ptr: C.initMetalDevice()}
}

func (d *MetalDevice) Type() DeviceType { return GPU }

func (d *MetalDevice) IsAvailable() bool {
	if d.ptr == nil {
		return false
	}
	return bool(C.isMetalAvailable(d.ptr))
}

func (d *MetalDevice) Close() {
	if d.ptr != nil {
		C.freeMetalDevice(d.ptr)
		d.ptr = nil
	}
}

// MatMul performs matrix multiplication C = A * B using MPS.
// A: M x K, B: K x N, C: M x N
func (d *MetalDevice) MatMul(A, B, C []float32, M, N, K int) {
	C.matMulMPS(d.ptr,
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C[0])),
		C.int(M), C.int(N), C.int(K))
}
