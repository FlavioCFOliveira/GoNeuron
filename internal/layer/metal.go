// +build darwin

package layer

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#cgo CFLAGS: -fobjc-arc
#include <stdbool.h>

// Forward declarations of C functions implemented in .m files
void* initMetalDevice();
bool isMetalAvailable(void* device);
void freeMetalDevice(void* device);
void* createMetalBuffer(void* device, float* data, int length);
void* createEmptyMetalBuffer(void* device, int length);
void updateMetalBuffer(void* buffer, float* data, int length);
void readMetalBuffer(void* buffer, float* data, int length);
void freeMetalBuffer(void* buffer);
void matMulMPSPersistent(void* device, void* bufA, void* bufB, void* bufC, int M, int N, int K, float beta);
void lstmStepFusedPersistent(void* device, void* bufPreAct, void* bufCPrev, void* bufCNew, void* bufHNew, void* bufI, void* bufF, void* bufG, void* bufO, int outSize);
void waitForMetal(void* device);
void matMulMPS(void* device, float* A, float* B, float* C, int M, int N, int K);
void matMulFused(void* device, float* A, float* B, float* Bias, float* C, int M, int N, int K, int activation);
void applyActivationMPS(void* device, float* data, int length, int activation);
void lstmStepFused(void* device, float* preAct, float* cPrev, float* cNew, float* hNew, float* iGate, float* fGate, float* gGate, float* oGate, int outSize);
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Activation types for fused kernels
const (
	ActivationNone    = 0
	ActivationSigmoid = 1
	ActivationTanh    = 2
	ActivationReLU    = 3
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

// MetalBuffer represents a buffer on the GPU.
type MetalBuffer struct {
	ptr unsafe.Pointer
}

func (d *MetalDevice) CreateBuffer(data []float32) *MetalBuffer {
	if d.ptr == nil || len(data) == 0 {
		return nil
	}
	return &MetalBuffer{ptr: C.createMetalBuffer(d.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))}
}

func (d *MetalDevice) CreateEmptyBuffer(length int) *MetalBuffer {
	if d.ptr == nil || length <= 0 {
		return nil
	}
	return &MetalBuffer{ptr: C.createEmptyMetalBuffer(d.ptr, C.int(length))}
}

func (b *MetalBuffer) Update(data []float32) {
	if b.ptr != nil && len(data) > 0 {
		C.updateMetalBuffer(b.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
	}
}

func (b *MetalBuffer) Read(data []float32) {
	if b.ptr != nil && len(data) > 0 {
		C.readMetalBuffer(b.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
	}
}

func (b *MetalBuffer) Free() {
	if b.ptr != nil {
		C.freeMetalBuffer(b.ptr)
		b.ptr = nil
	}
}

func (d *MetalDevice) MatMulPersistent(A, B, bufC *MetalBuffer, M, N, K int, beta float32) {
	if d.ptr != nil && A != nil && B != nil && bufC != nil {
		C.matMulMPSPersistent(d.ptr, A.ptr, B.ptr, bufC.ptr, C.int(M), C.int(N), C.int(K), C.float(beta))
	}
}

func (d *MetalDevice) LSTMStepPersistent(preAct, cPrev, cNew, hNew, iGate, fGate, gGate, oGate *MetalBuffer, outSize int) {
	if d.ptr != nil {
		C.lstmStepFusedPersistent(d.ptr, preAct.ptr, cPrev.ptr, cNew.ptr, hNew.ptr, iGate.ptr, fGate.ptr, gGate.ptr, oGate.ptr, C.int(outSize))
	}
}

func (d *MetalDevice) Wait() {
	if d.ptr != nil {
		C.waitForMetal(d.ptr)
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

// MatMulFused performs matrix multiplication C = A * B + Bias and applies activation.
func (d *MetalDevice) MatMulFused(A, B, Bias, C []float32, M, N, K int, activation int) {
	var biasPtr *C.float
	if Bias != nil && len(Bias) > 0 {
		biasPtr = (*C.float)(unsafe.Pointer(&Bias[0]))
	}

	C.matMulFused(d.ptr,
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&B[0])),
		biasPtr,
		(*C.float)(unsafe.Pointer(&C[0])),
		C.int(M), C.int(N), C.int(K), C.int(activation))
}

// ApplyActivation applies an activation function to data on GPU.
func (d *MetalDevice) ApplyActivation(data []float32, activation int) {
	if len(data) == 0 {
		return
	}
	C.applyActivationMPS(d.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)), C.int(activation))
}

// LSTMStep performs a fused LSTM forward step on GPU.
func (d *MetalDevice) LSTMStep(preAct, cPrev, cNew, hNew, iGate, fGate, gGate, oGate []float32, outSize int) {
	C.lstmStepFused(d.ptr,
		(*C.float)(unsafe.Pointer(&preAct[0])),
		(*C.float)(unsafe.Pointer(&cPrev[0])),
		(*C.float)(unsafe.Pointer(&cNew[0])),
		(*C.float)(unsafe.Pointer(&hNew[0])),
		(*C.float)(unsafe.Pointer(&iGate[0])),
		(*C.float)(unsafe.Pointer(&fGate[0])),
		(*C.float)(unsafe.Pointer(&gGate[0])),
		(*C.float)(unsafe.Pointer(&oGate[0])),
		C.int(outSize))
}
