//go:build darwin
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
void matMulMPSPersistent(void* device, void* bufA, void* bufB, void* bufC, int M, int N, int K, float beta, bool transA, bool transB);
void matMulFusedPersistent(void* device, void* bufA, void* bufB, void* bufBias, void* bufC, int M, int N, int K, int activation);
void lstmStepFusedPersistent(void* device, void* bufPreAct, void* bufCPrev, void* bufCNew, void* bufHNew, void* bufI, void* bufF, void* bufG, void* bufO, int outSize);
void avgPool2DPersistent(void* device, void* input, void* output, int batchSize, int channels, int inH, int inW, int outH, int outW, int kSize, int stride, int padding);
void avgPool2DBackwardPersistent(void* device, void* gradOutput, void* gradInput, int batchSize, int channels, int inH, int inW, int outH, int outW, int kSize, int stride, int padding);
void maxPool2DPersistent(void* device, void* input, void* output, void* argmax, int batchSize, int channels, int inH, int inW, int outH, int outW, int kSize, int stride, int padding);
void maxPool2DBackwardPersistent(void* device, void* gradOutput, void* gradInput, void* argmax, int batchSize, int channels, int inH, int inW, int outH, int outW, int kSize, int stride, int padding);
void dropoutPersistent(void* device, void* input, void* output, void* mask, int length, float p, bool training, unsigned long long seed);
void dropoutBackwardPersistent(void* device, void* gradOutput, void* gradInput, void* mask, int length, float p);
void batchNorm2DForwardPersistent(void* device, void* input, void* output, void* runningMean, void* runningVar, void* savedMean, void* savedStd, void* gamma, void* beta, int batchSize, int channels, int spatialSize, float eps, float momentum, bool training, bool affine);
void batchNorm2DBackwardPersistent(void* device, void* gradOutput, void* input, void* gradInput, void* savedMean, void* savedStd, void* gamma, void* gradGamma, void* gradBeta, int batchSize, int channels, int spatialSize, float eps, float momentum, bool affine);
void batchNorm2DStatsPersistent(void* device, void* input, void* mean, void* var, int batchSize, int channels, int spatialSize, float eps);
void denseBackwardGradWPersistent(void* device, void* gradOutput, void* input, void* gradWeights, int batchSize, int inSize, int outSize);
void denseBackwardGradBPersistent(void* device, void* gradOutput, void* gradBias, int batchSize, int outSize);
void layerNormForwardPersistent(void* device, void* input, void* output, void* gamma, void* beta, int batchSize, int featureSize, float eps);
void layerNormBackwardPersistent(void* device, void* gradOutput, void* input, void* gradInput, void* gamma, int batchSize, int featureSize, float eps);
void rmsNormForwardPersistent(void* device, void* input, void* output, void* gamma, int batchSize, int featureSize, float eps);
void rmsNormBackwardPersistent(void* device, void* gradOutput, void* input, void* gradInput, void* gamma, int batchSize, int featureSize, float eps);
void softmaxForwardPersistent(void* device, void* input, void* output, int batchSize, int featureSize);
void softmaxBackwardPersistent(void* device, void* gradOutput, void* output, void* gradInput, int batchSize, int featureSize);
void* createConv2DState(void* device, int inC, int outC, int kSize, int stride, int padding, float* weights, float* biases);
void conv2dForwardMPS(void* device, void* state, void* bufIn, void* bufOut, int batchSize, int inH, int inW, int outH, int outW);
void conv2dBackwardMPS(void* device, void* state, void* bufIn, void* bufGradOut, void* bufGradIn, void* bufGradW, void* bufGradB, int batchSize, int inH, int inW, int outH, int outW);
void conv2dReloadWeightsMPS(void* state);
void freeConv2DState(void* state);
void waitForMetal(void* device);
void matMulMPS(void* device, float* A, float* B, float* C, int M, int N, int K);
void matMulFused(void* device, float* A, float* B, float* Bias, float* C, int M, int N, int K, int activation);
void applyActivationMPS(void* device, float* data, int length, int activation);
void lstmStepFused(void* device, float* preAct, float* cPrev, float* cNew, float* hNew, float* iGate, float* fGate, float* gGate, float* oGate, int outSize);
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// Activation types for fused kernels
const (
	MetalActivationNone    = 0
	MetalActivationSigmoid = 1
	MetalActivationTanh    = 2
	MetalActivationReLU    = 3
	MetalActivationGELU    = 4
)

// MetalDevice handles computations on Apple Silicon GPU via Metal Performance Shaders.
type MetalDevice struct {
	ptr unsafe.Pointer
	mu  sync.Mutex
}

func NewMetalDevice() *MetalDevice {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		return &MetalDevice{}
	}
	return &MetalDevice{ptr: C.initMetalDevice()}
}

func (d *MetalDevice) Type() DeviceType { return GPU }

func (d *MetalDevice) IsAvailable() bool {
	if d.ptr == nil {
		return false
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return bool(C.isMetalAvailable(d.ptr))
}

func (d *MetalDevice) Close() {
	if d.ptr != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.freeMetalDevice(d.ptr)
		d.ptr = nil
	}
}

// MetalBuffer represents a buffer on the GPU.
type MetalBuffer struct {
	ptr    unsafe.Pointer
	device *MetalDevice
	length int
}

func (d *MetalDevice) CreateBuffer(data []float32) *MetalBuffer {
	if d.ptr == nil || len(data) == 0 {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return &MetalBuffer{
		ptr:    C.createMetalBuffer(d.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))),
		device: d,
		length: len(data),
	}
}

func (d *MetalDevice) CreateEmptyBuffer(length int) *MetalBuffer {
	if d.ptr == nil || length <= 0 {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return &MetalBuffer{
		ptr:    C.createEmptyMetalBuffer(d.ptr, C.int(length)),
		device: d,
		length: length,
	}
}

func (b *MetalBuffer) Update(data []float32) {
	if b == nil || b.ptr == nil || len(data) == 0 {
		return
	}
	if b.device != nil {
		b.device.mu.Lock()
		defer b.device.mu.Unlock()
		C.waitForMetal(b.device.ptr)
		// Bounds checking
		writeLen := len(data)
		if writeLen > b.length {
			writeLen = b.length
		}
		C.updateMetalBuffer(b.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(writeLen))
	}
}

func (b *MetalBuffer) Read(data []float32) {
	if b == nil || b.ptr == nil || len(data) == 0 {
		return
	}
	if b.device != nil {
		b.device.mu.Lock()
		defer b.device.mu.Unlock()
		C.waitForMetal(b.device.ptr)
		// Bounds checking
		readLen := len(data)
		if readLen > b.length {
			readLen = b.length
		}
		C.readMetalBuffer(b.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(readLen))
	}
}

func (d *MetalDevice) CreateIntBuffer(data []int32) *MetalBuffer {
	if d.ptr == nil || len(data) == 0 {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return &MetalBuffer{
		ptr:    C.createMetalBuffer(d.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data))),
		device: d,
		length: len(data),
	}
}

func (b *MetalBuffer) ReadInt(data []int32) {
	if b.ptr != nil && len(data) > 0 {
		if b.device != nil {
			b.device.mu.Lock()
			defer b.device.mu.Unlock()
			C.waitForMetal(b.device.ptr)
			readLen := len(data)
			if readLen > b.length {
				readLen = b.length
			}
			C.readMetalBuffer(b.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(readLen))
		}
	}
}

func (b *MetalBuffer) Free() {
	if b.ptr != nil {
		if b.device != nil {
			b.device.mu.Lock()
			defer b.device.mu.Unlock()
		}
		C.freeMetalBuffer(b.ptr)
		b.ptr = nil
	}
}

func (d *MetalDevice) MatMulPersistent(A, B, bufC *MetalBuffer, M, N, K int, beta float32) {
	if d.ptr != nil && A != nil && B != nil && bufC != nil {
		// Safety: Validate buffer lengths against dimensions
		if A.length < M*K || B.length < K*N || bufC.length < M*N {
			return
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.matMulMPSPersistent(d.ptr, A.ptr, B.ptr, bufC.ptr, C.int(M), C.int(N), C.int(K), C.float(beta), false, false)
	}
}

func (d *MetalDevice) MatMulTransposePersistent(A, B, bufC *MetalBuffer, M, N, K int, beta float32, transA, transB bool) {
	if d.ptr != nil && A != nil && B != nil && bufC != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.matMulMPSPersistent(d.ptr, A.ptr, B.ptr, bufC.ptr, C.int(M), C.int(N), C.int(K), C.float(beta), C.bool(transA), C.bool(transB))
	}
}

func (d *MetalDevice) MatMulFusedPersistent(A, B, Bias, bufC *MetalBuffer, M, N, K int, activation int) {
	if d.ptr != nil && A != nil && B != nil && bufC != nil {
		// Safety: Validate buffer lengths against dimensions
		if A.length < M*K || B.length < K*N || bufC.length < M*N {
			return
		}
		if Bias != nil && Bias.length < N {
			return
		}
		var biasPtr unsafe.Pointer
		if Bias != nil {
			biasPtr = Bias.ptr
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.matMulFusedPersistent(d.ptr, A.ptr, B.ptr, biasPtr, bufC.ptr, C.int(M), C.int(N), C.int(K), C.int(activation))
	}
}

func (d *MetalDevice) LSTMStepPersistent(preAct, cPrev, cNew, hNew, iGate, fGate, gGate, oGate *MetalBuffer, outSize int) {
	if d.ptr != nil {
		// Safety: Validate all buffer lengths
		if preAct == nil || preAct.length < 4*outSize ||
			cPrev == nil || cPrev.length < outSize ||
			cNew == nil || cNew.length < outSize ||
			hNew == nil || hNew.length < outSize ||
			iGate == nil || iGate.length < outSize ||
			fGate == nil || fGate.length < outSize ||
			gGate == nil || gGate.length < outSize ||
			oGate == nil || oGate.length < outSize {
			return
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.lstmStepFusedPersistent(d.ptr, preAct.ptr, cPrev.ptr, cNew.ptr, hNew.ptr, iGate.ptr, fGate.ptr, gGate.ptr, oGate.ptr, C.int(outSize))
	}
}

func (d *MetalDevice) AvgPool2DPersistent(input, output *MetalBuffer, batchSize, channels, inH, inW, outH, outW, kSize, stride, padding int) {
	if d.ptr != nil && input != nil && output != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.avgPool2DPersistent(d.ptr, input.ptr, output.ptr, C.int(batchSize), C.int(channels), C.int(inH), C.int(inW), C.int(outH), C.int(outW), C.int(kSize), C.int(stride), C.int(padding))
	}
}

func (d *MetalDevice) AvgPool2DBackwardPersistent(gradOutput, gradInput *MetalBuffer, batchSize, channels, inH, inW, outH, outW, kSize, stride, padding int) {
	if d.ptr != nil && gradOutput != nil && gradInput != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.avgPool2DBackwardPersistent(d.ptr, gradOutput.ptr, gradInput.ptr, C.int(batchSize), C.int(channels), C.int(inH), C.int(inW), C.int(outH), C.int(outW), C.int(kSize), C.int(stride), C.int(padding))
	}
}

func (d *MetalDevice) MaxPool2DPersistent(input, output, argmax *MetalBuffer, batchSize, channels, inH, inW, outH, outW, kSize, stride, padding int) {
	if d.ptr != nil && input != nil && output != nil && argmax != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.maxPool2DPersistent(d.ptr, input.ptr, output.ptr, argmax.ptr, C.int(batchSize), C.int(channels), C.int(inH), C.int(inW), C.int(outH), C.int(outW), C.int(kSize), C.int(stride), C.int(padding))
	}
}

func (d *MetalDevice) MaxPool2DBackwardPersistent(gradOutput, gradInput, argmax *MetalBuffer, batchSize, channels, inH, inW, outH, outW, kSize, stride, padding int) {
	if d.ptr != nil && gradOutput != nil && gradInput != nil && argmax != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.maxPool2DBackwardPersistent(d.ptr, gradOutput.ptr, gradInput.ptr, argmax.ptr, C.int(batchSize), C.int(channels), C.int(inH), C.int(inW), C.int(outH), C.int(outW), C.int(kSize), C.int(stride), C.int(padding))
	}
}

func (d *MetalDevice) DropoutPersistent(input, output, mask *MetalBuffer, length int, p float32, training bool, seed uint64) {
	if d.ptr != nil && input != nil && output != nil && mask != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.dropoutPersistent(d.ptr, input.ptr, output.ptr, mask.ptr, C.int(length), C.float(p), C.bool(training), C.ulonglong(seed))
	}
}

func (d *MetalDevice) DropoutBackwardPersistent(gradOutput, gradInput, mask *MetalBuffer, length int, p float32) {
	if d.ptr != nil && gradOutput != nil && gradInput != nil && mask != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.dropoutBackwardPersistent(d.ptr, gradOutput.ptr, gradInput.ptr, mask.ptr, C.int(length), C.float(p))
	}
}

func (d *MetalDevice) BatchNorm2DForwardPersistent(input, output, runningMean, runningVar, savedMean, savedStd, gamma, beta *MetalBuffer, batchSize, channels, spatialSize int, eps, momentum float32, training, affine bool) {
	if d.ptr != nil && input != nil && output != nil && runningMean != nil && runningVar != nil && savedMean != nil && savedStd != nil {
		var gPtr, bPtr unsafe.Pointer
		if gamma != nil {
			gPtr = gamma.ptr
		}
		if beta != nil {
			bPtr = beta.ptr
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.batchNorm2DForwardPersistent(d.ptr, input.ptr, output.ptr, runningMean.ptr, runningVar.ptr, savedMean.ptr, savedStd.ptr, gPtr, bPtr, C.int(batchSize), C.int(channels), C.int(spatialSize), C.float(eps), C.float(momentum), C.bool(training), C.bool(affine))
	}
}

func (d *MetalDevice) BatchNorm2DBackwardPersistent(gradOutput, input, gradInput, savedMean, savedStd, gamma, gradGamma, gradBeta *MetalBuffer, batchSize, channels, spatialSize int, eps, momentum float32, affine bool) {
	if d.ptr != nil && gradOutput != nil && input != nil && gradInput != nil && savedMean != nil && savedStd != nil {
		var gPtr, gGPtr, gBPtr unsafe.Pointer
		if gamma != nil {
			gPtr = gamma.ptr
		}
		if gradGamma != nil {
			gGPtr = gradGamma.ptr
		}
		if gradBeta != nil {
			gBPtr = gradBeta.ptr
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.batchNorm2DBackwardPersistent(d.ptr, gradOutput.ptr, input.ptr, gradInput.ptr, savedMean.ptr, savedStd.ptr, gPtr, gGPtr, gBPtr, C.int(batchSize), C.int(channels), C.int(spatialSize), C.float(eps), C.float(momentum), C.bool(affine))
	}
}

func (d *MetalDevice) BatchNorm2DStatsPersistent(input, mean, varBuf *MetalBuffer, batchSize, channels, spatialSize int, eps float32) {
	if d.ptr != nil && input != nil && mean != nil && varBuf != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.batchNorm2DStatsPersistent(d.ptr, input.ptr, mean.ptr, varBuf.ptr, C.int(batchSize), C.int(channels), C.int(spatialSize), C.float(eps))
	}
}

func (d *MetalDevice) DenseBackwardPersistent(gradOutput, input, gradWeights, gradBias *MetalBuffer, batchSize, inSize, outSize int) {
	if d.ptr != nil && gradOutput != nil && input != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		if gradWeights != nil {
			C.denseBackwardGradWPersistent(d.ptr, gradOutput.ptr, input.ptr, gradWeights.ptr, C.int(batchSize), C.int(inSize), C.int(outSize))
		}
		if gradBias != nil {
			C.denseBackwardGradBPersistent(d.ptr, gradOutput.ptr, gradBias.ptr, C.int(batchSize), C.int(outSize))
		}
	}
}

func (d *MetalDevice) LayerNormForwardPersistent(input, output, gamma, beta *MetalBuffer, batchSize, featureSize int, eps float32) {
	if d.ptr != nil && input != nil && output != nil {
		var gPtr, bPtr unsafe.Pointer
		if gamma != nil {
			gPtr = gamma.ptr
		}
		if beta != nil {
			bPtr = beta.ptr
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.layerNormForwardPersistent(d.ptr, input.ptr, output.ptr, gPtr, bPtr, C.int(batchSize), C.int(featureSize), C.float(eps))
	}
}

func (d *MetalDevice) LayerNormBackwardPersistent(gradOutput, input, gradInput, gamma *MetalBuffer, batchSize, featureSize int, eps float32) {
	if d.ptr != nil && gradOutput != nil && input != nil && gradInput != nil {
		var gPtr unsafe.Pointer
		if gamma != nil {
			gPtr = gamma.ptr
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.layerNormBackwardPersistent(d.ptr, gradOutput.ptr, input.ptr, gradInput.ptr, gPtr, C.int(batchSize), C.int(featureSize), C.float(eps))
	}
}

func (d *MetalDevice) RMSNormForwardPersistent(input, output, gamma *MetalBuffer, batchSize, featureSize int, eps float32) {
	if d.ptr != nil && input != nil && output != nil {
		var gPtr unsafe.Pointer
		if gamma != nil {
			gPtr = gamma.ptr
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.rmsNormForwardPersistent(d.ptr, input.ptr, output.ptr, gPtr, C.int(batchSize), C.int(featureSize), C.float(eps))
	}
}

func (d *MetalDevice) RMSNormBackwardPersistent(gradOutput, input, gradInput, gamma *MetalBuffer, batchSize, featureSize int, eps float32) {
	if d.ptr != nil && gradOutput != nil && input != nil && gradInput != nil {
		var gPtr unsafe.Pointer
		if gamma != nil {
			gPtr = gamma.ptr
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.rmsNormBackwardPersistent(d.ptr, gradOutput.ptr, input.ptr, gradInput.ptr, gPtr, C.int(batchSize), C.int(featureSize), C.float(eps))
	}
}

func (d *MetalDevice) SoftmaxForwardPersistent(input, output *MetalBuffer, batchSize, featureSize int) {
	if d.ptr != nil && input != nil && output != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.softmaxForwardPersistent(d.ptr, input.ptr, output.ptr, C.int(batchSize), C.int(featureSize))
	}
}

func (d *MetalDevice) SoftmaxBackwardPersistent(gradOutput, output, gradInput *MetalBuffer, batchSize, featureSize int) {
	if d.ptr != nil && gradOutput != nil && output != nil && gradInput != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.softmaxBackwardPersistent(d.ptr, gradOutput.ptr, output.ptr, gradInput.ptr, C.int(batchSize), C.int(featureSize))
	}
}

func (d *MetalDevice) CreateConv2DState(inC, outC, kSize, stride, padding int, weights, biases []float32) unsafe.Pointer {
	if d.ptr == nil {
		return nil
	}
	var wPtr, bPtr *C.float
	if len(weights) > 0 {
		wPtr = (*C.float)(unsafe.Pointer(&weights[0]))
	}
	if len(biases) > 0 {
		bPtr = (*C.float)(unsafe.Pointer(&biases[0]))
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return C.createConv2DState(d.ptr, C.int(inC), C.int(outC), C.int(kSize), C.int(stride), C.int(padding), wPtr, bPtr)
}

func (d *MetalDevice) Conv2DForward(state unsafe.Pointer, input, output *MetalBuffer, batchSize, inH, inW, outH, outW int) {
	if d.ptr != nil && state != nil && input != nil && output != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.conv2dForwardMPS(d.ptr, state, input.ptr, output.ptr, C.int(batchSize), C.int(inH), C.int(inW), C.int(outH), C.int(outW))
	}
}

func (d *MetalDevice) Conv2DBackward(state unsafe.Pointer, input, gradOutput, gradInput, gradWeights, gradBiases *MetalBuffer, batchSize, inH, inW, outH, outW int) {
	if d.ptr != nil && state != nil && input != nil && gradOutput != nil && gradInput != nil {
		var gWPtr, gBPtr unsafe.Pointer
		if gradWeights != nil {
			gWPtr = gradWeights.ptr
		}
		if gradBiases != nil {
			gBPtr = gradBiases.ptr
		}
		d.mu.Lock()
		defer d.mu.Unlock()
		C.conv2dBackwardMPS(d.ptr, state, input.ptr, gradOutput.ptr, gradInput.ptr, gWPtr, gBPtr, C.int(batchSize), C.int(inH), C.int(inW), C.int(outH), C.int(outW))
	}
}

func (d *MetalDevice) Conv2DReloadWeights(state unsafe.Pointer) {
	if state != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.conv2dReloadWeightsMPS(state)
	}
}

func (d *MetalDevice) FreeConv2DState(state unsafe.Pointer) {
	if state != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.freeConv2DState(state)
	}
}

func (d *MetalDevice) Wait() {
	if d.ptr != nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		C.waitForMetal(d.ptr)
	}
}

// MatMul performs matrix multiplication C = A * B using MPS.
// A: M x K, B: K x N, C: M x N
func (d *MetalDevice) MatMul(A, B, C []float32, M, N, K int) {
	d.mu.Lock()
	defer d.mu.Unlock()
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

	d.mu.Lock()
	defer d.mu.Unlock()
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
	d.mu.Lock()
	defer d.mu.Unlock()
	C.applyActivationMPS(d.ptr, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)), C.int(activation))
}

// LSTMStep performs a fused LSTM forward step on GPU.
func (d *MetalDevice) LSTMStep(preAct, cPrev, cNew, hNew, iGate, fGate, gGate, oGate []float32, outSize int) {
	d.mu.Lock()
	defer d.mu.Unlock()
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
