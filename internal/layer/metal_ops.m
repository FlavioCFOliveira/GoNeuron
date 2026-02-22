#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdbool.h>

// Handle cases where MPSDataTypeFloat64 is not defined
#ifndef MPSDataTypeFloat64
#define MPSDataTypeFloat64 (MPSDataType)(MPSDataTypeFloatBit | 64)
#endif

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
} MetalContext;

void* initMetalDevice() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return NULL;

    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    ctx->device = device;
    ctx->commandQueue = [device newCommandQueue];
    return (void*)ctx;
}

bool isMetalAvailable(void* ptr) {
    return ptr != NULL;
}

void freeMetalDevice(void* ptr) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;
    free(ctx);
}

void matMulMPS(void* ptr, double* A, double* B, double* C, int M, int N, int K) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;

    @autoreleasepool {
        // Create buffers pointing to the same unified memory
        id<MTLBuffer> bufferA = [ctx->device newBufferWithBytesNoCopy:A
                                                               length:M * K * sizeof(double)
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        id<MTLBuffer> bufferB = [ctx->device newBufferWithBytesNoCopy:B
                                                               length:K * N * sizeof(double)
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        id<MTLBuffer> bufferC = [ctx->device newBufferWithBytesNoCopy:C
                                                               length:M * N * sizeof(double)
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        // Try to use Float32 if Float64 fails, or use a safe cast
        // Apple Silicon supports Float32 better than Float64 in many MPS kernels
        // but since our library is float64, we'll use the constant value directly or float32 if needed.
        // MPSDataTypeFloat32 = 0x10020
        // MPSDataTypeFloat64 = 0x10040 (65600)

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:K
                                                                          rowBytes:K * sizeof(double)
                                                                          dataType:65600];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                           columns:N
                                                                          rowBytes:N * sizeof(double)
                                                                          dataType:65600];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:N
                                                                          rowBytes:N * sizeof(double)
                                                                          dataType:65600];

        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
                                                                            transposeLeft:NO
                                                                           transposeRight:NO
                                                                               resultRows:M
                                                                            resultColumns:N
                                                                          interiorColumns:K
                                                                                    alpha:1.0
                                                                                     beta:0.0];

        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        [kernel encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}
