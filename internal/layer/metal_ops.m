#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdbool.h>

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

void matMulMPS(void* ptr, float* A, float* B, float* C, int M, int N, int K) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;

    @autoreleasepool {
        int sizeA = M * K;
        int sizeB = K * N;
        int sizeC = M * N;

        id<MTLBuffer> bufferA = [ctx->device newBufferWithBytes:A length:sizeA * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [ctx->device newBufferWithBytes:B length:sizeB * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [ctx->device newBufferWithBytes:C length:sizeC * sizeof(float) options:MTLResourceStorageModeShared];

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

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

        // Copy results back
        memcpy(C, [bufferC contents], sizeC * sizeof(float));
    }
}
