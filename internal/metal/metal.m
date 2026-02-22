#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import "metal.h"

@interface MetalState : NSObject
@property (strong) id<MTLDevice> device;
@property (strong) id<MTLCommandQueue> commandQueue;
@end

@implementation MetalState
@end

bool IsMetalAvailable() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

MetalContext InitMetal() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return NULL;

        MetalState *state = [[MetalState alloc] init];
        state.device = device;
        state.commandQueue = [device newCommandQueue];

        return (MetalContext)CFBridgingRetain(state);
    }
}

void FreeMetal(MetalContext ctx) {
    @autoreleasepool {
        id state = CFBridgingRelease(ctx);
        state = nil;
    }
}

MetalBuffer CreateSharedBuffer(MetalContext ctx, size_t size) {
    @autoreleasepool {
        MetalState *state = (__bridge MetalState *)ctx;
        id<MTLBuffer> buffer = [state.device newBufferWithLength:size
                                                      options:MTLResourceStorageModeShared];
        return (MetalBuffer)CFBridgingRetain(buffer);
    }
}

void FreeBuffer(MetalBuffer buf) {
    @autoreleasepool {
        id buffer = CFBridgingRelease(buf);
        buffer = nil;
    }
}

void* GetBufferContents(MetalBuffer buf) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
    return [buffer contents];
}

void MPSMatMul(MetalContext ctx,
               MetalBuffer A, MetalBuffer B, MetalBuffer C,
               int M, int N, int K) {
    @autoreleasepool {
        MetalState *state = (__bridge MetalState *)ctx;
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)A;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)B;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)C;

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:K
                                                                          rowBytes:K * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                           columns:N
                                                                          rowBytes:N * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:N
                                                                          rowBytes:N * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication *mul = [[MPSMatrixMultiplication alloc] initWithDevice:state.device
                                                                         transposeLeft:NO
                                                                        transposeRight:NO
                                                                            resultRows:M
                                                                         resultColumns:N
                                                                       interiorColumns:K
                                                                                 alpha:1.0
                                                                                  beta:0.0];

        id<MTLCommandBuffer> commandBuffer = [state.commandQueue commandBuffer];
        [mul encodeToCommandBuffer:commandBuffer leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}
