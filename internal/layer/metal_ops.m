#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdbool.h>

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> lstmPipelineState;
    id<MTLComputePipelineState> activationPipelineState;
    id<MTLComputePipelineState> biasActivationPipelineState;
} MetalContext;

typedef enum {
    ActivationNone = 0,
    ActivationSigmoid = 1,
    ActivationTanh = 2,
    ActivationReLU = 3
} ActivationType;

static NSString *const metal_src = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void lstm_step(\n"
"    device const float* pre_activations [[buffer(0)]],\n"
"    device const float* c_prev [[buffer(1)]],\n"
"    device float* c_new [[buffer(2)]],\n"
"    device float* h_new [[buffer(3)]],\n"
"    device float* i_gate [[buffer(4)]],\n"
"    device float* f_gate [[buffer(5)]],\n"
"    device float* g_gate [[buffer(6)]],\n"
"    device float* o_gate [[buffer(7)]],\n"
"    constant uint& outSize [[buffer(8)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid >= outSize) return;\n"
"    float i_pre = pre_activations[gid];\n"
"    float f_pre = pre_activations[outSize + gid];\n"
"    float g_pre = pre_activations[2 * outSize + gid];\n"
"    float o_pre = pre_activations[3 * outSize + gid];\n"
"    float i = 1.0f / (1.0f + exp(-i_pre));\n"
"    float f = 1.0f / (1.0f + exp(-f_pre));\n"
"    float g = tanh(g_pre);\n"
"    float o = 1.0f / (1.0f + exp(-o_pre));\n"
"    float c = f * c_prev[gid] + i * g;\n"
"    float h = o * tanh(c);\n"
"    c_new[gid] = c;\n"
"    h_new[gid] = h;\n"
"    i_gate[gid] = i;\n"
"    f_gate[gid] = f;\n"
"    g_gate[gid] = g;\n"
"    o_gate[gid] = o;\n"
"}\n"
"kernel void bias_activation(\n"
"    device float* data [[buffer(0)]],\n"
"    device const float* bias [[buffer(1)]],\n"
"    constant uint& M [[buffer(2)]],\n"
"    constant uint& N [[buffer(3)]],\n"
"    constant int& type [[buffer(4)]],\n"
"    constant bool& hasBias [[buffer(5)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid >= M * N) return;\n"
"    uint col = gid % N;\n"
"    float x = data[gid];\n"
"    if (hasBias) x += bias[col];\n"
"    if (type == 1) { // Sigmoid\n"
"        data[gid] = 1.0f / (1.0f + exp(-x));\n"
"    } else if (type == 2) { // Tanh\n"
"        data[gid] = tanh(x);\n"
"    } else if (type == 3) { // ReLU\n"
"        data[gid] = x > 0.0f ? x : 0.0f;\n"
"    } else {\n"
"        data[gid] = x;\n"
"    }\n"
"}\n";

void* initMetalDevice() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return NULL;

    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    ctx->device = device;
    ctx->commandQueue = [device newCommandQueue];
    ctx->lstmPipelineState = nil;
    ctx->activationPipelineState = nil;
    ctx->biasActivationPipelineState = nil;

    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:metal_src options:nil error:&error];
    if (library) {
        id<MTLFunction> lstmFunc = [library newFunctionWithName:@"lstm_step"];
        if (lstmFunc) {
            ctx->lstmPipelineState = [device newComputePipelineStateWithFunction:lstmFunc error:&error];
        }
        id<MTLFunction> actFunc = [library newFunctionWithName:@"bias_activation"];
        if (actFunc) {
            ctx->biasActivationPipelineState = [device newComputePipelineStateWithFunction:actFunc error:&error];
            // Backward compatibility
            ctx->activationPipelineState = ctx->biasActivationPipelineState;
        }
    }

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

void* createMetalBuffer(void* ptr, float* data, int length) {
    if (!ptr) return NULL;
    MetalContext* ctx = (MetalContext*)ptr;
    id<MTLBuffer> buffer = [ctx->device newBufferWithBytes:data length:length * sizeof(float) options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buffer;
}

void* createEmptyMetalBuffer(void* ptr, int length) {
    if (!ptr) return NULL;
    MetalContext* ctx = (MetalContext*)ptr;
    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:length * sizeof(float) options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buffer;
}

void updateMetalBuffer(void* bufferPtr, float* data, int length) {
    if (!bufferPtr) return;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferPtr;
    // Size validation
    NSUInteger maxLen = [buffer length] / sizeof(float);
    if (length > (int)maxLen) length = (int)maxLen;
    memcpy([buffer contents], data, length * sizeof(float));
}

void readMetalBuffer(void* bufferPtr, float* data, int length) {
    if (!bufferPtr) return;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferPtr;
    // Size validation
    NSUInteger maxLen = [buffer length] / sizeof(float);
    if (length > (int)maxLen) length = (int)maxLen;
    memcpy(data, [buffer contents], length * sizeof(float));
}

void freeMetalBuffer(void* bufferPtr) {
    if (!bufferPtr) return;
    id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)bufferPtr;
    buffer = nil;
}

void matMulFusedPersistent(void* ptr, void* bufA, void* bufB, void* bufBias, void* bufC, int M, int N, int K, int activation) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;

    @autoreleasepool {
        id<MTLBuffer> bufferA = (__bridge id<MTLBuffer>)bufA;
        id<MTLBuffer> bufferB = (__bridge id<MTLBuffer>)bufB;
        id<MTLBuffer> bufferBias = (__bridge id<MTLBuffer>)bufBias;
        id<MTLBuffer> bufferC = (__bridge id<MTLBuffer>)bufC;

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

        if ((bufferBias || activation != ActivationNone) && ctx->biasActivationPipelineState) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:ctx->biasActivationPipelineState];
            [encoder setBuffer:bufferC offset:0 atIndex:0];
            [encoder setBuffer:bufferBias ? bufferBias : bufferC offset:0 atIndex:1];
            uint mVal = (uint)M;
            uint nVal = (uint)N;
            bool hasBias = bufferBias != nil;
            [encoder setBytes:&mVal length:sizeof(uint) atIndex:2];
            [encoder setBytes:&nVal length:sizeof(uint) atIndex:3];
            [encoder setBytes:&activation length:sizeof(int) atIndex:4];
            [encoder setBytes:&hasBias length:sizeof(bool) atIndex:5];

            uint total = M * N;
            MTLSize gridSize = MTLSizeMake(total, 1, 1);
            NSUInteger threadGroupSize = ctx->biasActivationPipelineState.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > total) threadGroupSize = total;
            MTLSize threadGroupSizeMTL = MTLSizeMake(threadGroupSize, 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSizeMTL];
            [encoder endEncoding];
        }

        [commandBuffer commit];
    }
}

void matMulMPSPersistent(void* ptr, void* bufA, void* bufB, void* bufC, int M, int N, int K, float beta) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;

    @autoreleasepool {
        id<MTLBuffer> bufferA = (__bridge id<MTLBuffer>)bufA;
        id<MTLBuffer> bufferB = (__bridge id<MTLBuffer>)bufB;
        id<MTLBuffer> bufferC = (__bridge id<MTLBuffer>)bufC;

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
                                                                                     beta:beta];

        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        [kernel encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        [commandBuffer commit];
        // Note: No waitUntilCompleted here to allow async execution
    }
}

void lstmStepFusedPersistent(void* ptr, void* bufPreAct, void* bufCPrev, void* bufCNew, void* bufHNew,
                          void* bufI, void* bufF, void* bufG, void* bufO, int outSize) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;
    if (!ctx->lstmPipelineState) return;

    @autoreleasepool {
        id<MTLBuffer> bPreAct = (__bridge id<MTLBuffer>)bufPreAct;
        id<MTLBuffer> bCPrev = (__bridge id<MTLBuffer>)bufCPrev;
        id<MTLBuffer> bCNew = (__bridge id<MTLBuffer>)bufCNew;
        id<MTLBuffer> bHNew = (__bridge id<MTLBuffer>)bufHNew;
        id<MTLBuffer> bI = (__bridge id<MTLBuffer>)bufI;
        id<MTLBuffer> bF = (__bridge id<MTLBuffer>)bufF;
        id<MTLBuffer> bG = (__bridge id<MTLBuffer>)bufG;
        id<MTLBuffer> bO = (__bridge id<MTLBuffer>)bufO;

        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:ctx->lstmPipelineState];
        [encoder setBuffer:bPreAct offset:0 atIndex:0];
        [encoder setBuffer:bCPrev offset:0 atIndex:1];
        [encoder setBuffer:bCNew offset:0 atIndex:2];
        [encoder setBuffer:bHNew offset:0 atIndex:3];
        [encoder setBuffer:bI offset:0 atIndex:4];
        [encoder setBuffer:bF offset:0 atIndex:5];
        [encoder setBuffer:bG offset:0 atIndex:6];
        [encoder setBuffer:bO offset:0 atIndex:7];
        [encoder setBytes:&outSize length:sizeof(int) atIndex:8];

        MTLSize gridSize = MTLSizeMake(outSize, 1, 1);
        NSUInteger threadGroupSize = ctx->lstmPipelineState.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > outSize) threadGroupSize = outSize;
        MTLSize threadGroupSizeMTL = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSizeMTL];
        [encoder endEncoding];
        [commandBuffer commit];
        // [commandBuffer waitUntilCompleted]; // Removed for async performance
    }
}

void waitForMetal(void* ptr) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;
    id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void matMulMPS(void* ptr, float* A, float* B, float* C, int M, int N, int K) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;

    @autoreleasepool {
        id<MTLBuffer> bufferA = [ctx->device newBufferWithBytes:A length:M * K * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [ctx->device newBufferWithBytes:B length:K * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [ctx->device newBufferWithBytes:C length:M * N * sizeof(float) options:MTLResourceStorageModeShared];

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

        memcpy(C, [bufferC contents], M * N * sizeof(float));
    }
}

void matMulFused(void* ptr, float* A, float* B, float* Bias, float* C, int M, int N, int K, int activation) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;

    @autoreleasepool {
        int sizeA = M * K;
        int sizeB = K * N;
        int sizeC = M * N;

        id<MTLBuffer> bufferA = [ctx->device newBufferWithBytes:A length:sizeA * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [ctx->device newBufferWithBytes:B length:sizeB * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [ctx->device newBufferWithLength:sizeC * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferBias = nil;
        if (Bias) {
            bufferBias = [ctx->device newBufferWithBytes:Bias length:N * sizeof(float) options:MTLResourceStorageModeShared];
        }

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

        if ((bufferBias || activation != ActivationNone) && ctx->biasActivationPipelineState) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:ctx->biasActivationPipelineState];
            [encoder setBuffer:bufferC offset:0 atIndex:0];
            [encoder setBuffer:bufferBias ? bufferBias : bufferC offset:0 atIndex:1];
            bool hasBias = bufferBias != nil;
            [encoder setBytes:&M length:sizeof(int) atIndex:2];
            [encoder setBytes:&N length:sizeof(int) atIndex:3];
            [encoder setBytes:&activation length:sizeof(int) atIndex:4];
            [encoder setBytes:&hasBias length:sizeof(bool) atIndex:5];

            int totalElements = M * N;
            MTLSize gridSize = MTLSizeMake(totalElements, 1, 1);
            NSUInteger threadGroupSize = ctx->biasActivationPipelineState.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > totalElements) threadGroupSize = totalElements;
            MTLSize threadGroupSizeMTL = MTLSizeMake(threadGroupSize, 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSizeMTL];
            [encoder endEncoding];
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        memcpy(C, [bufferC contents], sizeC * sizeof(float));
    }
}

void applyActivationMPS(void* ptr, float* data, int length, int activation) {
    if (!ptr || activation == ActivationNone) return;
    MetalContext* ctx = (MetalContext*)ptr;
    if (!ctx->biasActivationPipelineState) return;

    @autoreleasepool {
        id<MTLBuffer> buffer = [ctx->device newBufferWithBytes:data length:length * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:ctx->biasActivationPipelineState];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBuffer:buffer offset:0 atIndex:1]; // Dummy bias
        int M = 1;
        int N = length;
        bool hasBias = false;
        [encoder setBytes:&M length:sizeof(int) atIndex:2];
        [encoder setBytes:&N length:sizeof(int) atIndex:3];
        [encoder setBytes:&activation length:sizeof(int) atIndex:4];
        [encoder setBytes:&hasBias length:sizeof(bool) atIndex:5];

        MTLSize gridSize = MTLSizeMake(length, 1, 1);
        NSUInteger threadGroupSize = ctx->biasActivationPipelineState.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > length) threadGroupSize = length;
        MTLSize threadGroupSizeMTL = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSizeMTL];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        memcpy(data, [buffer contents], length * sizeof(float));
    }
}

void lstmStepFused(void* ptr, float* preAct, float* cPrev, float* cNew, float* hNew,
                  float* iGate, float* fGate, float* gGate, float* oGate, int outSize) {
    if (!ptr) return;
    MetalContext* ctx = (MetalContext*)ptr;
    if (!ctx->lstmPipelineState) return;

    @autoreleasepool {
        id<MTLBuffer> bufPreAct = [ctx->device newBufferWithBytes:preAct length:4 * outSize * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufCPrev = [ctx->device newBufferWithBytes:cPrev length:outSize * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufCNew = [ctx->device newBufferWithLength:outSize * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufHNew = [ctx->device newBufferWithLength:outSize * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufI = [ctx->device newBufferWithLength:outSize * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufF = [ctx->device newBufferWithLength:outSize * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufG = [ctx->device newBufferWithLength:outSize * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [ctx->device newBufferWithLength:outSize * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:ctx->lstmPipelineState];
        [encoder setBuffer:bufPreAct offset:0 atIndex:0];
        [encoder setBuffer:bufCPrev offset:0 atIndex:1];
        [encoder setBuffer:bufCNew offset:0 atIndex:2];
        [encoder setBuffer:bufHNew offset:0 atIndex:3];
        [encoder setBuffer:bufI offset:0 atIndex:4];
        [encoder setBuffer:bufF offset:0 atIndex:5];
        [encoder setBuffer:bufG offset:0 atIndex:6];
        [encoder setBuffer:bufO offset:0 atIndex:7];
        [encoder setBytes:&outSize length:sizeof(int) atIndex:8];

        MTLSize gridSize = MTLSizeMake(outSize, 1, 1);
        NSUInteger threadGroupSize = ctx->lstmPipelineState.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > outSize) threadGroupSize = outSize;
        MTLSize threadGroupSizeMTL = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSizeMTL];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        memcpy(cNew, [bufCNew contents], outSize * sizeof(float));
        memcpy(hNew, [bufHNew contents], outSize * sizeof(float));
        memcpy(iGate, [bufI contents], outSize * sizeof(float));
        memcpy(fGate, [bufF contents], outSize * sizeof(float));
        memcpy(gGate, [bufG contents], outSize * sizeof(float));
        memcpy(oGate, [bufO contents], outSize * sizeof(float));
    }
}
