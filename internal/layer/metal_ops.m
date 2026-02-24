#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> lstmPipelineState;
    id<MTLComputePipelineState> activationPipelineState;
    id<MTLComputePipelineState> biasActivationPipelineState;
    id<MTLComputePipelineState> conv2dForwardPSO;
    id<MTLComputePipelineState> conv2dGradInPSO;
    id<MTLComputePipelineState> conv2dGradWPSO;
    id<MTLComputePipelineState> conv2dGradBPSO;
    id<MTLComputePipelineState> avgPool2DForwardPSO;
    id<MTLComputePipelineState> avgPool2DBackwardPSO;
    id<MTLComputePipelineState> maxPool2DForwardPSO;
    id<MTLComputePipelineState> maxPool2DBackwardPSO;
    id<MTLComputePipelineState> dropoutForwardPSO;
    id<MTLComputePipelineState> dropoutBackwardPSO;
    id<MTLComputePipelineState> batchNorm2DForwardPSO;
    id<MTLComputePipelineState> batchNorm2DBackwardPSO;
    id<MTLComputePipelineState> batchNorm2DStatsPSO;
    id<MTLComputePipelineState> denseGradBPSO;
    id<MTLComputePipelineState> layerNormForwardPSO;
    id<MTLComputePipelineState> layerNormBackwardPSO;
    id<MTLComputePipelineState> rmsNormForwardPSO;
    id<MTLComputePipelineState> rmsNormBackwardPSO;
    id<MTLComputePipelineState> softmaxForwardPSO;
    id<MTLComputePipelineState> softmaxBackwardPSO;
} MetalContext;

typedef enum {
    ActivationNone = 0,
    ActivationSigmoid = 1,
    ActivationTanh = 2,
    ActivationReLU = 3,
    ActivationGELU = 4
} ActivationType;

typedef struct {
    uint batchSize;
    uint inChannels;
    uint outChannels;
    uint inHeight;
    uint inWidth;
    uint outHeight;
    uint outWidth;
    uint kernelSize;
    uint stride;
    uint padding;
} Conv2DParams;

typedef struct {
    uint batchSize;
    uint channels;
    uint inH;
    uint inW;
    uint outH;
    uint outW;
    uint kSize;
    uint stride;
    uint padding;
} Pool2DParams;

typedef struct {
    uint batchSize;
    uint channels;
    uint spatialSize;
    float eps;
    float momentum;
    bool training;
    bool affine;
} BatchNorm2DParams;

typedef struct {
    uint batchSize;
    uint featureSize;
    float eps;
} LayerNormParams;

typedef struct {
    MetalContext* ctx;
    id<MTLBuffer> weights;
    id<MTLBuffer> bias;
    int inChannels;
    int outChannels;
    int kernelSize;
    int stride;
    int padding;
} Conv2DLayerState;

static NSString *const metal_src = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"typedef struct {\n"
"    uint batchSize;\n"
"    uint channels;\n"
"    uint inH;\n"
"    uint inW;\n"
"    uint outH;\n"
"    uint outW;\n"
"    uint kSize;\n"
"    uint stride;\n"
"    uint padding;\n"
"} Pool2DParams;\n"
"\n"
"typedef struct {\n"
"    uint batchSize;\n"
"    uint channels;\n"
"    uint spatialSize;\n"
"    float eps;\n"
"    float momentum;\n"
"    bool training;\n"
"    bool affine;\n"
"} BatchNorm2DParams;\n"
"\n"
"typedef struct {\n"
"    uint batchSize;\n"
"    uint inChannels;\n"
"    uint outChannels;\n"
"    uint inHeight;\n"
"    uint inWidth;\n"
"    uint outHeight;\n"
"    uint outWidth;\n"
"    uint kernelSize;\n"
"    uint stride;\n"
"    uint padding;\n"
"} Conv2DParams;\n"
"\n"
"typedef struct {\n"
"    uint batchSize;\n"
"    uint featureSize;\n"
"    float eps;\n"
"} LayerNormParams;\n"
"\n"
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
"\n"
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
"    } else if (type == 4) { // GeLU\n"
"        data[gid] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));\n"
"    } else {\n"
"        data[gid] = x;\n"
"    }\n"
"}\n"
"\n"
"kernel void conv2d_forward(\n"
"    device const float* input [[buffer(0)]],\n"
"    device const float* weights [[buffer(1)]],\n"
"    device const float* bias [[buffer(2)]],\n"
"    device float* output [[buffer(3)]],\n"
"    constant Conv2DParams& params [[buffer(4)]],\n"
"    uint3 gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid.x >= params.outWidth || gid.y >= params.outHeight || gid.z >= (params.batchSize * params.outChannels)) return;\n"
"    \n"
"    uint b = gid.z / params.outChannels;\n"
"    uint co = gid.z % params.outChannels;\n"
"    uint oh = gid.y;\n"
"    uint ow = gid.x;\n"
"    \n"
"    float sum = bias[co];\n"
"    \n"
"    for (uint ci = 0; ci < params.inChannels; ci++) {\n"
"        for (uint kh = 0; kh < params.kernelSize; kh++) {\n"
"            int hi = (int)oh * (int)params.stride + (int)kh - (int)params.padding;\n"
"            if (hi >= 0 && hi < (int)params.inHeight) {\n"
"                for (uint kw = 0; kw < params.kernelSize; kw++) {\n"
"                    int wi = (int)ow * (int)params.stride + (int)kw - (int)params.padding;\n"
"                    if (wi >= 0 && wi < (int)params.inWidth) {\n"
"                        uint input_idx = ((b * params.inChannels + ci) * params.inHeight + hi) * params.inWidth + wi;\n"
"                        uint weight_idx = ((co * params.inChannels + ci) * params.kernelSize + kh) * params.kernelSize + kw;\n"
"                        sum += input[input_idx] * weights[weight_idx];\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    \n"
"    uint out_idx = ((b * params.outChannels + co) * params.outHeight + oh) * params.outWidth + ow;\n"
"    output[out_idx] = sum;\n"
"}\n"
"\n"
"kernel void conv2d_grad_input(\n"
"    device float* gradInput [[buffer(0)]],\n"
"    device const float* gradOutput [[buffer(1)]],\n"
"    device const float* weights [[buffer(2)]],\n"
"    constant Conv2DParams& params [[buffer(3)]],\n"
"    uint3 gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid.x >= params.inWidth || gid.y >= params.inHeight || gid.z >= (params.batchSize * params.inChannels)) return;\n"
"    \n"
"    uint b = gid.z / params.inChannels;\n"
"    uint ci = gid.z % params.inChannels;\n"
"    uint ih = gid.y;\n"
"    uint iw = gid.x;\n"
"    \n"
"    float sum = 0.0f;\n"
"    \n"
"    for (uint co = 0; co < params.outChannels; co++) {\n"
"        for (uint kh = 0; kh < params.kernelSize; kh++) {\n"
"            int oh_num = (int)ih + (int)params.padding - (int)kh;\n"
"            if (oh_num >= 0 && oh_num % (int)params.stride == 0) {\n"
"                int oh = oh_num / (int)params.stride;\n"
"                if (oh >= 0 && oh < (int)params.outHeight) {\n"
"                    for (uint kw = 0; kw < params.kernelSize; kw++) {\n"
"                        int ow_num = (int)iw + (int)params.padding - (int)kw;\n"
"                        if (ow_num >= 0 && ow_num % (int)params.stride == 0) {\n"
"                            int ow = ow_num / (int)params.stride;\n"
"                            if (ow >= 0 && ow < (int)params.outWidth) {\n"
"                                uint out_idx = ((b * params.outChannels + co) * params.outHeight + oh) * params.outWidth + ow;\n"
"                                uint weight_idx = ((co * params.inChannels + ci) * params.kernelSize + kh) * params.kernelSize + kw;\n"
"                                sum += gradOutput[out_idx] * weights[weight_idx];\n"
"                            }\n"
"                        }\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    \n"
"    uint in_idx = ((b * params.inChannels + ci) * params.inHeight + ih) * params.inWidth + iw;\n"
"    gradInput[in_idx] = sum;\n"
"}\n"
"\n"
"kernel void conv2d_grad_weights(\n"
"    device float* gradWeights [[buffer(0)]],\n"
"    device const float* gradOutput [[buffer(1)]],\n"
"    device const float* input [[buffer(2)]],\n"
"    constant Conv2DParams& params [[buffer(3)]],\n"
"    uint3 gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid.x >= params.kernelSize || gid.y >= params.kernelSize || gid.z >= (params.outChannels * params.inChannels)) return;\n"
"    \n"
"    uint co = gid.z / params.inChannels;\n"
"    uint ci = gid.z % params.inChannels;\n"
"    uint kh = gid.y;\n"
"    uint kw = gid.x;\n"
"    \n"
"    float sum = 0.0f;\n"
"    \n"
"    for (uint b = 0; b < params.batchSize; b++) {\n"
"        for (uint oh = 0; oh < params.outHeight; oh++) {\n"
"            int hi = (int)oh * (int)params.stride + (int)kh - (int)params.padding;\n"
"            if (hi >= 0 && hi < (int)params.inHeight) {\n"
"                for (uint ow = 0; ow < params.outWidth; ow++) {\n"
"                    int wi = (int)ow * (int)params.stride + (int)kw - (int)params.padding;\n"
"                    if (wi >= 0 && wi < (int)params.inWidth) {\n"
"                        uint out_idx = ((b * params.outChannels + co) * params.outHeight + oh) * params.outWidth + ow;\n"
"                        uint in_idx = ((b * params.inChannels + ci) * params.inHeight + hi) * params.inWidth + wi;\n"
"                        sum += gradOutput[out_idx] * input[in_idx];\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    \n"
"    uint weight_idx = ((co * params.inChannels + ci) * params.kernelSize + kh) * params.kernelSize + kw;\n"
"    gradWeights[weight_idx] = sum;\n"
"}\n"
"\n"
"kernel void conv2d_grad_bias(\n"
"    device float* gradBias [[buffer(0)]],\n"
"    device const float* gradOutput [[buffer(1)]],\n"
"    constant Conv2DParams& params [[buffer(2)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid >= params.outChannels) return;\n"
"    \n"
"    float sum = 0.0f;\n"
"    for (uint b = 0; b < params.batchSize; b++) {\n"
"        for (uint oh = 0; oh < params.outHeight; oh++) {\n"
"            for (uint ow = 0; ow < params.outWidth; ow++) {\n"
"                uint out_idx = ((b * params.outChannels + gid) * params.outHeight + oh) * params.outWidth + ow;\n"
"                sum += gradOutput[out_idx];\n"
"            }\n"
"        }\n"
"    }\n"
"    \n"
"    gradBias[gid] = sum;\n"
"}\n"
"\n"
"kernel void avg_pool2d_forward(\n"
"    device const float* input [[buffer(0)]],\n"
"    device float* output [[buffer(1)]],\n"
"    constant Pool2DParams& params [[buffer(2)]],\n"
"    uint3 gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid.x >= params.outW || gid.y >= params.outH || gid.z >= (params.batchSize * params.channels)) return;\n"
"    \n"
"    uint b = gid.z / params.channels;\n"
"    uint c = gid.z % params.channels;\n"
"    uint oh = gid.y;\n"
"    uint ow = gid.x;\n"
"    \n"
"    float sum = 0.0f;\n"
"    uint count = 0;\n"
"    \n"
"    for (uint kh = 0; kh < params.kSize; kh++) {\n"
"        int hi = (int)oh * (int)params.stride + (int)kh - (int)params.padding;\n"
"        if (hi >= 0 && hi < (int)params.inH) {\n"
"            for (uint kw = 0; kw < params.kSize; kw++) {\n"
"                int wi = (int)ow * (int)params.stride + (int)kw - (int)params.padding;\n"
"                if (wi >= 0 && wi < (int)params.inW) {\n"
"                    uint idx = ((b * params.channels + c) * params.inH + hi) * params.inW + wi;\n"
"                    sum += input[idx];\n"
"                    count++;\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    \n"
"    uint out_idx = ((b * params.channels + c) * params.outH + oh) * params.outW + ow;\n"
"    output[out_idx] = count > 0 ? sum / (float)count : 0.0f;\n"
"}\n"
"\n"
"kernel void avg_pool2d_backward(\n"
"    device const float* gradOutput [[buffer(0)]],\n"
"    device float* gradInput [[buffer(1)]],\n"
"    constant Pool2DParams& params [[buffer(2)]],\n"
"    uint3 gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid.x >= params.inW || gid.y >= params.inH || gid.z >= (params.batchSize * params.channels)) return;\n"
"    \n"
"    uint b = gid.z / params.channels;\n"
"    uint c = gid.z % params.channels;\n"
"    uint ih = gid.y;\n"
"    uint iw = gid.x;\n"
"    \n"
"    float sum = 0.0f;\n"
"    \n"
"    for (uint kh = 0; kh < params.kSize; kh++) {\n"
"        int oh_num = (int)ih + (int)params.padding - (int)kh;\n"
"        if (oh_num >= 0 && oh_num % (int)params.stride == 0) {\n"
"            int oh = oh_num / (int)params.stride;\n"
"            if (oh >= 0 && oh < (int)params.outH) {\n"
"                for (uint kw = 0; kw < params.kSize; kw++) {\n"
"                    int ow_num = (int)iw + (int)params.padding - (int)kw;\n"
"                    if (ow_num >= 0 && ow_num % (int)params.stride == 0) {\n"
"                        int ow = ow_num / (int)params.stride;\n"
"                        if (ow >= 0 && ow < (int)params.outW) {\n"
"                            uint count = 0;\n"
"                            for (uint pkh = 0; pkh < params.kSize; pkh++) {\n"
"                                for (uint pkw = 0; pkw < params.kSize; pkw++) {\n"
"                                    int phi = oh * (int)params.stride + (int)pkh - (int)params.padding;\n"
"                                    int pwi = ow * (int)params.stride + (int)pkw - (int)params.padding;\n"
"                                    if (phi >= 0 && phi < (int)params.inH && pwi >= 0 && pwi < (int)params.inW) count++;\n"
"                                }\n"
"                            }\n"
"                            if (count > 0) {\n"
"                                uint out_idx = ((b * params.channels + c) * params.outH + oh) * params.outW + ow;\n"
"                                sum += gradOutput[out_idx] / (float)count;\n"
"                            }\n"
"                        }\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    \n"
"    uint in_idx = ((b * params.channels + c) * params.inH + ih) * params.inW + iw;\n"
"    gradInput[in_idx] = sum;\n"
"}\n"
"\n"
"kernel void max_pool2d_forward(\n"
"    device const float* input [[buffer(0)]],\n"
"    device float* output [[buffer(1)]],\n"
"    device int* argmax [[buffer(2)]],\n"
"    constant Pool2DParams& params [[buffer(3)]],\n"
"    uint3 gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid.x >= params.outW || gid.y >= params.outH || gid.z >= (params.batchSize * params.channels)) return;\n"
"    \n"
"    uint b = gid.z / params.channels;\n"
"    uint c = gid.z % params.channels;\n"
"    uint oh = gid.y;\n"
"    uint ow = gid.x;\n"
"    \n"
"    float maxVal = -1e38;\n"
"    int maxIdx = -1;\n"
"    \n"
"    for (uint kh = 0; kh < params.kSize; kh++) {\n"
"        int hi = (int)oh * (int)params.stride + (int)kh - (int)params.padding;\n"
"        if (hi >= 0 && hi < (int)params.inH) {\n"
"            for (uint kw = 0; kw < params.kSize; kw++) {\n"
"                int wi = (int)ow * (int)params.stride + (int)kw - (int)params.padding;\n"
"                if (wi >= 0 && wi < (int)params.inW) {\n"
"                    uint idx = ((b * params.channels + c) * params.inH + hi) * params.inW + wi;\n"
"                    float val = input[idx];\n"
"                    if (val > maxVal) {\n"
"                        maxVal = val;\n"
"                        maxIdx = (int)idx;\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    \n"
"    uint out_idx = ((b * params.channels + c) * params.outH + oh) * params.outW + ow;\n"
"    output[out_idx] = maxVal;\n"
"    argmax[out_idx] = maxIdx;\n"
"}\n"
"\n"
"kernel void max_pool2d_backward(\n"
"    device const float* gradOutput [[buffer(0)]],\n"
"    device float* gradInput [[buffer(1)]],\n"
"    device const int* argmax [[buffer(2)]],\n"
"    constant Pool2DParams& params [[buffer(3)]],\n"
"    uint3 gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid.x >= params.outW || gid.y >= params.outH || gid.z >= (params.batchSize * params.channels)) return;\n"
"    \n"
"    uint out_idx = ((gid.z) * params.outH + gid.y) * params.outW + gid.x;\n"
"    float grad = gradOutput[out_idx];\n"
"    int maxIdx = argmax[out_idx];\n"
"    \n"
"    if (maxIdx >= 0) {\n"
"        gradInput[maxIdx] += grad;\n"
"    }\n"
"}\n"
"\n"
"kernel void dropout_fwd(\n"
"    device const float* input [[buffer(0)]],\n"
"    device float* output [[buffer(1)]],\n"
"    device float* mask [[buffer(2)]],\n"
"    constant uint& length [[buffer(3)]],\n"
"    constant float& p [[buffer(4)]],\n"
"    constant bool& training [[buffer(5)]],\n"
"    constant uint64_t& seed [[buffer(6)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid >= length) return;\n"
"    if (!training) {\n"
"        output[gid] = input[gid];\n"
"        return;\n"
"    }\n"
"    uint64_t x = seed + gid + 1;\n"
"    x ^= x << 13; x ^= x >> 7; x ^= x << 17;\n"
"    float rf = (float)(x % 1000000) / 1000000.0f;\n"
"    float keepProb = 1.0f - p;\n"
"    if (rf < p) {\n"
"        mask[gid] = 0.0f; output[gid] = 0.0f;\n"
"    } else {\n"
"        mask[gid] = 1.0f; output[gid] = input[gid] / keepProb;\n"
"    }\n"
"}\n"
"\n"
"kernel void dropout_bwd(\n"
"    device const float* gradOut [[buffer(0)]],\n"
"    device float* gradIn [[buffer(1)]],\n"
"    device const float* mask [[buffer(2)]],\n"
"    constant uint& length [[buffer(3)]],\n"
"    constant float& p [[buffer(4)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid >= length) return;\n"
"    if (mask[gid] > 0.0f) {\n"
"        gradIn[gid] = gradOut[gid] / (1.0f - p);\n"
"    } else {\n"
"        gradIn[gid] = 0.0f;\n"
"    }\n"
"}\n"
"kernel void batchnorm2d_forward(\n"
"    device const float* input [[buffer(0)]],\n"
"    device float* output [[buffer(1)]],\n"
"    device float* runningMean [[buffer(2)]],\n"
"    device float* runningVar [[buffer(3)]],\n"
"    device float* savedMean [[buffer(4)]],\n"
"    device float* savedStd [[buffer(5)]],\n"
"    device const float* gamma [[buffer(6)]],\n"
"    device const float* beta [[buffer(7)]],\n"
"    constant BatchNorm2DParams& params [[buffer(8)]],\n"
"    uint3 gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid.x >= params.spatialSize || gid.z >= (params.batchSize * params.channels)) return;\n"
"    \n"
"    uint b = gid.z / params.channels;\n"
"    uint c = gid.z % params.channels;\n"
"    uint s = gid.x;\n"
"    uint idx = (b * params.channels + c) * params.spatialSize + s;\n"
"    \n"
"    float mean, std;\n"
"    if (params.training) {\n"
"        mean = savedMean[c];\n"
"        std = savedStd[c];\n"
"    } else {\n"
"        mean = runningMean[c];\n"
"        std = sqrt(runningVar[c] + params.eps);\n"
"    }\n"
"    \n"
"    float normalized = (input[idx] - mean) / std;\n"
"    if (params.affine) {\n"
"        output[idx] = gamma[c] * normalized + beta[c];\n"
"    } else {\n"
"        output[idx] = normalized;\n"
"    }\n"
"}\n"
"kernel void batchnorm2d_backward(\n"
"    device const float* gradOutput [[buffer(0)]],\n"
"    device const float* input [[buffer(1)]],\n"
"    device float* gradInput [[buffer(2)]],\n"
"    device const float* savedMean [[buffer(3)]],\n"
"    device const float* savedStd [[buffer(4)]],\n"
"    device const float* gamma [[buffer(5)]],\n"
"    device float* gradGamma [[buffer(6)]],\n"
"    device float* gradBeta [[buffer(7)]],\n"
"    constant BatchNorm2DParams& params [[buffer(8)]],\n"
"    uint3 gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid.x >= params.channels) return;\n"
"    uint c = gid.x;\n"
"    float sum_dy = 0.0f;\n"
"    float sum_dy_xhat = 0.0f;\n"
"    float mean = savedMean[c];\n"
"    float invStd = 1.0f / savedStd[c];\n"
"    \n"
"    for (uint b = 0; b < params.batchSize; b++) {\n"
"        for (uint s = 0; s < params.spatialSize; s++) {\n"
"            uint idx = (b * params.channels + c) * params.spatialSize + s;\n"
"            float dy = gradOutput[idx];\n"
"            float x_hat = (input[idx] - mean) * invStd;\n"
"            sum_dy += dy;\n"
"            sum_dy_xhat += dy * x_hat;\n"
"        }\n"
"    }\n"
"    \n"
"    if (params.affine) {\n"
"        gradGamma[c] = sum_dy_xhat;\n"
"        gradBeta[c] = sum_dy;\n"
"    }\n"
"    \n"
"    float g = params.affine ? gamma[c] : 1.0f;\n"
"    float m = (float)(params.batchSize * params.spatialSize);\n"
"    \n"
"    for (uint b = 0; b < params.batchSize; b++) {\n"
"        for (uint s = 0; s < params.spatialSize; s++) {\n"
"            uint idx = (b * params.channels + c) * params.spatialSize + s;\n"
"            float x_hat = (input[idx] - mean) * invStd;\n"
"            gradInput[idx] = (g * invStd / m) * (m * gradOutput[idx] - sum_dy - x_hat * sum_dy_xhat);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void batchnorm2d_stats(\n"
"    device const float* input [[buffer(0)]],\n"
"    device float* mean [[buffer(1)]],\n"
"    device float* var [[buffer(2)]],\n"
"    constant BatchNorm2DParams& params [[buffer(3)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tgs [[threads_per_threadgroup]],\n"
"    uint group_id [[threadgroup_position_in_grid]]\n"
") {\n"
"    uint c = group_id;\n"
"    if (c >= params.channels) return;\n"
"    float sum = 0.0f, sumSq = 0.0f;\n"
"    uint n = params.batchSize * params.spatialSize;\n"
"    for (uint i = tid; i < n; i += tgs) {\n"
"        uint b = i / params.spatialSize, s = i % params.spatialSize;\n"
"        uint idx = (b * params.channels + c) * params.spatialSize + s;\n"
"        float val = input[idx]; sum += val; sumSq += val * val;\n"
"    }\n"
"    sum = simd_sum(sum); sumSq = simd_sum(sumSq);\n"
"    threadgroup float sS[32], sSSq[32];\n"
"    if (tid % 32 == 0) { sS[tid/32] = sum; sSSq[tid/32] = sumSq; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        sum = (tid < (tgs/32)) ? sS[tid] : 0.0f;\n"
"        sumSq = (tid < (tgs/32)) ? sSSq[tid] : 0.0f;\n"
"        sum = simd_sum(sum); sumSq = simd_sum(sumSq);\n"
"        if (tid == 0) {\n"
"            float m = sum / (float)n;\n"
"            mean[c] = m; var[c] = (sumSq / (float)n) - (m * m);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void dense_grad_bias(\n"
"    device const float* gradOutput [[buffer(0)]],\n"
"    device float* gradBias [[buffer(1)]],\n"
"    constant uint& batchSize [[buffer(2)]],\n"
"    constant uint& outSize [[buffer(3)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid >= outSize) return;\n"
"    \n"
"    float sum = 0.0f;\n"
"    for (uint b = 0; b < batchSize; b++) {\n"
"        sum += gradOutput[b * outSize + gid];\n"
"    }\n"
"    gradBias[gid] = sum;\n"
"}\n"
"\n"
"kernel void layernorm_forward(\n"
"    device const float* input [[buffer(0)]],\n"
"    device float* output [[buffer(1)]],\n"
"    device const float* gamma [[buffer(2)]],\n"
"    device const float* beta [[buffer(3)]],\n"
"    constant LayerNormParams& params [[buffer(4)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tgs [[threads_per_threadgroup]],\n"
"    uint group_id [[threadgroup_position_in_grid]]\n"
") {\n"
"    uint b = group_id;\n"
"    if (b >= params.batchSize) return;\n"
"    uint offset = b * params.featureSize;\n"
"    float sum = 0.0f, sumSq = 0.0f;\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        float val = input[offset + i];\n"
"        sum += val; sumSq += val * val;\n"
"    }\n"
"    sum = simd_sum(sum); sumSq = simd_sum(sumSq);\n"
"    threadgroup float sS[32], sSSq[32];\n"
"    if (tid % 32 == 0) { sS[tid/32] = sum; sSSq[tid/32] = sumSq; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        sum = (tid < (tgs/32)) ? sS[tid] : 0.0f;\n"
"        sumSq = (tid < (tgs/32)) ? sSSq[tid] : 0.0f;\n"
"        sum = simd_sum(sum); sumSq = simd_sum(sumSq);\n"
"        if (tid == 0) {\n"
"            float mu = sum / (float)params.featureSize;\n"
"            float var = (sumSq / (float)params.featureSize) - (mu * mu);\n"
"            sS[0] = mu; sSSq[0] = 1.0f / sqrt(var + params.eps);\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float mu = sS[0], invStd = sSSq[0];\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        float x_hat = (input[offset + i] - mu) * invStd;\n"
"        output[offset + i] = (gamma && beta) ? (gamma[i] * x_hat + beta[i]) : x_hat;\n"
"    }\n"
"}\n"
"\n"
"kernel void layernorm_backward(\n"
"    device const float* gradOutput [[buffer(0)]],\n"
"    device const float* input [[buffer(1)]],\n"
"    device float* gradInput [[buffer(2)]],\n"
"    device const float* gamma [[buffer(3)]],\n"
"    constant LayerNormParams& params [[buffer(4)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tgs [[threads_per_threadgroup]],\n"
"    uint group_id [[threadgroup_position_in_grid]]\n"
") {\n"
"    uint b = group_id;\n"
"    if (b >= params.batchSize) return;\n"
"    uint offset = b * params.featureSize;\n"
"    float sum_x = 0.0f, sum_xSq = 0.0f;\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        float val = input[offset + i];\n"
"        sum_x += val; sum_xSq += val * val;\n"
"    }\n"
"    sum_x = simd_sum(sum_x); sum_xSq = simd_sum(sum_xSq);\n"
"    threadgroup float sS[32], sSSq[32];\n"
"    if (tid % 32 == 0) { sS[tid/32] = sum_x; sSSq[tid/32] = sum_xSq; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        float sx = (tid < (tgs/32)) ? sS[tid] : 0.0f;\n"
"        float sx2 = (tid < (tgs/32)) ? sSSq[tid] : 0.0f;\n"
"        sx = simd_sum(sx); sx2 = simd_sum(sx2);\n"
"        if (tid == 0) {\n"
"            float mu = sx / (float)params.featureSize;\n"
"            float var = (sx2 / (float)params.featureSize) - (mu * mu);\n"
"            sS[0] = mu; sSSq[0] = 1.0f / sqrt(var + params.eps);\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float mu = sS[0], invStd = sSSq[0];\n"
"    float sum_dy = 0.0f, sum_dy_xhat = 0.0f;\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        float x_hat = (input[offset + i] - mu) * invStd;\n"
"        float dy = gradOutput[offset + i];\n"
"        if (gamma) dy *= gamma[i];\n"
"        sum_dy += dy; sum_dy_xhat += dy * x_hat;\n"
"    }\n"
"    sum_dy = simd_sum(sum_dy); sum_dy_xhat = simd_sum(sum_dy_xhat);\n"
"    if (tid % 32 == 0) { sS[tid/32] = sum_dy; sSSq[tid/32] = sum_dy_xhat; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        float sdy = (tid < (tgs/32)) ? sS[tid] : 0.0f;\n"
"        float sdyx = (tid < (tgs/32)) ? sSSq[tid] : 0.0f;\n"
"        sdy = simd_sum(sdy); sdyx = simd_sum(sdyx);\n"
"        if (tid == 0) { sS[0] = sdy; sSSq[0] = sdyx; }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float sdy = sS[0], sdyx = sSSq[0];\n"
"    float m = (float)params.featureSize;\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        float x_hat = (input[offset + i] - mu) * invStd;\n"
"        float dy = gradOutput[offset + i];\n"
"        if (gamma) dy *= gamma[i];\n"
"        gradInput[offset + i] = (invStd / m) * (m * dy - sdy - x_hat * sdyx);\n"
"    }\n"
"}\n"
"\n"
"kernel void rmsnorm_forward(\n"
"    device const float* input [[buffer(0)]],\n"
"    device float* output [[buffer(1)]],\n"
"    device const float* gamma [[buffer(2)]],\n"
"    constant LayerNormParams& params [[buffer(3)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tgs [[threads_per_threadgroup]],\n"
"    uint group_id [[threadgroup_position_in_grid]]\n"
") {\n"
"    uint b = group_id;\n"
"    if (b >= params.batchSize) return;\n"
"    uint offset = b * params.featureSize;\n"
"    float sumSq = 0.0f;\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        float val = input[offset + i]; sumSq += val * val;\n"
"    }\n"
"    sumSq = simd_sum(sumSq);\n"
"    threadgroup float sS[32];\n"
"    if (tid % 32 == 0) sS[tid/32] = sumSq;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        float ss = (tid < (tgs/32)) ? sS[tid] : 0.0f;\n"
"        ss = simd_sum(ss);\n"
"        if (tid == 0) sS[0] = 1.0f / sqrt(ss / (float)params.featureSize + params.eps);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float invRms = sS[0];\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        output[offset + i] = gamma ? (input[offset + i] * invRms * gamma[i]) : (input[offset + i] * invRms);\n"
"    }\n"
"}\n"
"\n"
"kernel void rmsnorm_backward(\n"
"    device const float* gradOutput [[buffer(0)]],\n"
"    device const float* input [[buffer(1)]],\n"
"    device float* gradInput [[buffer(2)]],\n"
"    device const float* gamma [[buffer(3)]],\n"
"    constant LayerNormParams& params [[buffer(4)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tgs [[threads_per_threadgroup]],\n"
"    uint group_id [[threadgroup_position_in_grid]]\n"
") {\n"
"    uint b = group_id;\n"
"    if (b >= params.batchSize) return;\n"
"    uint offset = b * params.featureSize;\n"
"    float sumSq = 0.0f;\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        float val = input[offset + i]; sumSq += val * val;\n"
"    }\n"
"    sumSq = simd_sum(sumSq);\n"
"    threadgroup float sS[32], sSSq[32];\n"
"    if (tid % 32 == 0) sS[tid/32] = sumSq;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        float ss = (tid < (tgs/32)) ? sS[tid] : 0.0f;\n"
"        ss = simd_sum(ss);\n"
"        if (tid == 0) {\n"
"            float m = (float)params.featureSize;\n"
"            float rmsSq = ss / m + params.eps;\n"
"            sS[0] = rmsSq; sSSq[0] = 1.0f / sqrt(rmsSq);\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float rmsSq = sS[0], invRms = sSSq[0];\n"
"    float sum_dy_x = 0.0f;\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        float dy = gradOutput[offset + i];\n"
"        if (gamma) dy *= gamma[i];\n"
"        sum_dy_x += dy * input[offset + i];\n"
"    }\n"
"    sum_dy_x = simd_sum(sum_dy_x);\n"
"    if (tid % 32 == 0) sS[tid/32] = sum_dy_x;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        float sdyx = (tid < (tgs/32)) ? sS[tid] : 0.0f;\n"
"        sdyx = simd_sum(sdyx);\n"
"        if (tid == 0) sS[0] = sdyx;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float sdyx = sS[0], m = (float)params.featureSize;\n"
"    for (uint i = tid; i < params.featureSize; i += tgs) {\n"
"        float dy = gradOutput[offset + i];\n"
"        if (gamma) dy *= gamma[i];\n"
"        gradInput[offset + i] = invRms * (dy - (input[offset + i] * sdyx) / (m * rmsSq));\n"
"    }\n"
"}\n"
"\n"
"kernel void softmax_forward(\n"
"    device const float* input [[buffer(0)]],\n"
"    device float* output [[buffer(1)]],\n"
"    constant uint& batchSize [[buffer(2)]],\n"
"    constant uint& featureSize [[buffer(3)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tgs [[threads_per_threadgroup]],\n"
"    uint group_id [[threadgroup_position_in_grid]]\n"
") {\n"
"    uint b = group_id;\n"
"    if (b >= batchSize) return;\n"
"    uint offset = b * featureSize;\n"
"    float maxVal = -1e38;\n"
"    for (uint i = tid; i < featureSize; i += tgs) {\n"
"        maxVal = max(maxVal, input[offset + i]);\n"
"    }\n"
"    maxVal = simd_max(maxVal);\n"
"    threadgroup float sS[32];\n"
"    if (tid % 32 == 0) sS[tid/32] = maxVal;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        float mv = (tid < (tgs/32)) ? sS[tid] : -1e38;\n"
"        mv = simd_max(mv); if (tid == 0) sS[0] = mv;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    maxVal = sS[0];\n"
"    float sum = 0.0f;\n"
"    for (uint i = tid; i < featureSize; i += tgs) {\n"
"        float e = exp(input[offset + i] - maxVal);\n"
"        output[offset + i] = e; sum += e;\n"
"    }\n"
"    sum = simd_sum(sum);\n"
"    if (tid % 32 == 0) sS[tid/32] = sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        float s = (tid < (tgs/32)) ? sS[tid] : 0.0f;\n"
"        s = simd_sum(s); if (tid == 0) sS[0] = 1.0f / s;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float invSum = sS[0];\n"
"    for (uint i = tid; i < featureSize; i += tgs) {\n"
"        output[offset + i] *= invSum;\n"
"    }\n"
"}\n"
"\n"
"kernel void softmax_backward(\n"
"    device const float* gradOutput [[buffer(0)]],\n"
"    device const float* output [[buffer(1)]],\n"
"    device float* gradInput [[buffer(2)]],\n"
"    constant uint& batchSize [[buffer(3)]],\n"
"    constant uint& featureSize [[buffer(4)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tgs [[threads_per_threadgroup]],\n"
"    uint group_id [[threadgroup_position_in_grid]]\n"
") {\n"
"    uint b = group_id;\n"
"    if (b >= batchSize) return;\n"
"    uint offset = b * featureSize;\n"
"    float sum = 0.0f;\n"
"    for (uint i = tid; i < featureSize; i += tgs) {\n"
"        sum += gradOutput[offset + i] * output[offset + i];\n"
"    }\n"
"    sum = simd_sum(sum);\n"
"    threadgroup float sS[32];\n"
"    if (tid % 32 == 0) sS[tid/32] = sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid < 32) {\n"
"        float s = (tid < (tgs/32)) ? sS[tid] : 0.0f;\n"
"        s = simd_sum(s);\n"
"        if (tid == 0) sS[0] = s;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float s = sS[0];\n"
"    for (uint i = tid; i < featureSize; i += tgs) {\n"
"        gradInput[offset + i] = output[offset + i] * (gradOutput[offset + i] - s);\n"
"    }\n"
"}\n"
"";

void* initMetalDevice() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return NULL;

    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    ctx->device = device;
    ctx->commandQueue = [device newCommandQueue];

    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:metal_src options:nil error:&error];
    if (!library) {
        fprintf(stderr, "Metal library compilation failed: %s\n", [[error localizedDescription] UTF8String]);
        free(ctx);
        return NULL;
    }

    id<MTLComputePipelineState> (^loadPSO)(NSString*) = ^(NSString* name) {
        NSError* err = nil;
        id<MTLFunction> fn = [library newFunctionWithName:name];
        if (!fn) {
            fprintf(stderr, "Metal function not found: %s\n", [name UTF8String]);
            return (id<MTLComputePipelineState>)nil;
        }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            fprintf(stderr, "Failed to create pipeline state for %s: %s\n", [name UTF8String], [[err localizedDescription] UTF8String]);
        }
        return pso;
    };

    ctx->lstmPipelineState = loadPSO(@"lstm_step");
    ctx->biasActivationPipelineState = loadPSO(@"bias_activation");
    ctx->activationPipelineState = ctx->biasActivationPipelineState;

    ctx->conv2dForwardPSO = loadPSO(@"conv2d_forward");
    ctx->conv2dGradInPSO = loadPSO(@"conv2d_grad_input");
    ctx->conv2dGradWPSO = loadPSO(@"conv2d_grad_weights");
    ctx->conv2dGradBPSO = loadPSO(@"conv2d_grad_bias");

    ctx->avgPool2DForwardPSO = loadPSO(@"avg_pool2d_forward");
    ctx->avgPool2DBackwardPSO = loadPSO(@"avg_pool2d_backward");
    ctx->maxPool2DForwardPSO = loadPSO(@"max_pool2d_forward");
    ctx->maxPool2DBackwardPSO = loadPSO(@"max_pool2d_backward");

    ctx->dropoutForwardPSO = loadPSO(@"dropout_fwd");
    ctx->dropoutBackwardPSO = loadPSO(@"dropout_bwd");
    ctx->batchNorm2DForwardPSO = loadPSO(@"batchnorm2d_forward");
    ctx->batchNorm2DBackwardPSO = loadPSO(@"batchnorm2d_backward");
    ctx->batchNorm2DStatsPSO = loadPSO(@"batchnorm2d_stats");

    ctx->denseGradBPSO = loadPSO(@"dense_grad_bias");

    ctx->layerNormForwardPSO = loadPSO(@"layernorm_forward");
    ctx->layerNormBackwardPSO = loadPSO(@"layernorm_backward");
    ctx->rmsNormForwardPSO = loadPSO(@"rmsnorm_forward");
    ctx->rmsNormBackwardPSO = loadPSO(@"rmsnorm_backward");
    ctx->softmaxForwardPSO = loadPSO(@"softmax_forward");
    ctx->softmaxBackwardPSO = loadPSO(@"softmax_backward");

    return (void*)ctx;
}

bool isMetalAvailable(void* ptr) { return ptr != NULL; }
void freeMetalDevice(void* ptr) { if (ptr) { MetalContext* ctx = (MetalContext*)ptr; free(ctx); } }

void* createMetalBuffer(void* ptr, float* data, int length) {
    MetalContext* ctx = (MetalContext*)ptr;
    return (__bridge_retained void*)[ctx->device newBufferWithBytes:data length:length * sizeof(float) options:MTLResourceStorageModeShared];
}

void* createEmptyMetalBuffer(void* ptr, int length) {
    MetalContext* ctx = (MetalContext*)ptr;
    return (__bridge_retained void*)[ctx->device newBufferWithLength:length * sizeof(float) options:MTLResourceStorageModeShared];
}

void updateMetalBuffer(void* bufferPtr, float* data, int length) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferPtr;
    memcpy([buffer contents], data, length * sizeof(float));
}

void readMetalBuffer(void* bufferPtr, float* data, int length) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferPtr;
    memcpy(data, [buffer contents], length * sizeof(float));
}

void freeMetalBuffer(void* bufferPtr) {
    if (bufferPtr) {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)bufferPtr;
        buffer = nil;
    }
}

void waitForMetal(void* ptr) {
    MetalContext* ctx = (MetalContext*)ptr;
    id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
    [cb commit];
    [cb waitUntilCompleted];
}

void* createConv2DState(void* ptr, int inC, int outC, int kSize, int stride, int padding, float* weights, float* biases) {
    MetalContext* ctx = (MetalContext*)ptr;
    Conv2DLayerState* state = (Conv2DLayerState*)malloc(sizeof(Conv2DLayerState));
    state->ctx = ctx;
    state->inChannels = inC;
    state->outChannels = outC;
    state->kernelSize = kSize;
    state->stride = stride;
    state->padding = padding;
    state->weights = [ctx->device newBufferWithBytes:weights length:outC*inC*kSize*kSize*sizeof(float) options:MTLResourceStorageModeShared];
    state->bias = [ctx->device newBufferWithBytes:biases length:outC*sizeof(float) options:MTLResourceStorageModeShared];
    return state;
}

void conv2dForwardMPS(void* ptr, void* statePtr, void* bufIn, void* bufOut, int bS, int inH, int inW, int oH, int oW) {
    MetalContext* ctx = (MetalContext*)ptr;
    Conv2DLayerState* s = (Conv2DLayerState*)statePtr;
    @autoreleasepool {
        Conv2DParams p = { (uint)bS, (uint)s->inChannels, (uint)s->outChannels, (uint)inH, (uint)inW, (uint)oH, (uint)oW, (uint)s->kernelSize, (uint)s->stride, (uint)s->padding };
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->conv2dForwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bufIn offset:0 atIndex:0];
        [enc setBuffer:s->weights offset:0 atIndex:1];
        [enc setBuffer:s->bias offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)bufOut offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(oW, oH, bS*s->outChannels) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding]; [cb commit];
    }
}

void conv2dBackwardMPS(void* ptr, void* statePtr, void* bufIn, void* bufGOut, void* bufGIn, void* bufGW, void* bufGB, int bS, int iH, int iW, int oH, int oW) {
    MetalContext* ctx = (MetalContext*)ptr;
    Conv2DLayerState* s = (Conv2DLayerState*)statePtr;
    @autoreleasepool {
        Conv2DParams p = { (uint)bS, (uint)s->inChannels, (uint)s->outChannels, (uint)iH, (uint)iW, (uint)oH, (uint)oW, (uint)s->kernelSize, (uint)s->stride, (uint)s->padding };
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        if (bufGIn) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->conv2dGradInPSO];
            [enc setBuffer:(__bridge id<MTLBuffer>)bufGIn offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)bufGOut offset:0 atIndex:1];
            [enc setBuffer:s->weights offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(iW, iH, bS*s->inChannels) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
        }
        if (bufGW) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->conv2dGradWPSO];
            [enc setBuffer:(__bridge id<MTLBuffer>)bufGW offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)bufGOut offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)bufIn offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(s->kernelSize, s->kernelSize, s->outChannels*s->inChannels) threadsPerThreadgroup:MTLSizeMake(s->kernelSize, s->kernelSize, 1)];
            [enc endEncoding];
        }
        if (bufGB) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->conv2dGradBPSO];
            [enc setBuffer:(__bridge id<MTLBuffer>)bufGB offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)bufGOut offset:0 atIndex:1];
            [enc setBytes:&p length:sizeof(p) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(s->outChannels, 1, 1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [enc endEncoding];
        }
        [cb commit];
    }
}

void conv2dReloadWeightsMPS(void* state) {}
void freeConv2DState(void* state) { if(state) { Conv2DLayerState* s=(Conv2DLayerState*)state; s->weights=nil; s->bias=nil; free(s); } }

void avgPool2DPersistent(void* ptr, void* bI, void* bO, int bS, int c, int iH, int iW, int oH, int oW, int k, int s, int p) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        Pool2DParams params = {(uint)bS, (uint)c, (uint)iH, (uint)iW, (uint)oH, (uint)oW, (uint)k, (uint)s, (uint)p};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->avgPool2DForwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bO offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(oW, oH, bS*c) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding]; [cb commit];
    }
}

void avgPool2DBackwardPersistent(void* ptr, void* gO, void* gI, int bS, int c, int iH, int iW, int oH, int oW, int k, int s, int p) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        Pool2DParams params = {(uint)bS, (uint)c, (uint)iH, (uint)iW, (uint)oH, (uint)oW, (uint)k, (uint)s, (uint)p};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->avgPool2DBackwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)gO offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)gI offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(iW, iH, bS*c) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding]; [cb commit];
    }
}

void maxPool2DPersistent(void* ptr, void* bI, void* bO, void* bA, int bS, int c, int iH, int iW, int oH, int oW, int k, int s, int p) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        Pool2DParams params = {(uint)bS, (uint)c, (uint)iH, (uint)iW, (uint)oH, (uint)oW, (uint)k, (uint)s, (uint)p};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->maxPool2DForwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bO offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)bA offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(oW, oH, bS*c) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding]; [cb commit];
    }
}

void maxPool2DBackwardPersistent(void* ptr, void* gO, void* gI, void* bA, int bS, int c, int iH, int iW, int oH, int oW, int k, int s, int p) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        Pool2DParams params = {(uint)bS, (uint)c, (uint)iH, (uint)iW, (uint)oH, (uint)oW, (uint)k, (uint)s, (uint)p};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->maxPool2DBackwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)gO offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)gI offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)bA offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(oW, oH, bS*c) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding]; [cb commit];
    }
}

void dropoutPersistent(void* ptr, void* bI, void* bO, void* bM, int len, float p, bool tr, unsigned long long s) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->dropoutForwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bO offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)bM offset:0 atIndex:2];
        uint uL = (uint)len; [enc setBytes:&uL length:4 atIndex:3];
        [enc setBytes:&p length:4 atIndex:4]; [enc setBytes:&tr length:1 atIndex:5];
        [enc setBytes:&s length:8 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(len, 1, 1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc endEncoding]; [cb commit];
    }
}

void dropoutBackwardPersistent(void* ptr, void* gO, void* gI, void* bM, int len, float p) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->dropoutBackwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)gO offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)gI offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)bM offset:0 atIndex:2];
        uint uL = (uint)len; [enc setBytes:&uL length:4 atIndex:3];
        [enc setBytes:&p length:4 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(len, 1, 1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc endEncoding]; [cb commit];
    }
}

void batchNorm2DForwardPersistent(void* ptr, void* bI, void* bO, void* rM, void* rV, void* sM, void* sS, void* g, void* b, int bS, int c, int sp, float eps, float mom, bool tr, bool aff) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        BatchNorm2DParams params = {(uint)bS, (uint)c, (uint)sp, eps, mom, tr, aff};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->batchNorm2DForwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bO offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)rM offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)rV offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)sM offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)sS offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)g offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)b offset:0 atIndex:7];
        [enc setBytes:&params length:sizeof(params) atIndex:8];
        [enc dispatchThreads:MTLSizeMake(sp, 1, bS*c) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void batchNorm2DBackwardPersistent(void* ptr, void* gO, void* bI, void* gI, void* sM, void* sS, void* g, void* gG, void* gB, int bS, int c, int sp, float eps, float mom, bool aff) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        BatchNorm2DParams params = {(uint)bS, (uint)c, (uint)sp, eps, mom, true, aff};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->batchNorm2DBackwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)gO offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)gI offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)sM offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)sS offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)g offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)gG offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)gB offset:0 atIndex:7];
        [enc setBytes:&params length:sizeof(params) atIndex:8];
        [enc dispatchThreads:MTLSizeMake(c, 1, 1) threadsPerThreadgroup:MTLSizeMake(c > 32 ? 32 : c, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void matMulMPSPersistent(void* ptr, void* bA, void* bB, void* bC, int M, int N, int K, float beta, bool transA, bool transB) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:transA ? K : M columns:transA ? M : K rowBytes:(transA ? M : K)*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:transB ? N : K columns:transB ? K : N rowBytes:(transB ? K : N)*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*4 dataType:MPSDataTypeFloat32];
        MPSMatrixMultiplication *k = [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device transposeLeft:transA transposeRight:transB resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:beta];
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        [k encodeToCommandBuffer:cb leftMatrix:[[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bA descriptor:descA]
                      rightMatrix:[[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bB descriptor:descB]
                     resultMatrix:[[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)bC descriptor:descC]];
        [cb commit];
    }
}

void matMulFusedPersistent(void* ptr, void* bA, void* bB, void* bBias, void* bC, int M, int N, int K, int act) {
    matMulMPSPersistent(ptr, bA, bB, bC, M, N, K, 0.0f, false, false);
    if (bBias || act != ActivationNone) {
        MetalContext* ctx = (MetalContext*)ptr;
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->biasActivationPipelineState];
        [enc setBuffer:(__bridge id<MTLBuffer>)bC offset:0 atIndex:0];
        [enc setBuffer:bBias ? (__bridge id<MTLBuffer>)bBias : (__bridge id<MTLBuffer>)bC offset:0 atIndex:1];
        uint m=M, n=N; bool hb=bBias!=nil;
        [enc setBytes:&m length:4 atIndex:2]; [enc setBytes:&n length:4 atIndex:3]; [enc setBytes:&act length:4 atIndex:4]; [enc setBytes:&hb length:1 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(M*N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void lstmStepFusedPersistent(void* ptr, void* bPA, void* bCP, void* bCN, void* bHN, void* bI, void* bF, void* bG, void* bO, int oS) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->lstmPipelineState];
        [enc setBuffer:(__bridge id<MTLBuffer>)bPA offset:0 atIndex:0]; [enc setBuffer:(__bridge id<MTLBuffer>)bCP offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)bCN offset:0 atIndex:2]; [enc setBuffer:(__bridge id<MTLBuffer>)bHN offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:4]; [enc setBuffer:(__bridge id<MTLBuffer>)bF offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)bG offset:0 atIndex:6]; [enc setBuffer:(__bridge id<MTLBuffer>)bO offset:0 atIndex:7];
        uint uoS = (uint)oS; [enc setBytes:&uoS length:4 atIndex:8];
        [enc dispatchThreads:MTLSizeMake(oS, 1, 1) threadsPerThreadgroup:MTLSizeMake(oS > 256 ? 256 : oS, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void matMulMPS(void* d, float* A, float* B, float* C, int M, int N, int K) {
    void *bA = createMetalBuffer(d, A, M*K), *bB = createMetalBuffer(d, B, K*N), *bC = createEmptyMetalBuffer(d, M*N);
    matMulMPSPersistent(d, bA, bB, bC, M, N, K, 0.0f, false, false); waitForMetal(d); readMetalBuffer(bC, C, M*N);
    freeMetalBuffer(bA); freeMetalBuffer(bB); freeMetalBuffer(bC);
}

void matMulFused(void* d, float* A, float* B, float* Bias, float* C, int M, int N, int K, int act) {
    void *bA = createMetalBuffer(d, A, M*K), *bB = createMetalBuffer(d, B, K*N), *bBias = Bias ? createMetalBuffer(d, Bias, N) : NULL, *bC = createEmptyMetalBuffer(d, M*N);
    matMulFusedPersistent(d, bA, bB, bBias, bC, M, N, K, act); waitForMetal(d); readMetalBuffer(bC, C, M*N);
    freeMetalBuffer(bA); freeMetalBuffer(bB); if(bBias) freeMetalBuffer(bBias); freeMetalBuffer(bC);
}

void applyActivationMPS(void* d, float* data, int len, int act) {
    void *buf = createMetalBuffer(d, data, len); MetalContext* ctx = (MetalContext*)d;
    id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->biasActivationPipelineState]; [enc setBuffer:(__bridge id<MTLBuffer>)buf offset:0 atIndex:0]; [enc setBuffer:(__bridge id<MTLBuffer>)buf offset:0 atIndex:1];
    uint m=1, n=len; bool hb=false; [enc setBytes:&m length:4 atIndex:2]; [enc setBytes:&n length:4 atIndex:3]; [enc setBytes:&act length:4 atIndex:4]; [enc setBytes:&hb length:1 atIndex:5];
    [enc dispatchThreads:MTLSizeMake(len, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    readMetalBuffer(buf, data, len); freeMetalBuffer(buf);
}

void lstmStepFused(void* d, float* pA, float* cP, float* cN, float* hN, float* i, float* f, float* g, float* o, int oS) {
    void *bPA = createMetalBuffer(d, pA, 4*oS), *bCP = createMetalBuffer(d, cP, oS), *bCN = createEmptyMetalBuffer(d, oS), *bHN = createEmptyMetalBuffer(d, oS), *bI = createEmptyMetalBuffer(d, oS), *bF = createEmptyMetalBuffer(d, oS), *bG = createEmptyMetalBuffer(d, oS), *bO = createEmptyMetalBuffer(d, oS);
    lstmStepFusedPersistent(d, bPA, bCP, bCN, bHN, bI, bF, bG, bO, oS); waitForMetal(d);
    readMetalBuffer(bCN, cN, oS); readMetalBuffer(bHN, hN, oS); readMetalBuffer(bI, i, oS); readMetalBuffer(bF, f, oS); readMetalBuffer(bG, g, oS); readMetalBuffer(bO, o, oS);
    freeMetalBuffer(bPA); freeMetalBuffer(bCP); freeMetalBuffer(bCN); freeMetalBuffer(bHN); freeMetalBuffer(bI); freeMetalBuffer(bF); freeMetalBuffer(bG); freeMetalBuffer(bO);
}

void batchNorm2DStatsPersistent(void* ptr, void* bI, void* m, void* v, int bS, int c, int sp, float eps) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        BatchNorm2DParams params = {(uint)bS, (uint)c, (uint)sp, eps, 0.0f, true, false};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->batchNorm2DStatsPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)m offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)v offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(c, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void denseBackwardGradWPersistent(void* ptr, void* bDZ, void* bIn, void* bGW, int bS, int iS, int oS) {
    // gradW = dz^T * input
    // dz: [batch, outSize] -> dz^T: [outSize, batch]
    // input: [batch, inSize]
    // result: [outSize, inSize]
    matMulMPSPersistent(ptr, bDZ, bIn, bGW, oS, iS, bS, 0.0f, true, false);
}

void denseBackwardGradBPersistent(void* ptr, void* bDZ, void* bGB, int bS, int oS) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->denseGradBPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bDZ offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bGB offset:0 atIndex:1];
        uint ubS = (uint)bS, uoS = (uint)oS;
        [enc setBytes:&ubS length:4 atIndex:2];
        [enc setBytes:&uoS length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(oS, 1, 1) threadsPerThreadgroup:MTLSizeMake(oS > 256 ? 256 : oS, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void layerNormForwardPersistent(void* ptr, void* bI, void* bO, void* g, void* b, int bS, int fS, float eps) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        LayerNormParams params = {(uint)bS, (uint)fS, eps};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->layerNormForwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bO offset:0 atIndex:1];
        [enc setBuffer:g ? (__bridge id<MTLBuffer>)g : nil offset:0 atIndex:2];
        [enc setBuffer:b ? (__bridge id<MTLBuffer>)b : nil offset:0 atIndex:3];
        [enc setBytes:&params length:sizeof(params) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(bS, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void layerNormBackwardPersistent(void* ptr, void* gO, void* bI, void* gI, void* g, int bS, int fS, float eps) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        LayerNormParams params = {(uint)bS, (uint)fS, eps};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->layerNormBackwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)gO offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)gI offset:0 atIndex:2];
        [enc setBuffer:g ? (__bridge id<MTLBuffer>)g : nil offset:0 atIndex:3];
        [enc setBytes:&params length:sizeof(params) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(bS, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void rmsNormForwardPersistent(void* ptr, void* bI, void* bO, void* g, int bS, int fS, float eps) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        LayerNormParams params = {(uint)bS, (uint)fS, eps};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->rmsNormForwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bO offset:0 atIndex:1];
        [enc setBuffer:g ? (__bridge id<MTLBuffer>)g : nil offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(bS, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void rmsNormBackwardPersistent(void* ptr, void* gO, void* bI, void* gI, void* g, int bS, int fS, float eps) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        LayerNormParams params = {(uint)bS, (uint)fS, eps};
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->rmsNormBackwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)gO offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)gI offset:0 atIndex:2];
        [enc setBuffer:g ? (__bridge id<MTLBuffer>)g : nil offset:0 atIndex:3];
        [enc setBytes:&params length:sizeof(params) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(bS, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void softmaxForwardPersistent(void* ptr, void* bI, void* bO, int bS, int fS) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->softmaxForwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)bI offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bO offset:0 atIndex:1];
        uint ubS = (uint)bS, ufS = (uint)fS;
        [enc setBytes:&ubS length:4 atIndex:2];
        [enc setBytes:&ufS length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(bS, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}

void softmaxBackwardPersistent(void* ptr, void* gO, void* bO, void* gI, int bS, int fS) {
    MetalContext* ctx = (MetalContext*)ptr;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->softmaxBackwardPSO];
        [enc setBuffer:(__bridge id<MTLBuffer>)gO offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)bO offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)gI offset:0 atIndex:2];
        uint ubS = (uint)bS, ufS = (uint)fS;
        [enc setBytes:&ubS length:4 atIndex:3];
        [enc setBytes:&ufS length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(bS, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit];
    }
}
