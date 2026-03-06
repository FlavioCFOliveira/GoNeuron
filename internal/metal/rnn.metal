// LSTM/GRU Metal Shaders for GoNeuron
// These kernels implement efficient LSTM and GRU forward passes
#include <metal_stdlib>
using namespace metal;

// Sigmoid activation function
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// LSTM Forward Kernel
// Each thread computes one hidden unit for one sample in the batch
kernel void lstm_forward(
    device const float* gateBuf [[buffer(0)]],   // Pre-activations: [batch, hidden*4]
    device const float* cPrev [[buffer(1)]],     // Previous cell state: [batch, hidden]
    device const float* hPrev [[buffer(2)]],     // Previous hidden state: [batch, hidden]
    device float* cOut [[buffer(3)]],            // Output cell state: [batch, hidden]
    device float* hOut [[buffer(4)]],            // Output hidden state: [batch, hidden]
    constant int& batchSize [[buffer(5)]],
    constant int& hiddenSize [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int batchIdx = gid.x;
    int hiddenIdx = gid.y;

    if (batchIdx >= batchSize || hiddenIdx >= hiddenSize) {
        return;
    }

    int baseIdx = batchIdx * hiddenSize * 4 + hiddenIdx;
    int stateIdx = batchIdx * hiddenSize + hiddenIdx;

    // Load gate pre-activations
    float i_pre = gateBuf[baseIdx];                    // Input gate
    float f_pre = gateBuf[baseIdx + hiddenSize];       // Forget gate
    float g_pre = gateBuf[baseIdx + 2 * hiddenSize];   // Cell candidate
    float o_pre = gateBuf[baseIdx + 3 * hiddenSize];   // Output gate

    // Apply activations
    float i = sigmoid(i_pre);
    float f = sigmoid(f_pre);
    float g = tanh(g_pre);
    float o = sigmoid(o_pre);

    // Load previous cell state
    float c_t_1 = cPrev[stateIdx];

    // Compute new cell state: c_t = f * c_{t-1} + i * g
    float c_t = f * c_t_1 + i * g;

    // Compute new hidden state: h_t = o * tanh(c_t)
    float h_t = o * tanh(c_t);

    // Store results
    cOut[stateIdx] = c_t;
    hOut[stateIdx] = h_t;
}

// GRU Forward Kernel
// Each thread computes one hidden unit for one sample in the batch
kernel void gru_forward(
    device const float* gateBuf [[buffer(0)]],   // Pre-activations: [batch, hidden*3]
    device const float* hPrev [[buffer(1)]],    // Previous hidden state: [batch, hidden]
    device float* hOut [[buffer(2)]],           // Output hidden state: [batch, hidden]
    constant int& batchSize [[buffer(3)]],
    constant int& hiddenSize [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int batchIdx = gid.x;
    int hiddenIdx = gid.y;

    if (batchIdx >= batchSize || hiddenIdx >= hiddenSize) {
        return;
    }

    int baseIdx = batchIdx * hiddenSize * 3 + hiddenIdx;
    int stateIdx = batchIdx * hiddenSize + hiddenIdx;

    // Load gate pre-activations
    float r_pre = gateBuf[baseIdx];                    // Reset gate
    float z_pre = gateBuf[baseIdx + hiddenSize];       // Update gate
    float n_pre = gateBuf[baseIdx + 2 * hiddenSize];     // Candidate activation

    // Load previous hidden state
    float h_t_1 = hPrev[stateIdx];

    // Apply activations
    float r = sigmoid(r_pre);
    float z = sigmoid(z_pre);
    float n = tanh(n_pre);

    // Compute new hidden state: h_t = (1 - z) * n + z * h_{t-1}
    float h_t = (1.0f - z) * n + z * h_t_1;

    // Store result
    hOut[stateIdx] = h_t;
}
