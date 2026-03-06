#ifndef METAL_H
#define METAL_H

#include <stdbool.h>
#include <stddef.h>

// Opaque pointer to Metal context
typedef void* MetalContext;
typedef void* MetalBuffer;

// Device & Availability
bool IsMetalAvailable();
MetalContext InitMetal();
void FreeMetal(MetalContext ctx);

// Buffer Management
MetalBuffer CreateSharedBuffer(MetalContext ctx, size_t size);
void FreeBuffer(MetalBuffer buf);
void* GetBufferContents(MetalBuffer buf);

// MPS Operations
void MPSMatMul(MetalContext ctx,
               MetalBuffer A, MetalBuffer B, MetalBuffer C,
               int M, int N, int K);

// LSTM/GRU Operations
// LSTM forward pass for one timestep
// gateBuf: buffer for all gates (input, forget, cell, output) - size: batchSize * hiddenSize * 4
// cPrev: previous cell state - size: batchSize * hiddenSize
// hPrev: previous hidden state - size: batchSize * hiddenSize
// cOut: output cell state - size: batchSize * hiddenSize
// hOut: output hidden state - size: batchSize * hiddenSize
void LSTMForward(MetalContext ctx,
                 MetalBuffer gateBuf,    // Pre-activations for all gates
                 MetalBuffer cPrev,      // Previous cell state
                 MetalBuffer hPrev,      // Previous hidden state
                 MetalBuffer cOut,       // Output cell state
                 MetalBuffer hOut,       // Output hidden state
                 int batchSize,
                 int hiddenSize);

// GRU forward pass for one timestep
// gateBuf: buffer for reset, update, candidate gates - size: batchSize * hiddenSize * 3
// hPrev: previous hidden state - size: batchSize * hiddenSize
// hOut: output hidden state - size: batchSize * hiddenSize
void GRUForward(MetalContext ctx,
                MetalBuffer gateBuf,    // Pre-activations for all gates
                MetalBuffer hPrev,      // Previous hidden state
                MetalBuffer hOut,       // Output hidden state
                int batchSize,
                int hiddenSize);

#endif
