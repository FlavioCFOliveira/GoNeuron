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

#endif
