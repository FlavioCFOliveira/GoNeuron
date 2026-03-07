# GoNeuron Performance Architecture - Agent Memory

## Overview

Este documento contém padrões de performance e otimizações identificados durante a análise da biblioteca GoNeuron. Use como referência para futuras otimizações.

---

## Padrões de Otimização Implementados

### 1. Arena-Based State Saving

**Localização:** `internal/layer/layer.go` - `ForwardWithArena()`

```go
func (d *Dense) ForwardWithArena(x []float32, arena *[]float32, offset *int) []float32 {
    // Salva input no arena em vez de alocar novo slice
    if arena != nil && offset != nil {
        d.arenaPtr = arena
        // ... resize logic ...
        copy((*arena)[*offset:], x)
        *offset += len(x)
    }
}
```

**Uso:** Usado internamente pelo Network.Forward(), mas não exposto na API pública.

---

### 2. LightweightClone para Worker Pools

**Localização:** `internal/net/net.go` - `trainBatchParallel()`

```go
func (n *Network) initWorkers(num int) {
    for i := len(oldPool); i < num; i++ {
        n.workerPool[i] = n.LightweightClone()
        workerGrads := make([]float32, len(n.grads))
        n.workerPool[i].grads = workerGrads
    }
}
```

**Propósito:** Evita `Clone()` caro durante `TrainBatch` - compartilha params/grads mas com buffers privados.

---

### 3. Contiguous Parameter Buffers

**Localização:** `internal/net/net.go` - `initBuffers()`

```go
func (n *Network) initBuffers() {
    totalSize := 0
    for _, l := range n.layers {
        totalSize += len(l.Params())
    }
    n.params = make([]float32, totalSize)
    n.grads = make([]float32, totalSize)
    // Rebinda camadas para usar views destes buffers
}
```

**Benefício:** Melhor localidade de cache, menos alocações, compatível com GPU.

---

### 4. Pre-allocated Buffers

**Padrão em todas as camadas:**
```go
type Dense struct {
    inputBuf  []float32  // Pré-alocado
    outputBuf []float32  // Pré-alocado
    preActBuf []float32  // Pré-alocado
    gradInBuf []float32  // Pré-alocado
    dzBuf     []float32  // Pré-alocado
}
```

**Exceção:** LSTM/GRU têm buffers dinâmicos devido a sequências variáveis.

---

### 5. GPU/Metal Integration

**Localização:** `internal/layer/metal.go` e camadas individuais

```go
if metal, ok := d.device.(*MetalDevice); ok && metal.IsAvailable() {
    d.bufWeights = metal.CreateBuffer(d.weights)
    metal.MatMulFusedPersistent(...)  // Kernel fused
}
```

**Otimizações Metal:**
- Kernel fusion (MatMul + Bias + Activation)
- Persistent buffers (minimiza CPU-GPU transfers)
- Redução de workers para GPU (max 4 para evitar overhead CGO)

---

## Anti-Padrões Identificados nos Exemplos

### 1. Loop de Inferência Sequencial

**Ruim:**
```go
for i := 0; i < len(xTest); i++ {
    pred := model.Predict(xTest[i])  // Aloca a cada iteração
}
```

**Bom:**
```go
results := model.PredictBatch(xTest)  // Usa ForwardBatch internamente
```

**Afeta:** mnist, cifar10, sequential_mnist, sentiment

---

### 2. Alocação em Loop Quente

**Ruim:**
```go
for i := 0; i < numSamples; i++ {
    x[i] = make([]float32, inSize)  // Alocação por sample
}
```

**Bom:**
```go
buffer := make([]float32, numSamples*inSize)  // Uma alocação
x = make([][]float32, numSamples)
for i := 0; i < numSamples; i++ {
    x[i] = buffer[i*inSize:(i+1)*inSize]  // Apenas slice
}
```

---

### 3. Não Usar TrainBatch

**Ruim:**
```go
for i := 0; i < len(inputs); i++ {
    model.Train(inputs[i], targets[i])  // Sample-a-sample
}
```

**Bom:**
```go
model.Fit(inputs, targets, epochs, batchSize)  // Com paralelização automática
// Ou:
model.TrainBatch(batchX, batchY)  // Batching manual
```

**Nota:** `trainBatchParallel` é usado automaticamente para batches > 64 samples.

---

## Checklist de Otimização

Ao criar novos exemplos ou otimizar existentes:

- [ ] Usar `PredictBatch` em vez de loop de `Predict`
- [ ] Usar `Fit()` ou `TrainBatch()` em vez de loop de `Train()`
- [ ] Pré-alocar buffers de dados quando possível
- [ ] Considerar `SetDevice()` para GPU quando disponível
- [ ] Usar callbacks para logging/checkpointing em vez de loops manuais
- [ ] Evitar alocações em loops quentes

---

## Arquivos-Chave para Performance

| Arquivo | Responsabilidade |
|---------|------------------|
| `internal/layer/layer.go` | Dense com arena, LightweightClone |
| `internal/net/net.go` | Network, TrainBatch, trainBatchParallel |
| `internal/layer/metal.go` | GPU kernels, buffer management |
| `internal/layer/matmul_optimized.go` | Otimizações de GEMM |

---

## Referência Rápida: API de Alto Nível vs Baixo Nível

### Alto Nível (Recomendado)
```go
model := goneuron.NewSequential(...)
model.Compile(optimizer, loss)
model.Fit(x, y, epochs, batchSize, callbacks...)
predictions := model.PredictBatch(xTest)
```

### Baixo Nível (Quando necessário)
```go
layers := []layer.Layer{...}
network := net.New(layers, loss, optimizer)
network.TrainBatch(x, y)
```

---

*Última atualização: 2026-03-06*
*Por: golang-performance-architect*
