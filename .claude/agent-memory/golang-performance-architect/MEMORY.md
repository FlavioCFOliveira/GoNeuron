# GoNeuron Performance Architecture - Agent Memory

## Overview

Padrões de performance e anti-patterns identificados no GoNeuron. Ver AUDIT/performance_audit.md para relatório completo.

---

## Padrões Implementados (Bons)

### Arena-Based State Saving (PERF-005 - IMPLEMENTADO)
- **Localização:** `internal/layer/layer.go:ForwardWithArena()`, `internal/net/net.go:CalculateTotalArenaSize()`
- Usa `arena *[]float32` e `offset *int` para salvar estados sem alocações
- **PERF-005:** Pre-cálculo de capacidade elimina redimensionamentos dinâmicos
- **Implementação:**
  ```go
  // Network pre-calcula tamanho total necessário
  func (n *Network) CalculateTotalArenaSize() int {
      total := 0
      for _, l := range n.layers {
          total += l.ArenaSize()
      }
      return total
  }

  // initBuffers() pre-aloca com capacidade exata
  totalArenaSize := n.CalculateTotalArenaSize()
  if totalArenaSize > 0 {
      n.arena = make([]float32, 0, totalArenaSize)
  }

  // Forward() reseta apenas o length, mantém capacity
  n.arena = n.arena[:0]
  ```
- **Benefícios:** Zero alocações durante forward pass, GC pressure eliminada
- **Thread-safety:** Cada worker tem sua própria arena (via LightweightClone)

### LightweightClone para Workers
- **Localização:** `internal/net/net.go:initWorkers()`
- Compartilha params/grads, mantém buffers privados
- Cada camada implementa `LightweightClone(params, grads []float32) Layer`

### MatMul Otimizado
- **Localização:** `internal/layer/matmul_optimized.go`
- Três níveis: simple < 1024, unrolled < 65536, parallel_blocked > 65536
- Block size 64 para L1 cache
- Unrolling 4x para ILP

### Loss Function Buffer Reuse (PERF-040 - IMPLEMENTADO)
- **Localização:** `internal/loss/loss.go` (todas as loss functions)
- **Problema:** `make([]float32, n)` em cada chamada de `Backward()`
- **Solução:** Buffer `gradBuf []float32` interno em cada struct de loss
- **Pattern:**
```go
type MSE struct {
    gradBuf []float32  // Buffer reusável
}

func (m *MSE) Backward(yPred, yTrue []float32) ([]float32, error) {
    n := len(yPred)
    // Reusa ou cresce buffer conforme necessário
    if cap(m.gradBuf) < n {
        m.gradBuf = make([]float32, n)
    }
    grad := m.gradBuf[:n]
    // ... computação do gradiente ...
    return grad, nil
}

// ResetGradBuffer permite liberar memória quando necessário
func (m *MSE) ResetGradBuffer() {
    m.gradBuf = nil
}
```
- **Thread-safety:**
  - `BackwardInPlace` com buffer externo é thread-safe
  - `Backward` com buffer interno requer instância própria por goroutine
  - Workers devem usar `BackwardInPlace` ou ter sua própria loss instance
- **Resultados:**
  - 0 allocs/op em todos os benchmarks backward
  - Redução de ~25% no tempo de execução
- **Losses implementadas:** MSE, CrossEntropy, Huber, L1Loss, BCELoss, BCEWithLogitsLoss, NLLLoss, CosineEmbeddingLoss, HingeEmbeddingLoss, TripletMarginLoss, MarginRankingLoss, KLDivLoss, MultiMarginLoss, MultiLabelSoftMarginLoss

### Conv2D Metal Backward Zero-Allocation (PERF-020)
- **Localização:** `internal/layer/conv2d.go:693-712`
- **Problema:** Alocação de `tempGradW` e `tempGradB` em cada backward pass
- **Solução:** Reusar `scratchBuf` pre-alocado + proteção com `arenaMu`
- **Pattern:**
```go
// Thread-safe buffer reuse for multi-worker training
c.arenaMu.Lock()
if cap(c.scratchBuf) < needSize {
    c.scratchBuf = make([]float32, needSize)
}
scratch := c.scratchBuf[:needSize]
c.arenaMu.Unlock()
// Usar scratch para leitura GPU e acumulação
```
- **Benefícios:** Zero alocações em hot path, thread-safe para worker pools

### BatchNorm2D istdBuf Pre-alocado (PERF-008 - IMPLEMENTADO)
- **Localização:** `internal/layer/batchnorm2d.go:ForwardWithArena()`
- **Problema:** `make([]float32, b.numFeatures)` a cada forward pass
- **Solução:** Buffer `istdBuf` pre-alocado no struct, reutilizado em cada chamada
- **Implementação:**
  ```go
  // No struct
  istdBuf []float32

  // Build() - pre-alocação
  b.istdBuf = make([]float32, numFeatures)

  // LightweightClone() - cada worker tem seu próprio buffer
  istdBuf: make([]float32, numFeatures)

  // ForwardWithArena() - reuso com capacity check
  if len(b.istdBuf) < b.numFeatures {
      b.istdBuf = make([]float32, b.numFeatures)
  }
  istdBuf := b.istdBuf[:b.numFeatures]
  for f := 0; f < b.numFeatures; f++ {
      istdBuf[f] = 1.0 / b.savedStd[f]
  }
  ```
- **Thread-safety:** Cada worker tem seu próprio buffer (via LightweightClone)
- **Benchmark:** 0 allocs/op em CPU forward após primeira chamada
- **Status:** Implementado em 2026-03-09

### Buffer Reuse em Loss Functions (PERF-040 - IMPLEMENTADO)
- **Localização:** `internal/loss/loss.go` - todas as loss functions
- **Problema:** `Backward()` alocava `[]float32` a cada chamada via `make([]float32, n)`
- **Solução:** Buffer `gradBuf []float32` pre-alocado no struct, cresce conforme necessário
- **Pattern:**
  ```go
  // Struct com buffer interno
  type MSE struct {
      gradBuf []float32
  }

  // Backward reusa buffer interno
  func (m *MSE) Backward(yPred, yTrue []float32) ([]float32, error) {
      n := len(yPred)
      if cap(m.gradBuf) < n {
          m.gradBuf = make([]float32, n)
      }
      grad := m.gradBuf[:n]
      // ... computa gradiente em grad ...
      return grad, nil
  }

  // Reset para liberar memória quando necessário
  func (m *MSE) ResetGradBuffer() {
      m.gradBuf = nil
  }
  ```
- **Loss functions modificadas:** MSE, CrossEntropy, Huber, L1Loss, BCELoss, BCEWithLogitsLoss, NLLLoss, CosineEmbeddingLoss, HingeEmbeddingLoss, TripletMarginLoss, MarginRankingLoss, KLDivLoss, MultiMarginLoss, MultiLabelSoftMarginLoss
- **Thread-safety:** `BackwardInPlace` continua thread-safe com buffer externo; cada goroutine deve ter sua própria instância para `Backward()`
- **Benchmark:** `0 B/op, 0 allocs/op` após primeira chamada
- **API:** Mantida compatibilidade - structs agora usam ponteiros (`&MSE{}` em vez de `MSE{}`)
- **Status:** Implementado em 2026-03-09

---

## Anti-Patterns Críticos Encontrados

### 1. Predict Individual em Loop (EX-001)
**Problema:** 37 exemplos usam `model.Predict(x[i])` em vez de `PredictBatch()`
**Impacto:** Alocação por iteração, 8x mais lento
**Arquivos:** mnist/main.go, cifar10/main.go, sequential_mnist/main.go, sentiment/main.go

**Correção:**
```go
// Ruim
for i := 0; i < len(xTest); i++ {
    pred := model.Predict(xTest[i])
}

// Bom
predictions, _ := model.PredictBatch(xTest)
```

### 2. Alocação em Loop de Data Loading (EX-002)
**Problema:** Alocação por sample em vez de buffer contíguo
**Arquivos:** Todos os exemplos de datasets

**Correção:**
```go
buffer := make([]float32, numSamples*inSize)
x = make([][]float32, numSamples)
for i := 0; i < numSamples; i++ {
    x[i] = buffer[i*inSize:(i+1)*inSize]
}
```

### 3. Alocação em Hot Path (LAYER-001)
**Problema:** `make([]float32, outSize)` no backward pass
**Localização:** `internal/layer/layer.go:399-406` (softmax backward)
**Correção:** Usar buffer pre-alocado ou stack para tamanhos pequenos

### 4. Conv2D Metal Buffers Realocados (LAYER-002)
**Problema:** Buffers GPU criados dinamicamente em cada forward
**Localização:** `internal/layer/conv2d.go:410-424`
**Correção:** Pre-alocar para tamanho máximo ou usar pool

### 5. BatchNorm istdBuf Alocado em Cada Forward (LAYER-003) ✓ RESOLVIDO
**Problema:** `make([]float32, b.numFeatures)` em ForwardWithArena
**Localização:** `internal/layer/batchnorm2d.go:325-328`
**Correção:** Buffer `istdBuf` pre-alocado no struct, inicializado em Build() e LightweightClone()
**Status:** Resolvido via PERF-008 em 2026-03-09

---

## Otimizações de Cache Pendentes

### Dense Backward Sem Blocking
- **Localização:** `internal/layer/layer.go:447-461`
- **Problema:** Acesso stride-n em gradW
- **Solução:** Implementar blocking 64x64

### Conv2D Generic Fallback
- **Localização:** `internal/layer/conv2d.go:467-514`
- **Problema:** 5 níveis de nesting sem unrolling
- **Solução:** Expandir otimização 3x3 para outros kernels

### LSTM Sem SIMD
- **Localização:** `internal/layer/lstm.go:269-293`
- **Problema:** Loops escalares para matmul
- **Solução:** Unrolling 4x como em matmul_optimized

---

## GPU/Metal Otimizações

### Worker Limit para GPU
```go
if n.device.Type() == layer.GPU {
    numWorkers = min(4, runtime.GOMAXPROCS(0))
}
```

### Metal Sync Reduction
- Problema: Múltiplos `buf.Read()` síncronos por layer
- Solução: Kernel fusion, persistent memory mapping

### Conv2D Activation Não Fused
- Problema: Leitura GPU + ativação CPU
- Solução: Kernel fused Conv2D+Activation

---

## Checklist de Performance

### Ao criar novos exemplos:
- [ ] Usar `PredictBatch()` em vez de loop de `Predict()`
- [ ] Pré-alocar buffers de dados com slicing
- [ ] Usar `Fit()` com callbacks em vez de loops manuais
- [ ] Evitar alocações em loops de geração (GPT-like)

### Ao otimizar camadas:
- [ ] Verificar escape analysis (evitar heap em hot paths)
- [ ] Implementar blocking para matrizes grandes
- [ ] Adicionar loop unrolling (4x ou 8x)
- [ ] Cachear type assertions
- [ ] Pre-alocar buffers GPU

---

## Arquivos-Chave

| Arquivo | Responsabilidade |
|---------|------------------|
| `internal/layer/layer.go` | Dense, LightweightClone, ForwardWithArena |
| `internal/layer/matmul_optimized.go` | GEMM otimizado com blocking |
| `internal/layer/conv2d.go` | Conv2D com kernel 3x3 otimizado |
| `internal/net/net.go` | Network, TrainBatch, worker pool |
| `internal/layer/metal.go` | GPU kernels, buffer management |

---

## Métricas de Referência

| Operação | Antes | Depois | Melhoria |
|----------|-------|--------|----------|
| MNIST Predict (10k) | 2.5s | 0.3s | 8x |
| Dense Forward (1024x512) | 150us | 80us | 1.9x |
| Conv2D 3x3 (224x224) | 12ms | 4ms | 3x |
| Alocações/epoca | Alta | -80% | 5x |

---

## Auditoria de Performance 2026-03-07 - Resultados

### Problemas Críticos Identificados

| ID | Severidade | Descrição | Localização |
|----|------------|-----------|-------------|
| PERF-001 | ALTA | Loop Predict individual em exemplos | mnist, cifar10, sentiment |
| PERF-002 | ALTA | Alocação em data loading | Todos os exemplos |
| PERF-003 | ALTA | Arena resize dinâmico | layer.go:250 |
| PERF-004 | ALTA | Heap alloc em softmax backward | layer.go:399 |
| PERF-005 | ALTA | Metal buffer reallocation | conv2d.go:410 |
| PERF-006 | ALTA | Dense backward sem cache blocking | layer.go:447 |
| PERF-007 | MEDIA | istdBuf alocado em cada forward | batchnorm2d.go:325 |
| PERF-008 | MEDIA | Conv2D fallback sem unrolling | conv2d.go:467 |
| PERF-009 | MEDIA | LSTM gates sem SIMD | lstm.go:269 |
| PERF-010 | MEDIA | Worker pool sem limite GPU | net.go |
| PERF-011 | MEDIA | BatchNorm2D normalization sem blocking | batchnorm2d.go:333 |
| PERF-012 | MEDIA | Conv2D Metal activation não fused | conv2d.go:433 |
| PERF-013 | BAIXA | savedOutput realocado | layer.go:318 |
| PERF-020 | ALTA | gradCopy allocation em Metal backward | conv2d.go:692 |

### Total: 41 issues (7 críticas, 18 altas, 12 médias, 5 baixas)

### Relatório Completo
`/Users/flaviocfo/dev/github.com/FlavioCFOliveira/GoNeuron/AUDIT/performance_audit.md`

---

*Para detalhes completos, ver AUDIT/performance_audit.md*
*Última atualização: 2026-03-09*
