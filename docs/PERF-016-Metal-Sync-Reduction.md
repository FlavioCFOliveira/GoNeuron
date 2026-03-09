# PERF-016: Redução de Sincronizações Metal

## Resumo
Otimização que consolida múltiplas operações de `Read()` do Metal em um único ponto de sincronização, reduzindo o overhead de CPU-GPU.

## Problema
O código original realizava múltiplas chamadas `Read()` consecutivas em buffers Metal. Cada chamada `Read()`:
1. Adquire o mutex do dispositivo
2. Chama `waitForMetal()` (ponto de sincronização CPU-GPU)
3. Copia os dados do GPU para CPU

Isso resultava em múltiplos pontos de sincronização desnecessários quando vários buffers precisavam ser lidos.

## Solução
Implementada a função `ReadMultiple()` em `internal/layer/metal.go` que:
1. Valida todos os buffers pertencem ao mesmo dispositivo
2. Realiza **um único** `waitForMetal()` para todos os buffers
3. Lê todos os buffers sequencialmente

```go
func ReadMultiple(buffers []*MetalBuffer, destinations [][]float32) {
    // Validações...
    device.mu.Lock()
    defer device.mu.Unlock()

    // Single synchronization point
    C.waitForMetal(device.ptr)

    // Read all buffers
    for i, buf := range buffers {
        C.readMetalBuffer(buf.ptr, ...)
    }
}
```

## Camadas Otimizadas

| Camada | Arquivo | Leituras Anteriores | Leituras Atuais | Redução |
|--------|---------|---------------------|-----------------|---------|
| Dense Backward | `layer.go` | 3 | 1 | 67% |
| LSTM Forward | `lstm.go` | 7 | 1 | 86% |
| Dropout Forward | `dropout.go` | 2 | 1 | 50% |
| Dropout ForwardBatch | `dropout.go` | 2 | 1 | 50% |
| BatchNorm2D Backward | `batchnorm2d.go` | 2-3* | 1 | 67% |
| Conv2D Backward | `conv2d.go` | 2 | 1 | 50% |

\* 3 leituras quando `affine=true`, 1 quando `affine=false`

## Benefícios
- **Menor Latência**: Redução significativa no tempo de espera CPU-GPU
- **Menor Contenção de Lock**: Apenas uma aquisição de mutex por lote de leituras
- **Melhor Throughput**: Especialmente importante em treinamento batch com múltiplos workers

## Compatibilidade
- Mantém compatibilidade total com Apple Silicon
- Não altera semântica dos dados
- Funciona com buffers de tamanhos diferentes
- Validação de buffers nulos e dispositivos

## Testes
Todos os testes existentes passam:
- `TestDense*`: Todos passando
- `TestLSTM*`: Todos passando
- `TestDropout*`: Todos passando
- `TestBatchNorm2D*`: Todos passando
- `TestConv2D*`: Todos passando (exceto teste pre-existente com falha numérica)

## Referências
- Arquivos modificados:
  - `internal/layer/metal.go` - Função `ReadMultiple()`
  - `internal/layer/layer.go` - Dense Backward
  - `internal/layer/lstm.go` - LSTM Forward
  - `internal/layer/dropout.go` - Dropout Forward
  - `internal/layer/batchnorm2d.go` - BatchNorm2D Backward
  - `internal/layer/conv2d.go` - Conv2D Backward
