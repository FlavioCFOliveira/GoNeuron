# GoNeuron Roadmap 2026

## Visão
Transformar GoNeuron na biblioteca de deep learning em Go mais performática, estável e fácil de usar, mantendo a filosofia de zero dependências externas.

---

## TAREFAS PENDENTES

Tabela com as tarefas ainda por concluir, ordenadas por severidade (ALTA > MÉDIA > BAIXA).

### Auditoria de Segurança (SEC)

| ID | SEVERIDADE | TAREFA | DESCRIÇÃO TÉCNICA ACIONÁVEL |
| :--- | :--- | :--- | :--- |
| ~~SEC-008~~ | ~~MÉDIA~~ | ~~Prevenir integer overflow em cálculos de tamanho~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| ~~SEC-009~~ | ~~MÉDIA~~ | ~~Implementar cleanup explícito de buffers Metal~~ | ~~Implementado em 2026-03-09 - método Close() adicionado à interface Layer~~ |
| ~~SEC-011~~ | ~~MÉDIA~~ | ~~Reportar erro em índices inválidos de MultiMarginLoss~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| ~~SEC-012~~ | ~~MÉDIA~~ | ~~Validar offsets em AccumulateBackward~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| SEC-013 | BAIXA | Usar seed aleatória segura em RNG | Substituir seed previsível por `crypto/rand` ou aceitar seed externo em `internal/layer/layer.go:22-42`. |
| SEC-014 | BAIXA | Validar taxa de aprendizagem em SGD | Adicionar verificação `learningRate <= 0 || learningRate > 1` em `internal/opt/opt.go:40-42`. Retornar erro ou panic com mensagem clara. |
| SEC-015 | BAIXA | Corrigir arredondamento em Conv2D | Requerer especificação explícita de dimensões ou validar que o produto está correto em `internal/layer/conv2d.go:344-357`. |
| SEC-016 | BAIXA | Melhorar validação de tipos em API pública | Adicionar tratamento completo de type assertions em `goneuron/goneuron.go:69-110` para retornar erros em vez de panic. |
| SEC-017 | BAIXA | Verificar nil pointer em Conv2D Backward | Adicionar validação `if c.arenaPtr == nil` em `internal/layer/conv2d.go:585-596` antes de aceder a arena. Retornar `ErrForwardNotCalled`. |

### Auditoria de Performance (PERF)

| ID | SEVERIDADE | TAREFA | DESCRIÇÃO TÉCNICA ACIONÁVEL |
| :--- | :--- | :--- | :--- |
| ~~PERF-003~~ | ~~ALTA~~ | ~~Usar PredictBatch em funções evaluate~~ | ~~Já implementado - exemplos cifar10, stock_prediction, stock_prediction_bilstm usam PredictBatch~~ |
| ~~PERF-005~~ | ~~ALTA~~ | ~~Pre-calcular capacidade da Arena~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| ~~PERF-007~~ | ~~ALTA~~ | ~~Implementar pooling de buffers Metal para Conv2D~~ | ~~Já implementado - usa maxGradOutSize/maxGradInSize para pooling~~ |
| ~~PERF-009~~ | ~~ALTA~~ | ~~Implementar blocking em Dense Backward~~ | ~~Já implementado - blocking 64x64 com loop unrolling 4x~~ |
| ~~PERF-013~~ | ~~ALTA~~ | ~~Limitar workers para GPU em TrainBatchParallel~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| ~~PERF-016~~ | ~~ALTA~~ | ~~Reduzir sincronizações Metal em Dense~~ | ~~Implementado em 2026-03-09 - adicionado ReadMultiple()~~ |
| ~~PERF-019~~ | ~~ALTA~~ | ~~Eliminar heap allocation em Softmax Backward~~ | ~~Já implementado - usa dzBuf pre-alocado no struct Dense~~ |
| ~~PERF-020~~ | ~~ALTA~~ | ~~Eliminar gradCopy em Conv2D Metal Backward~~ | ~~Implementado em 2026-03-09 - usa scratchBuf protegido por arenaMu~~ |
| ~~PERF-004~~ | ~~MÉDIA~~ | ~~Verificar suporte a batching em Sequential PredictBatch~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| PERF-006 | MÉDIA | Garantir buffer pre-calculado em Dense savedOutput | Mover verificação de capacidade para `Build()` ou usar tamanho fixo em `internal/layer/layer.go:318-321`. |
| ~~PERF-008~~ | ~~MÉDIA~~ | ~~Mover istdBuf para struct pre-alocado em BatchNorm2D~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| ~~PERF-010~~ | ~~MÉDIA~~ | ~~Implementar kernel unrolling para Conv2D~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| PERF-011 | MÉDIA | Implementar SIMD em LSTM Gate Calculations | Usar loop unrolling 4x ou 8x como em `matmul_optimized.go` em `internal/layer/lstm.go:269-293`. |
| PERF-012 | MÉDIA | Documentar e avaliar NCHW vs NHWC em Conv2D | Documentar uso de NCHW em `internal/layer/conv2d.go` e considerar reordering para NHWC em CPUs sem AVX. |
| PERF-014 | MÉDIA | Otimizar LightweightClone para GPU | Modificar `internal/layer/layer.go:710-719` para workers partilharem buffers GPU, mantendo apenas buffers CPU privados. |
| PERF-015 | MÉDIA | Garantir reuso de Worker Pool entre épocas | Verificar em `internal/net/net.go` que `workerPool` persiste entre chamadas de `Fit()`. |
| PERF-017 | MÉDIA | Implementar kernel fused Conv2D+Activation Metal | Implementar kernel Metal que combina convolução e ativação em `internal/layer/conv2d.go:433-439` para evitar leitura completa para CPU. |
| ~~PERF-022~~ | ~~MÉDIA~~ | ~~Cachear resultado de type assertions~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| ~~PERF-023~~ | ~~MÉDIA~~ | ~~Cachear tipo de ativação em layer~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| PERF-024 | MÉDIA | Adicionar context cancellation em Worker Pool | Modificar `internal/layer/matmul_optimized.go:78-152` para aceitar `context.Context` e permitir cancellation. |
| PERF-028 | MÉDIA | Adicionar benchmarks para todas as camadas | Criar benchmarks em `internal/layer/benchmark_test.go` para BatchNorm2D, Conv2D variados, LSTM/GRU sequências longas. |
| PERF-032 | MÉDIA | Pre-tokenizar dataset em sentiment example | Modificar `examples/sentiment/main.go:144-156` para tokenizar dataset uma vez antes do loop de predict. |
| PERF-033 | MÉDIA | Reusar buffers de batch entre épocas em CIFAR-10 | Modificar `examples/cifar10/main.go:158-163` para reutilizar `batchX` e `batchY` entre épocas. |
| PERF-034 | MÉDIA | Usar PredictBatch em exemplos stock_prediction | Atualizar todos os exemplos `stock_prediction*` para usar batch prediction em vez de loop individual. |
| PERF-038 | MÉDIA | Usar optimizer.Step() em gan_mnist | Substituir loop manual de atualização em `examples/gan_mnist/main.go:215-220` por `optimizer.Step()` ou versão vectorizada. |
| ~~PERF-040~~ | ~~MÉDIA~~ | ~~Adicionar buffer reuse em Loss Functions~~ | ~~Implementado em 2026-03-09 - ver TAREFAS TERMINADAS~~ |
| PERF-018 | BAIXA | Considerar pool de buffers Metal partilhados | Avaliar em `internal/layer/layer.go:227-236` uso de pool de buffers Metal compartilhados entre layers. |
| PERF-021 | BAIXA | Otimizar GetPerformanceMetrics | Substituir `map[string]interface{}` em `internal/layer/matmul_optimized.go:302-319` por struct definido ou parâmetros de saída. |
| PERF-026 | BAIXA | Avaliar NCHW vs NHWC em Conv2D | Documentar decisão em `internal/layer/conv2d.go` e considerar transposição se necessário. |
| PERF-027 | BAIXA | Documentar layout de matrizes em MatMul | Documentar decisão row-major em `internal/layer/matmul_optimized.go` e considerar transposição explícita. |
| PERF-029 | BAIXA | Converter MeasureMatMulPerformance para benchmark padrão | Atualizar `internal/layer/matmul_optimized.go:221-263` para usar `b.ResetTimer()` e seguir padrão de benchmark Go. |
| PERF-030 | MÉDIA | Documentar uso correto de ForwardWithArena | Adicionar documentação em `internal/layer/layer.go:92-96` sobre quando usar arena vs forward normal, thread-safety, e responsabilidades do caller. |
| PERF-031 | BAIXA | Documentar diferença entre Clone e LightweightClone | Adicionar documentação em `internal/layer/layer.go:80-82` explicando: Clone para serialização, LightweightClone para workers. |
| PERF-039 | BAIXA | Otimizar initBuffers em Network | Avaliar cache de offsets calculados em `internal/net/net.go` para evitar recalcular em cada `Build()`. |
| PERF-041 | BAIXA | Documentar precondição de StepInPlace | Adicionar documentação em `internal/opt/opt.go` sobre requisito de não-overlap entre params e gradients. |

### Tarefas do Roadmap Original

| ID | SEVERIDADE | TAREFA | DESCRIÇÃO TÉCNICA ACIONÁVEL |
| :--- | :--- | :--- | :--- |
| ROAD-005 | ALTA | Adicionar bounds checking em funções críticas | Implementar verificação de limites em `argmax` e outras funções que acedem a slices por índice calculado. |
| ROAD-006 | MÉDIA | Otimizar BatchNorm2D forward | Reduzir tempo de execução de 1.37ms para < 0.5ms através de blocking e melhoria de acesso a memória. |
| ROAD-007 | MÉDIA | Otimizar Conv2DLarge | Reduzir tempo de execução de 210ms para < 70ms para input 64x64x64 através de otimizações de kernel e buffer pooling. |
| ROAD-008 | MÉDIA | Otimizar LSTM sequence processing | Reduzir tempo de 54ms para < 20ms para 10 timesteps através de SIMD e blocking. |
| ROAD-009 | MÉDIA | Implementar scripts de download automático para datasets | Criar scripts Go que descarreguem e preparem MNIST, CIFAR-10, e outros datasets automaticamente em `scripts/download/`. |
| ROAD-010 | MÉDIA | Criar exemplo best_practices.go | Desenvolver exemplo em `examples/best_practices/main.go` demonstrando uso correto de ForwardWithArena, PredictBatch, TrainBatch, e otimizações. |
| ROAD-011 | MÉDIA | Normalizar formato de logging entre exemplos | Definir padrão de logging estruturado e aplicar a todos os 37 exemplos em `examples/`. |
| ROAD-012 | MÉDIA | Adicionar rand.Seed consistente em todos os exemplos | Garantir que todos os exemplos usem seed explícito para reproducibilidade. |
| ROAD-013 | BAIXA | Substituir ioutil por os.ReadFile em 4 arquivos | Atualizar exemplos que usam `ioutil` depreciado para usar `os.ReadFile` ou `os.ReadDir`. |

---

## TAREFAS TERMINADAS

Tabela com as tarefas concluídas, ordenadas por data de conclusão (mais recente primeiro).

| ID | SEVERIDADE | TAREFA | CONCLUSÃO | DESCRIÇÃO TÉCNICA ACIONÁVEL |
| :--- | :--- | :--- | :--- | :--- |
| PERF-040 | MÉDIA | Buffer reuse em Loss Functions | 2026-03-09 | Modificados `MSE` e `CrossEntropy` para usar stack-allocated buffer em vez de buffer pre-alocado no struct. Elimina alocações em `Backward()`. |
| PERF-023 | MÉDIA | Cachear tipo de ativação em layer | 2026-03-09 | Adicionadas constantes `ActType*` para tipos de ativação. Campo `actType int` no struct `Dense` para dispatch mais eficiente via switch em vez de type assertions repetidas. |
| PERF-022 | MÉDIA | Cachear resultado de type assertions | 2026-03-09 | Adicionado campo `useMetal bool` no struct `Dense`. Inicializado em `SetDevice()` e usado em hot paths para evitar type assertions repetidas de `*MetalDevice`. |
| SEC-012 | MÉDIA | Validar offsets em AccumulateBackward | 2026-03-09 | Implementadas validações de bounds para `inOff` e `paOff` em `AccumulateBackward()`. Verificações contra tamanho da arena antes de construir slices. Retorna `ErrInvalidState` se inválido. |
| SEC-011 | MÉDIA | Reportar erro em índices MultiMarginLoss | 2026-03-09 | Alterado comportamento de `continue` silencioso para retornar `ErrInvalidTarget` quando índice target está fora de bounds `[0, n)`. |
| SEC-009 | MÉDIA | Implementar cleanup explícito de buffers Metal | 2026-03-09 | Adicionado método `Close()` à interface `Layer`. Implementado em todas as camadas para libertar buffers GPU. Previne memory leaks em execuções longas. |
| SEC-008 | MÉDIA | Prevenir integer overflow em cálculos de tamanho | 2026-03-09 | Usado `math/bits.Mul64()` para verificar overflow em `LSTM` e `Transformer`. Validação antes de alocar buffers. Retorna erro se overflow detectado. |
| PERF-010 | MÉDIA | Implementar kernel unrolling para Conv2D | 2026-03-09 | Implementados kernels otimizados `conv2D5x5Optimized` e `conv2D7x7Optimized` com unrolling completo. Processamento em blocos de 2 pixels para melhor ILP. Fallback para kernels genéricos. |
| PERF-008 | MÉDIA | Mover istdBuf para struct pre-alocado em BatchNorm2D | 2026-03-09 | Adicionado campo `istdBuf []float32` no struct `BatchNorm2D`. Pre-alocação em `Build()`. Elimina alocação em loop quente durante forward pass. Thread-safe via `LightweightClone`. |
| PERF-004 | MÉDIA | Verificar suporte a batching em Sequential PredictBatch | 2026-03-09 | Adicionada interface `BatchArenaLayer` e verificação em tempo de execução em `Sequential.PredictBatch()`. Fallback para forward individual se alguma camada não suportar batching. |
| PERF-040 | MÉDIA | Adicionar buffer reuse em Loss Functions | 2026-03-09 | Modificado `MSE` e `CrossEntropy` para usar value receivers em vez de pointer receivers. Elimina alocações temporárias em Backward.
| PERF-023 | MÉDIA | Cachear tipo de ativação em layer | 2026-03-09 | Adicionadas constantes `ActTypeNone`, `ActTypeReLU`, etc. Campo `actType int` em struct Dense para dispatch rápido via switch. Type assertion feita apenas uma vez na construção.
| PERF-022 | MÉDIA | Cachear resultado de type assertions | 2026-03-09 | Adicionado campo `useMetal bool` em struct Dense. Inicializado em `SetDevice()`. Substitui type assertions repetidas em 10+ hot paths.
| PERF-010 | MÉDIA | Implementar kernel unrolling para Conv2D | 2026-03-09 | Implementados kernels otimizados `conv2D5x5Optimized` e `conv2D7x7Optimized` em `conv2d.go`. Unrolling completo dos kernels. Fallback para implementação genérica.
| PERF-008 | MÉDIA | Mover istdBuf para struct pre-alocado em BatchNorm2D | 2026-03-09 | Adicionado campo `istdBuf []float32` no struct. Pre-alocação em `Build()`. Elimina alocação em loop quente durante forward pass.
| PERF-004 | MÉDIA | Verificar suporte a batching em Sequential PredictBatch | 2026-03-09 | Adicionada interface `BatchArenaLayer`. Verificação em tempo de execução se todas as camadas suportam `ForwardBatchWithArena`. Fallback para forward individual se necessário.
| SEC-012 | MÉDIA | Validar offsets em AccumulateBackward | 2026-03-09 | Adicionadas verificações de bounds para `inOff` e `paOff` em `AccumulateBackward`. Retorna erro descritivo se offsets inválidos.
| SEC-011 | MÉDIA | Reportar erro em índices inválidos de MultiMarginLoss | 2026-03-09 | Índices inválidos (`target < 0 || target >= n`) retornam `ErrInvalidTarget` em vez de `continue` silencioso.
| SEC-009 | MÉDIA | Implementar cleanup explícito de buffers Metal | 2026-03-09 | Adicionado método `Close()` à interface Layer. Implementado em todas as camadas (Dense, Conv2D, LSTM, GRU, BatchNorm2D, etc.). Previne memory leaks de buffers GPU.
| SEC-008 | MÉDIA | Prevenir integer overflow em cálculos de tamanho | 2026-03-09 | Usa `math/bits.Mul64` para detectar overflow em multiplicações. Aplicado em `LSTM.Build()` e `Transformer`. Retorna erro se overflow detectado.
| PERF-020 | ALTA | Eliminar gradCopy em Conv2D Metal Backward | 2026-03-09 | Usa `scratchBuf` pre-alocado protegido por `arenaMu` para receber dados da GPU. Elimina alocações temporárias `tempGradW`/`tempGradB`. Thread-safe para worker pools. |
| PERF-016 | ALTA | Reduzir sincronizações Metal em Dense | 2026-03-09 | Adicionado `ReadMultiple()` em `metal.go` para consolidar múltiplos reads em uma única operação. Aplicado em Dropout, LSTM, BatchNorm2D para reduzir overhead CPU-GPU. |
| PERF-013 | ALTA | Limitar workers para GPU em TrainBatchParallel | 2026-03-09 | Modificado `trainBatchParallel` em `net.go` para usar `min(4, GOMAXPROCS(0))` quando device é GPU. Reduz overhead de sincronização CGO com drivers Metal/CUDA. |
| PERF-005 | ALTA | Pre-calcular capacidade da Arena | 2026-03-09 | Implementado `CalculateTotalArenaSize()` que percorre camadas e soma `ArenaSize()`. Arena pre-alocada em `initBuffers()` com capacidade exata. Elimina redimensionamentos dinâmicos. Assertions verificam capacidade em `ForwardWithArena`. |
| PERF-003 | ALTA | Usar PredictBatch em funções evaluate | 2026-03-09 | Verificação concluída. Exemplos cifar10, stock_prediction, stock_prediction_bilstm já usam `PredictBatch()`. Nenhuma alteração necessária. |
| PERF-009 | ALTA | Implementar blocking em Dense Backward | 2026-03-09 | Verificação concluída. Já implementado blocking 64x64 com loop unrolling 4x em `layer.go:510-540` para melhorar cache locality. |
| PERF-019 | ALTA | Eliminar heap allocation em Softmax Backward | 2026-03-09 | Verificação concluída. Já usa `dzBuf` pre-alocado no struct Dense, eliminando alocações em cada backward. |
| PERF-007 | ALTA | Implementar pooling de buffers Metal para Conv2D | 2026-03-09 | Verificação concluída. Já implementado com `maxGradOutSize` e `maxGradInSize` para reter buffers GPU entre passos. |
| ROAD-005 | ALTA | Adicionar bounds checking em funções críticas | 2026-03-09 | Verificação concluída. `argmax` em exemplos verifica slice vazio. `Embedding.ForwardWithArena` valida índices contra `[0, num_embeddings)`. |
| SEC-007 | MÉDIA | Validar versão GGUF em Load/Save | 2026-03-09 | Adicionadas constantes GGUFMinVersion/GGUFMaxVersion. Implementada função ValidateGGUFVersion e struct GGUFReader com ReadHeader que valida magic number e versão. Erros explícitos: ErrInvalidMagic, ErrVersionTooOld, ErrVersionTooNew. |
| SEC-003 | ALTA | Proteger Arena Operations contra race conditions | 2026-03-09 | Adicionado `sync.Mutex` (arenaMu) a GlobalAttention, Embedding, LayerNorm, RMSNorm, SwiGLU, Bidirectional. Proteção em ForwardWithArena contra concorrência em workloads paralelos. |
| SEC-007 | MÉDIA | Validar versão GGUF em Load/Save | 2026-03-09 | Adicionadas constantes `GGUFMinVersion` e `GGUFMaxVersion`. Implementada função `ValidateGGUFVersion()` e `GGUFReader.ReadHeader()` com validação de magic number e versão em `internal/net/gguf.go`. |
| SEC-010 | MÉDIA | Substituir panic por error returns em Activations | 2026-03-09 | Modificados `Softmax.Activate`, `Softmax.Derivative`, `LogSoftmax.Activate`, `LogSoftmax.Derivative` em `internal/activations/activations.go:149-157` para retornar `NaN` em vez de panic. API mantida compatível. |
| SEC-005 | MÉDIA | Validar parâmetros em construtores de Layer | 2026-03-09 | Modificados todos os construtores de layer (NewDense, NewConv2D, NewLSTM, NewGRU, etc.) para retornar `(*Type, error)`. Corrigido bug em `SequenceUnroller.ForwardWithArena` linha 126 que retornava tamanho incorreto. |
| SEC-001 | ALTA | Corrigir divisão por zero em Loss Functions | 2026-03-07 | Adicionada verificação `n == 0` em todas as loss functions (MSE, L1Loss, BCELoss, CrossEntropy, NLLLoss, etc.) em `internal/loss/loss.go`. Retorna erro `ErrEmptyInput` se batch size for zero. |
| SEC-002 | ALTA | Implementar bounds checking em Embedding Layer | 2026-03-07 | Validados índices em `internal/layer/embedding.go:114-128`. Usa `math.Floor` para conversão float-to-int e retorna erro se índice estiver fora de `[0, num_embeddings)`. |
| SEC-004 | ALTA | Adicionar validação de limites em CSV Loader | 2026-03-07 | Implementados `maxCSVRows=10_000_000`, `maxCSVCols=10_000`, `maxFileSize=1GB` em `internal/net/csv_loader.go:20-97`. Verificação de `os.Stat` antes de carregar. |
| ROAD-001 | ALTA | Corrigir divisões por zero em 5 exemplos stock_prediction | 2026-03-07 | Adicionada verificação de `len(xTest) > 0` antes de divisão em todos os exemplos stock_prediction. Corrigido em: stock_prediction, stock_prediction_attention, stock_prediction_bilstm, stock_prediction_cnn_lstm, stock_prediction_gru. |
| ROAD-002 | ALTA | Verificar Softmax/CrossEntropy em portuguese_gpt e portuguese_qa | 2026-03-07 | Verificação concluída. Exemplos já usam configuração correta: portuguese_gpt usa Linear+CrossEntropy, portuguese_qa usa LogSoftmax+NLLLoss. Nenhuma alteração necessária. |
| ROAD-003 | ALTA | Corrigir autoencoder_synthetic | 2026-03-07 | Dados normalizados para intervalo [-1, 1] em vez de [0, 1). Comentário atualizado em `examples/autoencoder_synthetic/main.go`. |
| ROAD-004 | ALTA | Adicionar verificação de erros em I/O binário | 2026-03-07 | Adicionada validação de magic numbers, número de bytes lidos, dimensões e dados extra em MNIST e CIFAR-10 loaders. Verificações em `examples/mnist/main.go` e `examples/cifar10/main.go`. |
| PERF-001 | ALTA | Substituir loops Predict por PredictBatch em exemplos | 2026-03-07 | Modificados `examples/mnist/main.go`, `examples/cifar10/main.go`, `examples/sequential_mnist/main.go`, `examples/sentiment/main.go` para usar `PredictBatch()`. Redução de ~90% nas alocações durante inferência. |
| PERF-002 | ALTA | Pre-alocar buffers contíguos em data loading | 2026-03-07 | Modificados `examples/mnist/main.go`, `examples/cifar10/main.go`, `examples/sequential_mnist/main.go` para usar buffer contíguo com slicing. Elimina N alocações (uma por sample). |
| PERF-035 | ALTA | Pre-alocar buffer de logits em geração de texto | 2026-03-07 | Modificados `examples/portuguese_gpt/main.go`, `examples/dialogue_bot/main.go`, `examples/poetry_generator/main.go` para pre-alocar logits fora do loop de geração. Reutilização via `copy()` em cada iteração. |
| PHASE3-001 | ALTA | BatchNorm2D performance optimization | 2026-03-07 | Otimização de forward pass com melhoria de cache locality. Commit: ebcb3f8 |
| PHASE3-002 | ALTA | Conv2D performance optimization | 2026-03-07 | Otimização de loops internos e buffer management. Commit: ebcb3f8 |
| PHASE3-003 | ALTA | MatMul optimized implementation | 2026-03-07 | Implementação de blocking e cache-efficient matrix multiplication. Commit: ebcb3f8 |
| PHASE1-001 | ALTA | Corrigir CrossEntropy Loss gradientes | 2026-03-06 | Correção de cálculo de gradientes na função CrossEntropy. Commit: 89f9371 |
| PHASE1-002 | ALTA | Corrigir NLLLoss forward/backward | 2026-03-06 | Validação de valores e correção de cálculo. Commit: 89f9371 |
| PHASE1-003 | ALTA | Corrigir exemplo portuguese_qa imports | 2026-03-06 | Correção de imports quebrados no exemplo. Commit: 89f9371 |
| PHASE1-004 | ALTA | Validar gradientes contra PyTorch | 2026-03-06 | Implementação de suite de validação comparativa. Commit: 89f9371 |
| PHASE1-005 | ALTA | Corrigir bug MoE layer index out of range | 2026-03-06 | Modificar `Sequential.Build` para construção correta de experts. Commit: ef9af84 |
| PHASE1-006 | ALTA | Corrigir time_series_forecast denormalização | 2026-03-06 | Corrigir cálculo de índice de feature em `Denormalize`. Commit: ef9af84 |
| PHASE1-007 | ALTA | Corrigir sentiment_imdb Flatten | 2026-03-06 | Remover `Flatten()` antes de `SequenceUnroller`. Commit: ef9af84 |
| PHASE2-001 | MÉDIA | Criar testes para package goneuron | 2026-03-06 | Cobertura de API pública com testes unitários. Commit: 89f9371 |
| PHASE2-002 | MÉDIA | Criar testes de integração para mnist/cifar10 | 2026-03-06 | Testes end-to-end para exemplos principais. Commit: 89f9371 |
| PHASE2-003 | MÉDIA | Implementar testes de gradiente numérico | 2026-03-06 | Validação matemática para todas as camadas. Commit: 89f9371 |
| PHASE2-004 | MÉDIA | Configurar GitHub Actions | 2026-03-06 | CI/CD para testes automáticos. Commit: 89f9371 |
| PHASE2-005 | MÉDIA | Adicionar análise estática (go vet, staticcheck) | 2026-03-06 | Integração de linters no CI. Commit: 89f9371 |
| PHASE2-006 | MÉDIA | Adicionar golangci-lint com regras de segurança | 2026-03-06 | Configuração de regras de segurança no linter. Commit: 89f9371 |
| PHASE2-007 | MÉDIA | Expandir cobertura de edge cases | 2026-03-06 | Testes para NaN/Inf, dimensões extremas, inputs vazios. Commit: 89f9371 |
| PHASE1-008 | MÉDIA | Substituir panic por error returns | 2026-03-06 | Refatoração de APIs públicas para retornar erros. Commit: 9853d08 |
| PHASE1-009 | MÉDIA | Adicionar validação de inputs em camadas | 2026-03-06 | Verificação de dimensões e valores em construtores. Commit: 9853d08 |
| PHASE1-010 | MÉDIA | Implementar logging estruturado | 2026-03-06 | Adicionar logging para depuração. Commit: 9853d08 |

---

## Resumo das Auditorias

### Auditoria de Segurança (2026-03-07)
- **Total de vulnerabilidades identificadas:** 17
- **Severidade ALTA:** 4 issues (SEC-001 a SEC-004)
- **Severidade MÉDIA:** 8 issues (SEC-005 a SEC-012)
- **Severidade BAIXA:** 5 issues (SEC-013 a SEC-017)
- **Relatório completo:** `AUDIT/security_audit.md`

### Auditoria de Performance (2026-03-07)
- **Total de problemas identificados:** 41
- **Severidade ALTA:** 18 issues (PERF-001 a PERF-020)
- **Severidade MÉDIA:** 12 issues (PERF-004, PERF-006, PERF-008, PERF-010 a PERF-012, PERF-014, PERF-015, PERF-017, PERF-022 a PERF-024, PERF-028, PERF-030, PERF-032 a PERF-040)
- **Severidade BAIXA:** 11 issues (PERF-018, PERF-021, PERF-026, PERF-027, PERF-029, PERF-031, PERF-039, PERF-041)
- **Relatório completo:** `AUDIT/performance_audit.md`

---

## Fases do Roadmap (Histórico)

### Fase 1: Estabilidade e Correções Críticas (Sprint 1-2) ✅ COMPLETA
**Objetivo:** Corrigir bugs críticos que afetam a correção matemática e a compilação.

### Fase 2: Qualidade e Testes (Sprint 3-4) ✅ COMPLETA
**Objetivo:** Aumentar cobertura de testes e garantir estabilidade.

### Fase 3: Performance e Otimização (Sprint 5-7) 🔄 EM PROGRESSO
**Objetivo:** Otimizar performance em CPU e expandir suporte GPU.

### Fase 4: Novas Funcionalidades (Sprint 8-10) ⏳ PENDENTE
**Objetivo:** Expandir capacidades da biblioteca.

### Fase 5: Developer Experience (Sprint 11-12) ⏳ PENDENTE
**Objetivo:** Melhorar a experiência de desenvolvimento.

### Fase 6: Produção e Deployment (Sprint 13-14) ⏳ PENDENTE
**Objetivo:** Tornar a biblioteca pronta para produção.

---

## Dependências entre Fases

```
Fase 1 (Estabilidade)
    ↓
Fase 2 (Testes) → Testes passam
    ↓
Fase 3 (Performance) → Baseline estabelecido
    ↓
Fase 4 (Features) → Core estável
    ↓
Fase 5 (DX) → APIs definidas
    ↓
Fase 6 (Produção) → Tudo validado
```

---

## Notas

- Cada sprint assume 2 semanas de trabalho
- Ajustes ao roadmap devem ser discutidos em issues
- Prioridades podem mudar baseadas em feedback da comunidade
- Métricas devem ser medidas e documentadas em cada fase
- Itens **AUDIT:** são findings de auditoria e devem ser priorizados
