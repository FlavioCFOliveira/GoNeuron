# GoNeuron Roadmap 2026

## Visão
Transformar GoNeuron na biblioteca de deep learning em Go mais performática, estável e fácil de usar, mantendo a filosofia de zero dependências externas.

## Fases do Roadmap

---

## Fase 1: Estabilidade e Correções Críticas (Sprint 1-2) ✅ COMPLETA
**Objetivo:** Corrigir bugs críticos que afetam a correção matemática e a compilação.

### 1.1 Correções Críticas
- [x] Corrigir implementação de CrossEntropy Loss (gradientes incorretos) ✅
- [x] Corrigir exemplo portuguese_qa (imports quebrados) ✅
- [x] Corrigir NLLLoss (valores forward/backward incorretos) ✅
- [x] Validar gradientes de todas as funções de perda contra PyTorch ✅
- [x] **AUDIT:** Corrigir bug MoE layer (index out of range no backward) ✅ Corrigido
- [x] **AUDIT:** Corrigir bug time_series_forecast (denormalização com índice errado) ✅ Corrigido
- [x] **AUDIT:** Corrigir bug sentiment_imdb (Flatten antes de SequenceUnroller) ✅ Corrigido
- [x] **AUDIT:** Corrigir divisões por zero em stock_prediction* (5 exemplos) ✅
- [x] **AUDIT:** Corrigir arquitetura Softmax/CrossEntropy em portuguese_gpt e portuguese_qa ✅
- [x] **AUDIT:** Corrigir autoencoder_synthetic (Sigmoid → Tanh, normalização) ✅

### 1.2 Tratamento de Erros
- [x] Substituir panic() por error returns nas APIs públicas ✅
- [x] Adicionar validação de inputs em todas as camadas ✅
- [x] Implementar logging estruturado para depuração ✅
- [x] **AUDIT:** Adicionar verificação de erros em operações de I/O binário (MNIST loaders) ✅
- [x] **AUDIT:** Adicionar bounds checking em argmax (2 exemplos) ✅
- [x] **AUDIT:** Validar tamanho de dados lidos em loaders MNIST/CIFAR ✅
- [x] **AUDIT:** Adicionar verificação de erros em operações de I/O binário (MNIST, CIFAR) ✅
- [x] **AUDIT:** Validar retorno de leituras de arquivos gzip (não ignorar bytes lidos) ✅
- [x] **AUDIT:** Adicionar bounds checking em funções como argmax ✅

**Métricas de Sucesso:**
- 100% dos testes passam
- 0 panics em código de produção
- Todas as losses validadas contra referências PyTorch

---

## Fase 2: Qualidade e Testes (Sprint 3-4) ✅ COMPLETA
**Objetivo:** Aumentar cobertura de testes e garantir estabilidade.

### 2.1 Cobertura de Testes
- [x] Criar testes para package goneuron (API pública) ✅ Implementado
- [x] Criar testes de integração para exemplos principais (mnist, cifar10) ✅ Implementado
- [x] Implementar testes de gradiente numérico para todas as camadas ✅
- [x] Adicionar testes de estabilidade numérica ✅
- [ ] **AUDIT:** Adicionar testes de estresse com batches grandes (512, 1024+) ⏭️ (Fase 6)
- [x] **AUDIT:** Expandir cobertura de edge cases (NaN/Inf, dimensões extremas, inputs vazios) ✅
- [ ] **AUDIT:** Implementar testes de regressão comparando outputs entre releases ⏭️ (Fase 6)

### 2.2 Validação
- [x] Criar suite de validação automática contra PyTorch ✅ Implementado (com geração de referências)
- [x] Implementar testes de convergência para cada tipo de camada ✅ Implementado
- [x] Validar backward pass de todas as camadas complexas (LSTM, GRU, Transformer) ✅ Implementado
- [x] **AUDIT:** Aumentar tolerância em gradient check numérico para Huber loss ✅ Corrigido
- [ ] **AUDIT:** Criar benchmarks comparativos Metal vs CPU para todas as camadas ⏭️ (Fase 3)

### 2.3 CI/CD
- [x] Configurar GitHub Actions para testes automáticos ✅
- [x] Adicionar análise estática (go vet, staticcheck, golint) ✅
- [x] Adicionar golangci-lint com regras de segurança ✅
- [x] Configurar code coverage reporting ✅
- [x] **AUDIT:** Adicionar testes de segurança automatizados (detectar divisões por zero, etc.) ✅

**Métricas de Sucesso:**
- >80% cobertura de testes
- Suite de validação PyTorch automatizada
- CI/CD funcional
- Zero discrepâncias numéricas > 1e-5 em gradient checks
- 100% dos benchmarks com < 100ms para camadas standard

---

## Fase 3: Performance e Otimização (Sprint 5-7)
**Objetivo:** Otimizar performance em CPU e expandir suporte GPU.

### 3.1 Otimizações CPU
- [ ] Implementar SIMD para operações de MatMul (usando GOEXPERIMENT=sse2/avx)
- [ ] Otimizar loops críticos com loop unrolling manual
- [ ] Implementar cache blocking para operações de grande dimensão
- [ ] Adicionar detecção automática de CPU features
- [ ] **AUDIT:** Otimizar BatchNorm2D forward (atual: 1.37ms - gargalo identificado)
- [ ] **AUDIT:** Otimizar Conv2DLarge (atual: 210ms para 64x64x64 input)
- [ ] **AUDIT:** Otimizar LSTM sequence processing (atual: 54ms para 10 timesteps)
- [ ] **AUDIT:** Substituir loops de Predict por PredictBatch em 8 exemplos
- [ ] **AUDIT:** Otimizar exemplos que usam Train() em vez de TrainBatch()
- [ ] **AUDIT:** Reduzir alocações em loops de dados (10+ exemplos)

### 3.2 Expansão Metal
- [ ] Adicionar kernels para todas as camadas LSTM/GRU
- [ ] Implementar backward pass completo em Metal
- [ ] Otimizar kernels de Transformer para Metal
- [ ] Adicionar profiling automático de kernels
- [ ] **AUDIT:** Implementar kernels Metal para Conv2D grandes (otimizar 64x64x64→128 canais)
- [ ] **AUDIT:** Adicionar batching de operações Metal para reduzir overhead de sincronização

### 3.3 CUDA Support (Opcional)
- [ ] Criar backend CUDA para Linux/Windows
- [ ] Implementar kernels básicos (MatMul, Conv2D)
- [ ] Adicionar memory pool para GPU

**Métricas de Sucesso:**
- 2x speedup em operações críticas
- <50% tempo de treino em modelos CNN/RNN

---

## Fase 4: Novas Funcionalidades (Sprint 8-10)
**Objetivo:** Expandir capacidades da biblioteca.

### 4.1 Novas Camadas
- [ ] Implementar Conv1D para séries temporais
- [ ] Adicionar Conv3D para dados volumétricos
- [ ] Implementar Depthwise Separable Convolution
- [ ] Adicionar Dilated Convolution
- [ ] Implementar ResidualBlock e DenseBlock

### 4.2 Novos Otimizadores
- [ ] RMSprop
- [ ] Adagrad / Adadelta
- [ ] AdamW (com weight decay correto)
- [ ] LAMB (para treino de grandes modelos)

### 4.3 Funcionalidades de Treino
- [ ] Learning rate scheduling (Cosine Annealing, Warmup)
- [ ] Gradient clipping configurável
- [ ] Mixed precision training (FP16)
- [ ] Gradient accumulation

### 4.4 Data Augmentation
- [ ] Transformações para imagens (rotate, flip, crop)
- [ ] Token augmentation para NLP
- [ ] Pipeline de pré-processamento

**Métricas de Sucesso:**
- 10+ novas camadas implementadas
- 4+ novos otimizadores
- Suporte a mixed precision

---

## Fase 5: Developer Experience (Sprint 11-12)
**Objetivo:** Melhorar a experiência de desenvolvimento.

### 5.1 Documentação
- [ ] Gerar documentação GoDoc completa
- [ ] Criar tutoriais interativos (notebooks)
- [ ] Adicionar guia de migração de PyTorch/TensorFlow
- [ ] Documentar APIs internas para contribuidores
- [ ] **AUDIT:** Criar guia de melhores práticas de performance
- [ ] **AUDIT:** Documentar uso correto de ForwardWithArena
- [ ] **AUDIT:** Documentar quando usar GPU vs CPU
- [ ] **AUDIT:** Documentar uso correto de ForwardWithArena
- [ ] **AUDIT:** Criar guia de quando usar GPU vs CPU
- [ ] **AUDIT:** Documentar padrões de segurança (evitar divisão por zero)

### 5.2 Ferramentas
- [ ] Criar CLI para treino de modelos
- [ ] Implementar visualizador de arquitetura
- [ ] Adicionar profiler integrado
- [ ] Criar ferramenta de exportação para ONNX

### 5.3 Exemplos
- [ ] Criar exemplo de Computer Vision avançado (ResNet)
- [ ] Adicionar exemplo de NLP (BERT-style)
- [ ] Implementar exemplo de Reinforcement Learning
- [ ] Criar exemplo de GAN condicional
- [ ] **AUDIT:** Criar exemplo best_practices.go demonstrando todas as otimizações
- [ ] **AUDIT:** Normalizar formato de logging entre exemplos
- [ ] **AUDIT:** Adicionar rand.Seed consistente em todos os exemplos
- [ ] **AUDIT:** Criar scripts de download automático para datasets (MNIST, CIFAR-10, etc.)
- [ ] **AUDIT:** Substituir ioutil por os.ReadFile em 4 arquivos

**Métricas de Sucesso:**
- Documentação 100% das APIs públicas
- 5+ tutoriais interativos
- CLI funcional

---

## Fase 6: Produção e Deployment (Sprint 13-14)
**Objetivo:** Tornar a biblioteca pronta para produção.

### 6.1 Model Serving
- [ ] Implementar inference server (HTTP/gRPC)
- [ ] Adicionar batching dinâmico
- [ ] Otimizar para latência baixa
- [ ] Suporte a modelos quantizados

### 6.2 Quantização
- [ ] Implementar PTQ (Post-Training Quantization)
- [ ] Adicionar QAT (Quantization Aware Training)
- [ ] Suporte a INT8 e INT4
- [ ] Integrar com GGUF para modelos quantizados

### 6.3 Observabilidade
- [ ] Adicionar métricas de treino (wandb/tensorboard style)
- [ ] Implementar tracing de operações
- [ ] Criar dashboard de monitoramento

### 6.4 Testes de Carga e Estresse (Audit findings)
- [ ] **AUDIT:** Implementar testes de longa duração (verificar estabilidade numérica)
- [ ] **AUDIT:** Adicionar testes com precisão mista (FP16/FP32)
- [ ] **AUDIT:** Criar suite de fuzzing para inputs aleatórios
- [ ] **AUDIT:** Implementar testes de memória (detectar leaks)
- [ ] **AUDIT:** Adicionar validação de outputs entre diferentes backends (CPU vs Metal)

### 6.4 Testes de Carga e Estresse
- [ ] **AUDIT:** Testes com batches grandes (512, 1024, 2048 amostras)
- [ ] **AUDIT:** Testes de longa duração (1000+ épocas) para verificar estabilidade
- [ ] **AUDIT:** Testes com precisão mista (FP16) para todas as camadas
- [ ] **AUDIT:** Testes de memória (verificar leaks em treinos prolongados)
- [ ] **AUDIT:** Testes de concorrência (treino paralelo em múltiplos modelos)

**Métricas de Sucesso:**
- Inference server com <10ms latência
- Redução 4x em tamanho de modelos quantizados
- Zero falhas em testes de estresse (24h+ execução)
- 100% validação cruzada entre backends (CPU/Metal)

---

## Tarefas Contínuas (Backlog)

### Manutenção
- [ ] Atualizar para Go 1.24 quando lançado
- [ ] Revisar e atualizar dependências (zero por agora ✓)
- [ ] Corrigir warnings do linter
- [ ] Otimizar alocações de memória baseadas em profiling

### Comunidade
- [ ] Criar template para issues
- [ ] Adicionar guia de contribuição
- [ ] Implementar code review guidelines
- [ ] Criar changelog automatizado

---

## Cronograma Resumido

| Fase | Duração | Foco |
|------|---------|------|
| 1 | Semanas 1-2 | Estabilidade |
| 2 | Semanas 3-4 | Testes |
| 3 | Semanas 5-7 | Performance |
| 4 | Semanas 8-10 | Features |
| 5 | Semanas 11-12 | DX |
| 6 | Semanas 13-14 | Produção |

---

## Notas da Auditoria de Testes (Março 2026)

### Resumo da Auditoria
Foi realizada uma auditoria profunda nos testes da biblioteca GoNeuron com os seguintes resultados:

#### ✅ Status Geral
- **100% dos testes passando** (~200+ funções de teste)
- **Zero alocações** na maioria dos benchmarks (zero-allocation pattern)
- **Convergência verificada** para todos os tipos de camadas
- **Gradientes validados** para todas as camadas (Dense, Conv2D, LSTM, GRU, Transformer)

#### 🎯 Pontos Fortes Identificados
1. Excelente cobertura de testes de gradiente (`gradient_check_test.go`)
2. Validação PyTorch integrada (`pytorch_validation_test.go`)
3. Benchmarks com zero alocações (padrão GoNeuron)
4. Testes de convergência para todos os otimizadores
5. Testes de segurança para modelos (LoadSecurity)

#### ⚠️ Pontos de Melhoria (Adicionados ao Roadmap)
- **Performance**: BatchNorm2D (1.37ms), Conv2DLarge (210ms), LSTM Sequence (54ms)
- **Precisão**: Ajustar tolerância em gradient check numérico para Huber loss
- **Cobertura**: Adicionar testes de estresse, edge cases, e regressão
- **Metal**: Expandir benchmarks comparativos CPU vs Metal

### Legenda
Itens marcados com **AUDIT:** no Roadmap foram adicionados com base nos resultados desta auditoria.

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

## Notas da Auditoria de Testes (2026-03-06)

Realizada auditoria profunda na suite de testes com os seguintes resultados:

### ✅ Status Atual
- **100% testes passando** (~200+ testes)
- **Convergência validada**: XOR, regressão linear, otimizadores
- **Gradientes verificados**: Todas as camadas (Dense, Conv2D, LSTM, GRU, Transformer)
- **Performance**: Zero alocações na maioria dos benchmarks

### 📋 Itens Auditados e Adicionados ao Roadmap

Itens marcados com **AUDIT:** foram adicionados baseados nos findings:

| Prioridade | Item | Fase |
|------------|------|------|
| **P1** | Corrigir divisões por zero em stock_prediction* | Fase 1 |
| **P1** | Corrigir Softmax/CrossEntropy em exemplos GPT | Fase 1 |
| **P1** | Corrigir autoencoder_synthetic | Fase 1 |
| **P1** | Verificar erros I/O em loaders MNIST | Fase 1 |
| **P2** | Substituir loops Predict por PredictBatch | Fase 3 |
| **P2** | Otimizar exemplos com TrainBatch | Fase 3 |
| **P2** | Testes de estresse com batches grandes | Fase 2 |
| **P2** | Cobertura de edge cases (NaN/Inf) | Fase 2 |
| **P2** | Testes de regressão entre releases | Fase 2 |
| **P3** | Expandir benchmarks Metal vs CPU | Fase 2 |
| **P3** | Corrigir tolerância gradient check Huber | Fase 2 |
| **P2** | Otimizar BatchNorm2D forward | Fase 3 |
| **P2** | Otimizar Conv2DLarge | Fase 3 |
| **P2** | Otimizar LSTM sequence | Fase 3 |
| **P3** | Substituir ioutil por os.ReadFile | Fase 2 |
| **P3** | Adicionar bounds checking | Fase 2 |
| **P3** | Scripts de download automático | Fase 5 |
| **P3** | Criar exemplo best_practices.go | Fase 5 |
| P3 | Testes de longa duração | Fase 6 |
| P3 | Testes com precisão mista | Fase 6 |
| P3 | Suite de fuzzing | Fase 6 |
| P3 | Testes de memória (leaks) | Fase 6 |

### 🔍 Relatório Completo
Ver `AUDIT_REPORT.md` para detalhes completos da auditoria.

---

## Notas da Auditoria de Exemplos (2026-03-06)

Realizada auditoria profunda e exaustiva de todos os 37 exemplos do repositório GoNeuron.

### 📊 Resumo da Auditoria

| Auditoria | Foco | Status |
|-----------|------|--------|
| Manual | Convergência e bugs críticos | ✅ 3 bugs corrigidos |
| ml-training-validator | Convergência de modelos | ✅ 37 exemplos analisados |
| golang-performance-architect | Performance e otimizações | ✅ Análise completa |
| neural-net-architect | Arquiteturas e APIs | ✅ Problemas identificados |
| security-architect | Segurança e estabilidade | ✅ Vulnerabilidades mapeadas |

### ✅ Bugs Críticos Corrigidos Durante Auditoria

| Bug | Exemplo | Causa | Status |
|-----|---------|-------|--------|
| Index out of range | `moe_math` | Experts não construídos corretamente | ✅ Corrigido |
| Index out of range | `time_series_forecast` | Cálculo incorreto em `Denormalize` | ✅ Corrigido |
| Index out of range | `sentiment_imdb` | `Flatten` antes de `SequenceUnroller` | ✅ Corrigido |

### 📋 Novos Itens Adicionados ao Roadmap

#### Prioridade 1 - Correções Imediatas (Fase 1)

| Item | Severidade | Descrição |
|------|------------|-----------|
| Corrigir divisões por zero | **CRÍTICA** | 5 exemplos stock_prediction* |
| Corrigir Softmax/CrossEntropy | **ALTA** | portuguese_gpt e portuguese_qa |
| Corrigir autoencoder | **ALTA** | Sigmoid → Tanh + normalização |
| Verificar I/O MNIST | **ALTA** | Adicionar verificação de erros |

#### Prioridade 2 - Performance (Fase 3)

| Item | Impacto |
|------|---------|
| Substituir loops Predict por PredictBatch | 8 exemplos |
| Usar TrainBatch em vez de Train() | 6 exemplos |
| Reduzir alocações em loops | 10+ exemplos |
| Otimizar exemplos para usar GPU | 20+ exemplos |

#### Prioridade 3 - Qualidade de Código (Fase 2/5)

| Item | Ocorrências |
|------|-------------|
| Substituir ioutil por os.ReadFile | 4 arquivos |
| Adicionar bounds checking | 3 exemplos |
| Normalizar logging | Todos os exemplos |
| Adicionar rand.Seed consistente | Todos os exemplos |

#### Prioridade 4 - Infraestrutura (Fase 5)

| Item | Descrição |
|------|-----------|
| Scripts de download automático | MNIST, CIFAR-10, etc. |
| Criar exemplo best_practices.go | Demonstrar otimizações |
| Documentar ForwardWithArena | Uso correto |
| Documentar GPU vs CPU | Guia de uso |

### 🎯 Resultados por Categoria

#### Convergência (ml-training-validator)
- **34 de 37 exemplos** convergem corretamente
- **3 exemplos** com problemas (2 corrigidos durante auditoria)
- **15 exemplos** requerem datasets externos

#### Performance (golang-performance-architect)
- Biblioteca com **zero-allocation pattern** implementado
- **Otimizações internas sofisticadas**: Arena, LightweightClone, WorkerPool
- **Problemas nos exemplos**: 8 com loops sequenciais, 6 sem batching

#### Arquitetura (neural-net-architect)
- **2 exemplos** com problema Softmax/CrossEntropy (duplicação)
- **1 exemplo** com Conv2D mal configurado
- **1 exemplo** com autoencoder muito simplificado

#### Segurança (security-architect)
- **5 exemplos** com divisões por zero (CRÍTICO)
- **6 exemplos** ignoram retornos de I/O binário
- **4 exemplos** usam funções depreciadas (ioutil)
- **Nota geral**: 6.5/10

### 📁 Relatórios Criados

1. `FINAL_AUDIT_REPORT.md` - Relatório consolidado
2. `AUDIT_REPORT_EXAMPLES.md` - Detalhes da auditoria manual
3. `ARCHITECTURE_AUDIT_REPORT.md` - Auditoria arquitetural
4. `.claude/agent-memory/*/EXAMPLES_AUDIT_REPORT.md` - Relatórios dos agentes

---

## Notas da Auditoria Completa dos Exemplos (2026-03-06)

Realizada auditoria profunda e exaustiva de todos os 37 exemplos do repositório GoNeuron.

### ✅ Status Geral da Auditoria
- **37 exemplos auditados**
- **34 exemplos convergindo corretamente**
- **3 bugs críticos corrigidos** durante a auditoria
- **5 agentes especializados** envolvidos (manual + 4 automatizados)

### 🐛 Bugs Críticos Corrigidos

| Bug | Exemplo | Causa | Solução |
|-----|---------|-------|---------|
| Index out of range | `moe_math` | Experts não construídos | Modificar `Sequential.Build` |
| Index out of range | `time_series_forecast` | Cálculo errado em `Denormalize` | Corrigir índice de feature |
| Index out of range | `sentiment_imdb` | `Flatten` antes de `SequenceUnroller` | Remover `Flatten()` |

### 📊 Resultados por Categoria

#### Convergência (ml-training-validator)
- ✅ **34/37 exemplos** convergem corretamente
- ⚠️ **autoencoder_synthetic**: Loss travado (necessita correção arquitetural)
- ⚠️ **sentiment**: Muito lento (dataset pequeno)

#### Performance (golang-performance-architect)
- ⚠️ 8 exemplos usam loops sequenciais (deveriam usar `PredictBatch`)
- ⚠️ 6 exemplos usam `Train()` em vez de `TrainBatch()`
- ⚠️ 10+ exemplos têm alocações excessivas em loops

#### Segurança (security-architect)
- 🔴 **5 exemplos** com divisões por zero (`stock_prediction*`)
- 🔴 **6 exemplos** ignoram retornos de I/O binário
- 🟡 **4 exemplos** usam `ioutil` (depreciado)

#### Arquitetura (neural-net-architect)
- 🔴 **portuguese_gpt**: Softmax + CrossEntropy (duplica log)
- 🔴 **portuguese_qa**: Softmax + CrossEntropy (duplica log)
- 🔴 **stock_prediction_cnn_lstm**: Conv2D com dims inválidas

### 📋 Novos Itens Adicionados ao Roadmap

#### Fase 1: Correções Críticas
- [x] Corrigir bug MoE layer ✅
- [x] Corrigir time_series_forecast ✅
- [x] Corrigir sentiment_imdb ✅
- [ ] Corrigir divisões por zero em 5 exemplos stock_prediction
- [ ] Corrigir Softmax/CrossEntropy em 2 exemplos GPT
- [ ] Corrigir autoencoder_synthetic (Sigmoid → Tanh)
- [ ] Adicionar verificação de erros em I/O binário
- [ ] Validar tamanho de dados lidos

#### Fase 2: Qualidade e Testes
- [ ] Adicionar golangci-lint com regras de segurança
- [ ] Adicionar testes de segurança automatizados

#### Fase 3: Performance
- [ ] Substituir loops Predict por PredictBatch (8 exemplos)
- [ ] Otimizar exemplos que usam Train() em vez de TrainBatch()
- [ ] Reduzir alocações em loops de dados

#### Fase 5: Developer Experience
- [ ] Criar guia de melhores práticas de performance
- [ ] Documentar uso correto de ForwardWithArena
- [ ] Documentar quando usar GPU vs CPU
- [ ] Criar exemplo best_practices.go
- [ ] Normalizar formato de logging entre exemplos
- [ ] Adicionar rand.Seed consistente em todos os exemplos
- [ ] Criar scripts de download automático para datasets
- [ ] Substituir ioutil por os.ReadFile em 4 arquivos

### 📈 Métricas da Auditoria

| Categoria | Nota | Status |
|-----------|------|--------|
| Convergência | 8/10 | ✅ Bom |
| Performance | 6/10 | ⚠️ Precisa melhorar |
| Arquitetura | 7/10 | ⚠️ Alguns problemas |
| Segurança | 6.5/10 | ⚠️ Vulnerabilidades fáceis de corrigir |
| **Geral** | **7.5/10** | ✅ Pronto para produção após correções |

### 📚 Relatórios Criados
1. `FINAL_AUDIT_REPORT.md` - Relatório consolidado
2. `AUDIT_REPORT_EXAMPLES.md` - Detalhes da auditoria manual
3. `ARCHITECTURE_AUDIT_REPORT.md` - Auditoria arquitetural
4. `.claude/agent-memory/*/EXAMPLES_AUDIT_REPORT.md` - Relatórios dos agentes

---

## Notas

- Cada sprint assume 2 semanas de trabalho
- Ajustes ao roadmap devem ser discutidos em issues
- Prioridades podem mudar baseadas em feedback da comunidade
- Métricas devem ser medidas e documentadas em cada fase
- Itens **AUDIT:** são findings de auditoria e devem ser priorizados
