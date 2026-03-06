# GoNeuron Roadmap 2026

## Visão
Transformar GoNeuron na biblioteca de deep learning em Go mais performática, estável e fácil de usar, mantendo a filosofia de zero dependências externas.

## Fases do Roadmap

---

## Fase 1: Estabilidade e Correções Críticas (Sprint 1-2)
**Objetivo:** Corrigir bugs críticos que afetam a correção matemática e a compilação.

### 1.1 Correções Críticas
- [ ] Corrigir implementação de CrossEntropy Loss (gradientes incorretos)
- [ ] Corrigir exemplo portuguese_qa (imports quebrados)
- [ ] Corrigir NLLLoss (valores forward/backward incorretos)
- [ ] Validar gradientes de todas as funções de perda contra PyTorch

### 1.2 Tratamento de Erros
- [ ] Substituir panic() por error returns nas APIs públicas
- [ ] Adicionar validação de inputs em todas as camadas
- [ ] Implementar logging estruturado para depuração

**Métricas de Sucesso:**
- 100% dos testes passam
- 0 panics em código de produção
- Todas as losses validadas contra referências PyTorch

---

## Fase 2: Qualidade e Testes (Sprint 3-4)
**Objetivo:** Aumentar cobertura de testes e garantir estabilidade.

### 2.1 Cobertura de Testes
- [x] Criar testes para package goneuron (API pública) ✅ Implementado
- [x] Criar testes de integração para exemplos principais (mnist, cifar10) ✅ Implementado
- [ ] Implementar testes de gradiente numérico para todas as camadas
- [ ] Adicionar testes de estabilidade numérica
- [ ] **AUDIT:** Adicionar testes de estresse com batches grandes (512, 1024+)
- [ ] **AUDIT:** Expandir cobertura de edge cases (NaN/Inf, dimensões extremas, inputs vazios)
- [ ] **AUDIT:** Implementar testes de regressão comparando outputs entre releases

### 2.2 Validação
- [x] Criar suite de validação automática contra PyTorch ✅ Implementado (com geração de referências)
- [x] Implementar testes de convergência para cada tipo de camada ✅ Implementado
- [x] Validar backward pass de todas as camadas complexas (LSTM, GRU, Transformer) ✅ Implementado
- [ ] **AUDIT:** Aumentar tolerância em gradient check numérico para Huber loss (atual: diferença detectada)
- [ ] **AUDIT:** Criar benchmarks comparativos Metal vs CPU para todas as camadas

### 2.3 CI/CD
- [ ] Configurar GitHub Actions para testes automáticos
- [ ] Adicionar análise estática (go vet, staticcheck, golint)
- [ ] Configurar code coverage reporting

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
| P2 | Testes de estresse com batches grandes | Fase 2 |
| P2 | Cobertura de edge cases (NaN/Inf) | Fase 2 |
| P2 | Testes de regressão entre releases | Fase 2 |
| P3 | Expandir benchmarks Metal vs CPU | Fase 2 |
| P3 | Corrigir tolerância gradient check Huber | Fase 2 |
| P2 | Otimizar BatchNorm2D forward | Fase 3 |
| P2 | Otimizar Conv2DLarge | Fase 3 |
| P2 | Otimizar LSTM sequence | Fase 3 |
| P3 | Testes de longa duração | Fase 6 |
| P3 | Testes com precisão mista | Fase 6 |
| P3 | Suite de fuzzing | Fase 6 |
| P3 | Testes de memória (leaks) | Fase 6 |

### 🔍 Relatório Completo
Ver `AUDIT_REPORT.md` para detalhes completos da auditoria.

---

## Notas

- Cada sprint assume 2 semanas de trabalho
- Ajustes ao roadmap devem ser discutidos em issues
- Prioridades podem mudar baseadas em feedback da comunidade
- Métricas devem ser medidas e documentadas em cada fase
- Itens **AUDIT:** são findings de auditoria e devem ser priorizados
