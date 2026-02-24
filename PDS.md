# Plano de Desenvolvimento de Software (PDS) - GoNeuron API Simplificada

## 1. Visão Geral
O projeto visa transformar o GoNeuron de uma biblioteca modular de baixo nível numa framework de Deep Learning de alta produtividade para Go, mantendo a performance crítica. O foco é a "Human API" (API para humanos), minimizando o boilerplate e a complexidade de gestão de memória para o utilizador final.

## 2. Objetivos
- **Redução de Código**: Diminuir em pelo menos 60% as linhas necessárias para definir e treinar um modelo.
- **Abstração de Gestão de Memória**: Ocultar a gestão de buffers e estados intermédios.
- **Unificação de Interfaces**: Criar um ponto de entrada único (`goneuron`) para todas as funcionalidades comuns.
- **Automação de Treino**: Implementar loops de treino robustos com suporte a callbacks.

## 3. Arquitetura da "Human API"

### 3.1. Camada de Abstração (`Sequential`)
A estrutura `Sequential` atua como um orchestrator que:
- Encapsula a lógica de `Network`.
- Gere automaticamente a compatibilidade de dimensões entre camadas.
- Propaga o estado de `Training/Inference` globalmente (ex: afetando Dropout e BatchNorm).

### 3.2. Fachada `goneuron`
Um pacote que re-exporta componentes internos para facilitar o acesso:
- **Factory Functions**: `Dense()`, `Conv2D()`, `LSTM()`, etc.
- **Constantes de Ativação**: `ReLU`, `Tanh`, `Softmax`.
- **Utilitários de Treino**: Callbacks e Schedulers integrados.

## 4. Estado Atual (Concluído)
- [x] Implementação do modelo `Sequential`.
- [x] Criação do pacote de fachada `goneuron`.
- [x] Implementação de métodos `Compile`, `Fit`, `Predict` e `Evaluate`.
- [x] Conversão de 100% dos exemplos existentes para a nova API.
- [x] Suporte a Hardware Acceleration (Metal/MPS) transparente via `SetDevice`.
- [x] Sistema de Callbacks (Logger, Checkpoint, EarlyStopping).
- [x] Implementação de camada Sparse MoE (Mixture of Experts) com Top-K routing.
- [x] **Zero-Allocation Contiguous Memory**: Implementação de buffers globais contíguos para parâmetros e gradientes.
- [x] **Parallel Training Architecture**: Implementação de `Worker Pool` e `LightweightClone` para treino paralelo eficiente sem alocações repetitivas.
- [x] **Arena-based State Saving**: Eliminação de alocações durante o forward/backward pass através de um sistema de offsets em memória contígua (Arena).

## 5. Roadmap de Desenvolvimento Futuro

### Fase 5: Otimização de Memória e GPU (Concluído)
- [x] **Contiguous Buffers**: Unificação de pesos e bias em slices contínuos.
- [x] **Activation Arena**: Sistema de offsets para buffers de ativação (`savedInputs`), atingindo 0 alocações no loop de treino sequencial.
- [x] **Worker Pool**: Treino paralelo com trabalhadores pré-alocados para minimizar overhead de clonagem.

### Fase 6: Otimização Avançada e Estabilidade (Concluído)
- [x] **Kernel Fusing**: Combinar operações de MatMul + Bias + Activation em kernels Metal únicos.
- [x] **GELU Support**: Implementação de ativação GELU acelerada em GPU.
- [x] **Stability Fixes**: Resolução de erros de CGO (`semasleep`) através de gestão de threads (`LockOSThread`).
- [x] **Metal Persistent Buffers**: Ciclo de vida completo para buffers de gradientes no GPU.
- [x] **GGUF Serialization**: Suporte nativo para exportação de modelos em formato GGUF v3 (F32/F16).

### Fase 7: Próximos Passos
- [ ] **Asynchronous Transfers**: Otimizar a transferência de dados entre CPU e GPU.
- [ ] **Functional API**: Permitir arquiteturas não lineares (ResNet, Inception) através de uma API de grafos.
- [ ] **Lazy Shape Inference**: Suporte a inferência de formas em tempo de execução para todas as camadas.

### Fase 8: Validação e Diagnóstico
- [x] **Data Loaders**: Implementação de interface para carregar datasets CSV e IDX (MNIST).
- [x] **Metrics Logging**: Sistema de logging e callbacks para monitorização de treino.
- [ ] **Visualização de Métricas**: Exportação de métricas para JSON/CSV.

### Fase 9: Expansão de Arquiteturas
- [x] **Transformer Blocks**: Blocos modulares com RoPE e SwiGLU.
- [ ] **Attention Mecanisms**: Suporte a Cross-Attention e Sliding Window.

## 6. Exemplo de Utilização (Padrão PDS)
```go
model := goneuron.NewSequential(
    goneuron.Conv2D(1, 32, 3, 1, 1, goneuron.ReLU),
    goneuron.MaxPool2D(32, 2, 2, 0),
    goneuron.Flatten(),
    goneuron.Dense(6272, 10, goneuron.Softmax),
)

model.Compile(goneuron.Adam(0.001), goneuron.CrossEntropy)
model.Fit(xTrain, yTrain, 10, 32, goneuron.Logger(1))
```

## 7. Conclusão
A modernização da API coloca o GoNeuron no mesmo nível de usabilidade de frameworks líderes de mercado, tornando o ecossistema Go mais apelativo para investigadores e engenheiros de Machine Learning.
