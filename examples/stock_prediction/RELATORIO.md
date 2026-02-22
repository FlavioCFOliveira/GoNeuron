# Relatório: Stock Prediction com LSTM no GoNeuron

Este relatório apresenta os resultados da implementação de um modelo de previsão de preços de ações (Google - GOOG) utilizando a biblioteca **GoNeuron** com uma arquitetura **LSTM** (Long Short-Term Memory).

## 1. Arquitetura do Modelo
O modelo foi construído com a seguinte estrutura:
- **Camada de Entrada**: Janela deslizante de 10 dias de preços de fechamento ('Close') normalizados.
- **Camada LSTM**: 32 unidades ocultas, utilizando o padrão de **Zero-Allocation** da biblioteca para máxima performance.
- **Camada Dense**: 1 neurônio de saída com ativação Linear para regressão do preço do dia seguinte.
- **Otimizador**: SGD (Stochastic Gradient Descent) com taxa de aprendizado de 0.01.
- **Função de Perda**: MSE (Mean Squared Error).

## 2. Metodologia de Dados
- **Dataset**: Dados históricos da Alphabet Inc. (GOOG).
- **Pré-processamento**: Normalização Min-Max para escalar os preços entre [0, 1].
- **Divisão**: 80% para treinamento, 20% para teste.
- **Sequenciamento**: Uso de janelas de 10 dias (`lookback=10`) para prever o 11º dia.

## 3. Resultados do Treinamento
O modelo demonstrou convergência clara durante as 100 épocas de treino:

| Métrica | Valor Inicial (Época 1) | Valor Final (Época 100) |
|---------|-------------------------|-------------------------|
| Perda Média (Loss) | 0.159561 | 0.039483 |
| Erro Quadrático Médio (Test RMSE) | - | $32.20 |

## 4. Performance e Benchmarking
A biblioteca GoNeuron mostrou-se extremamente eficiente no processamento de sequências LSTM:

- **Tempo Total de Treino (100 épocas)**: ~42.42 ms
- **Tempo Médio por Época**: ~424.19 µs
- **Uso de Memória (Alloc)**: 1650 KB
- **Eficiência**: O uso de buffers pré-alocados permitiu um tempo de resposta na ordem de microssegundos por amostra.

## 5. Avaliação Final (Conjunto de Teste)
O modelo obteve uma precisão notável para uma configuração simplificada:
- **MAPE (Mean Absolute Percentage Error)**: 3.14%
- **RMSE Final**: $32.20

### Amostras de Previsão:
| Real | Previsto | Erro ($) | Erro (%) |
|------|----------|----------|----------|
| $1017.11 | $986.78 | -$30.33 | -2.98% |
| $1016.64 | $987.52 | -$29.12 | -2.86% |
| $1025.50 | $988.86 | -$36.64 | -3.57% |

## 6. Conclusão
O exemplo demonstra com sucesso a capacidade da GoNeuron em lidar com séries temporais complexas usando LSTMs. A implementação de **Zero-Allocation** garante que, mesmo em Go, a rede neuronal opere com latência mínima, tornando-a ideal para aplicações de alta performance ou sistemas embarcados onde o controle de memória é crítico.
