package main

import (
	"fmt"
	"log"
	"os"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

/**
 * TUTORIAL: Treino e Exportação de Modelos para GGUF v3
 *
 * Este exemplo demonstra como criar uma rede neural, treiná-la para resolver o problema XOR,
 * e exportar o resultado para o formato GGUF v3 (padrão utilizado em LLMs como Llama.cpp).
 *
 * O que será demonstrado:
 * 1. Construção de uma arquitetura MLP (Multi-Layer Perceptron).
 * 2. Processo de treinamento automatizado (Fit).
 * 3. Exportação em Precisão Total (Float32).
 * 4. Exportação em Precisão Reduzida (Float16) para economia de espaço.
 */

func main() {
	fmt.Println("=== [TUTORIAL] Treino e Exportação GGUF v3 ===")

	// --- PASSO 1: DEFINIÇÃO DA ARQUITETURA ---
	// Criamos uma rede com 2 entradas, uma camada oculta de 8 neurónios e 1 saída.
	// Usamos Tanh na camada oculta para evitar o problema de "neurónios mortos" do ReLU no XOR.
	layers := []layer.Layer{
		layer.NewDense(2, 8, activations.Tanh{}),    // Camada de entrada -> oculta
		layer.NewDense(8, 1, activations.Sigmoid{}), // Camada oculta -> saída (0 a 1)
	}

	// Inicializamos a rede com o otimizador Adam e perda MSE (Mean Squared Error).
	model := net.New(layers, loss.MSE{}, opt.NewAdam(0.01))

	// --- PASSO 2: DADOS DE TREINAMENTO ---
	// Definimos a tabela verdade do XOR:
	// 0 XOR 0 = 0 | 0 XOR 1 = 1 | 1 XOR 0 = 1 | 1 XOR 1 = 0
	trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainY := [][]float32{{0}, {1}, {1}, {0}}

	// --- PASSO 3: TREINAMENTO ---
	fmt.Println("\n[1/4] Iniciando treinamento (500 épocas)...")
	model.Fit(trainX, trainY, 500, 4)
	fmt.Println("✓ Treinamento concluído.")

	// --- PASSO 4: EXPORTAÇÃO F32 (Full Precision) ---
	// O método SaveGGUF exporta os pesos em Float32 (4 bytes por peso).
	// É o formato ideal para máxima precisão.
	f32Path := "xor_model_f32.gguf"
	fmt.Printf("\n[2/4] Exportando para %s (Float32)...\n", f32Path)
	if err := model.SaveGGUF(f32Path); err != nil {
		log.Fatalf("Falha na exportação F32: %v", err)
	}
	fmt.Println("✓ Exportação F32 concluída com sucesso.")

	// --- PASSO 5: EXPORTAÇÃO F16 (Half Precision) ---
	// O método SaveGGUFExt permite especificar o formato do tensor.
	// GGMLTypeF16 converte pesos de 32-bits para 16-bits, reduzindo o ficheiro para metade
	// do tamanho com perda mínima de precisão.
	f16Path := "xor_model_f16.gguf"
	fmt.Printf("\n[3/4] Exportando para %s (Float16)...\n", f16Path)
	if err := model.SaveGGUFExt(f16Path, net.GGMLTypeF16); err != nil {
		log.Fatalf("Falha na exportação F16: %v", err)
	}
	fmt.Println("✓ Exportação F16 concluída com sucesso.")

	// --- PASSO 6: COMPARAÇÃO E RESULTADOS ---
	fmt.Println("\n[4/4] Resumo dos ficheiros gerados:")
	showStats(f32Path)
	showStats(f16Path)

	fmt.Println("\nInstrução: Os ficheiros .gguf podem agora ser integrados em backends C++ (GGML)")
	fmt.Println("ou utilizados para inferência portátil noutras linguagens.")
}

// showStats imprime o tamanho do ficheiro de forma legível.
func showStats(path string) {
	info, err := os.Stat(path)
	if err != nil {
		fmt.Printf("! Erro ao aceder %s: %v\n", path, err)
		return
	}
	fmt.Printf("  - %s: %d bytes\n", path, info.Size())
}
