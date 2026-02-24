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
 * TUTORIAL: Carregar Modelo (.gob) e Converter para GGUF
 *
 * Muitos modelos são treinados e salvos no formato nativo da biblioteca (.gob).
 * Este exemplo mostra como transformar esses modelos num formato de interoperabilidade (GGUF).
 *
 * O que será demonstrado:
 * 1. Simulação de um modelo pré-treinado salvo em disco.
 * 2. Carregamento dinâmico da arquitetura e pesos usando net.Load.
 * 3. Conversão "on-the-fly" para o ecossistema GGUF.
 */

func main() {
	fmt.Println("=== [TUTORIAL] Carregamento e Conversão para GGUF ===")

	origem := "checkpoint_modelo.gob"
	destino := "modelo_final_interoperavel.gguf"

	// --- PASSO 1: PREPARAÇÃO (Simulando um modelo existente) ---
	fmt.Println("\n[1/3] Criando um modelo pré-treinado para exemplo...")
	criarModeloExemplo(origem)

	// --- PASSO 2: CARREGAMENTO ---
	// O net.Load é poderoso pois reconstrói automaticamente os tipos de camadas
	// (Dense, Conv2D, etc.) que foram registados via gob.
	fmt.Printf("[2/3] A carregar modelo nativo de '%s'...\n", origem)
	modelo, perda, err := net.Load(origem)
	if err != nil {
		log.Fatalf("Erro crítico ao carregar: %v", err)
	}

	fmt.Printf("✓ Modelo restaurado com sucesso!")
	fmt.Printf("  Arquitetura: %d camadas | Perda: %T\n", len(modelo.Layers()), perda)

	// --- PASSO 3: EXPORTAÇÃO GGUF ---
	// Agora que o objeto 'modelo' está na memória, podemos guardá-lo em GGUF.
	// Isto permite que um modelo treinado em Go seja levado para C++ ou Python (via llama.cpp).
	fmt.Printf("\n[3/3] A converter para formato GGUF em '%s'...\n", destino)
	if err := modelo.SaveGGUF(destino); err != nil {
		log.Fatalf("Erro na conversão: %v", err)
	}

	fmt.Println("✓ Conversão concluída.")

	// --- RESULTADOS ---
	fmt.Println("\nComparação de formatos:")
	verTamanho(origem)
	verTamanho(destino)

	fmt.Println("\nNota: O ficheiro GGUF contém metadados sobre a arquitetura 'goneuron',")
	fmt.Println("permitindo que loaders universais identifiquem a origem do modelo.")
}

// criarModeloExemplo gera um ficheiro .gob para servir de base ao tutorial.
func criarModeloExemplo(path string) {
	// Uma arquitetura genérica de classificação
	camadas := []layer.Layer{
		layer.NewDense(128, 64, activations.ReLU{}),
		layer.NewDense(64, 10, activations.Softmax{}),
	}
	rede := net.New(camadas, loss.CrossEntropy{}, opt.NewAdam(0.001))

	if err := rede.Save(path); err != nil {
		log.Fatalf("Falha ao criar modelo de exemplo: %v", err)
	}
}

func verTamanho(path string) {
	s, err := os.Stat(path)
	if err != nil {
		return
	}
	fmt.Printf("  - %-30s: %d bytes\n", path, s.Size())
}
