package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

// Este exemplo demonstra um modelo Mixture of Experts (MoE).
// A rede tem 4 especialistas (experts) e ativa apenas os 2 melhores (k=2).
// O objetivo é aprender uma função que muda de comportamento dependendo do input.

func main() {
	fmt.Println("=== Mixture of Experts (MoE) Example ===")

	// 1. Gerar Dados Sintéticos
	// Se x < 0.5 -> y = x * 2 (Linear)
	// Se x >= 0.5 -> y = sin(x * 10) (Oscilatório)
	numSamples := 1000
	xTrain := make([][]float32, numSamples)
	yTrain := make([][]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		x := rand.Float32()
		xTrain[i] = []float32{x}

		var y float32
		if x < 0.5 {
			y = x * 2
		} else {
			y = float32(math.Sin(float64(x * 10)))
		}
		yTrain[i] = []float32{y}
	}

	// 2. Definir Arquitetura MoE
	// Input (1) -> MoE(4 Experts, k=2) -> Output (1)
	model := goneuron.NewSequential(
		goneuron.Dense(1, 16, goneuron.LeakyReLU(0.1)),
		goneuron.MoE(16, 16, 4, 2), // 16 in, 16 out, 4 experts, top-2 ativos
		goneuron.Dense(16, 1, goneuron.Linear),
	)

	// 3. Compile
	model.Compile(goneuron.Adam(0.01), goneuron.MSE)

	// 4. Sumário
	model.Summary()

	// 5. Treino
	fmt.Println("Treinando modelo MoE...")
	model.Fit(xTrain, yTrain, 500, 16, goneuron.Logger(50))

	// 6. Inferência e Teste
	fmt.Println("\n=== Teste de Inferência ===")
	testCases := []float32{0.1, 0.3, 0.6, 0.8}
	for _, tc := range testCases {
		pred := model.Predict([]float32{tc})

		var expected float32
		if tc < 0.5 {
			expected = tc * 2
		} else {
			expected = float32(math.Sin(float64(tc * 10)))
		}

		fmt.Printf("Input: %.2f | Pred: %.4f | Esperado: %.4f | Erro: %.4f\n",
			tc, pred[0], expected, float32(math.Abs(float64(pred[0]-expected))))
	}

	// 7. Guardar o Modelo
	err := model.Save("moe_model.gob")
	if err == nil {
		fmt.Println("\nModelo MoE guardado com sucesso!")
	}
}
