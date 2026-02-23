package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
)

// SimpleTokenizer simulates a basic subword tokenizer for Portuguese.
type SimpleTokenizer struct {
	vocab  map[string]int
	maxLen int
}

func NewSimpleTokenizer(maxLen int) *SimpleTokenizer {
	vocab := make(map[string]int)
	vocab["[PAD]"] = 0
	vocab["[CLS]"] = 1
	vocab["[SEP]"] = 2
	vocab["[UNK]"] = 3

	// Some common Portuguese words for the example.
	commonWords := []string{
		"o", "a", "do", "da", "em", "um", "uma", "que", "é", "de",
		"portugues", "português", "exemplo", "modelo", "aprendizado",
		"máquina", "neural", "rede", "sentença", "texto", "lisboa", "porto",
		"público", "tribunal", "justiça", "supremo", "jornal", "portugal",
	}

	for i, word := range commonWords {
		vocab[word] = i + 4
	}

	return &SimpleTokenizer{vocab: vocab, maxLen: maxLen}
}

func (t *SimpleTokenizer) Tokenize(text string) []float32 {
	text = strings.ToLower(text)
	words := strings.Fields(text)

	tokens := make([]float32, t.maxLen)
	tokens[0] = 1.0 // [CLS]

	for i := 0; i < t.maxLen-2 && i < len(words); i++ {
		id, ok := t.vocab[words[i]]
		if !ok {
			id = 3 // [UNK]
		}
		tokens[i+1] = float32(id)
	}

	// Add [SEP] if there's space.
	idx := len(words) + 1
	if idx < t.maxLen {
		tokens[idx] = 2.0
	}

	return tokens
}

// SimulatedDataset returns sentences from Portuguese sources like "Público" or "Supremo Tribunal de Justiça".
func SimulatedDataset() []string {
	return []string{
		"O Supremo Tribunal de Justiça decidiu favoravelmente.",
		"O jornal Público noticiou a nova rede neural.",
		"Exemplo de processamento de linguagem natural em português.",
		"Aprendizado de máquina com GoNeuron em Lisboa.",
		"Justiça portuguesa utiliza modelos avançados.",
	}
}

func main() {
	fmt.Println("=== GoNeuron: Portuguese Embeddings Architecture (High-Level API) ===")

	// BERT-like dimensions
	const (
		vocabSize    = 1000
		embeddingDim = 768
		seqLen       = 32
		numHeads     = 12
		ffDim        = 3072
	)

	// 1. Initialize Device
	device := layer.GetDefaultDevice()
	fmt.Printf("Device initialized: %v\n", device.Type())

	// 2. Build Architecture using High-Level API
	model := goneuron.NewSequential(
		goneuron.Embedding(vocabSize, embeddingDim),
		goneuron.PositionalEncoding(seqLen, embeddingDim),
		goneuron.TransformerBlock(embeddingDim, numHeads, seqLen, ffDim, false),
		goneuron.CLSPooling(seqLen, embeddingDim),
	)

	model.SetDevice(device)

	// 3. Prepare Simulated Data
	tokenizer := NewSimpleTokenizer(seqLen)
	sentences := SimulatedDataset()

	fmt.Println("\nInput Sentences (PT-PT Sources):")
	for i, sentence := range sentences {
		fmt.Printf(" [%d] %s\n", i+1, sentence)
	}

	// 4. Generate Embedding for a sample sentence
	sampleSentence := "O Supremo Tribunal de Justiça de Portugal."
	fmt.Printf("\nTarget: \"%s\"\n", sampleSentence)

	tokens := tokenizer.Tokenize(sampleSentence)

	start := time.Now()
	embedding := model.Predict(tokens)
	elapsed := time.Since(start)

	// 5. Results
	fmt.Printf("\nGeneration completed in %v", elapsed)
	fmt.Printf("\nVector Dimensions: %d", len(embedding))
	fmt.Printf("\nFirst 10 values: %v\n", embedding[:10])

	fmt.Println("\n=== Performance Metrics ===")
	fmt.Printf("Memory Layout: Contiguous float32\n")
	fmt.Printf("Zero-Allocation Pooling: Yes\n")
	fmt.Printf("Hardware Acceleration: %v\n", device.Type() == layer.GPU)

	model.Summary()
}
