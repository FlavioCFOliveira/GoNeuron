package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// Tokenizer handles basic word-level tokenization
type Tokenizer struct {
	wordToID map[string]int
	idToWord map[int]string
	vocab    []string
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		wordToID: make(map[string]int),
		idToWord: make(map[int]string),
		vocab:    []string{"[PAD]", "[UNK]", "[CTX]", "[Q]", "[A]", "[EOS]"},
	}
}

func (t *Tokenizer) BuildVocab(texts []string) {
	for _, text := range texts {
		words := t.tokenize(text)
		for _, word := range words {
			if _, ok := t.wordToID[word]; !ok {
				id := len(t.vocab)
				t.wordToID[word] = id
				t.idToWord[id] = word
				t.vocab = append(t.vocab, word)
			}
		}
	}
	// Add special tokens to maps
	for i, token := range t.vocab {
		t.wordToID[token] = i
		t.idToWord[i] = token
	}
}

func (t *Tokenizer) tokenize(text string) []string {
	text = strings.ToLower(text)
	text = strings.ReplaceAll(text, ".", " . ")
	text = strings.ReplaceAll(text, ",", " , ")
	text = strings.ReplaceAll(text, "?", " ? ")
	text = strings.ReplaceAll(text, "!", " ! ")
	return strings.Fields(text)
}

func (t *Tokenizer) Encode(text string, seqLen int) []int {
	words := t.tokenize(text)
	ids := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		if i < len(words) {
			id, ok := t.wordToID[words[i]]
			if !ok {
				ids[i] = t.wordToID["[UNK]"]
			} else {
				ids[i] = id
			}
		} else {
			ids[i] = t.wordToID["[PAD]"]
		}
	}
	return ids
}

func (t *Tokenizer) Decode(ids []int) string {
	var words []string
	for _, id := range ids {
		word, ok := t.idToWord[id]
		if !ok || word == "[PAD]" {
			continue
		}
		if word == "[EOS]" {
			break
		}
		words = append(words, word)
	}
	return strings.Join(words, " ")
}

func (t *Tokenizer) VocabSize() int {
	return len(t.vocab)
}

func main() {
	fmt.Println("=== GoNeuron: Portuguese Generative QA Example ===")

	// 1. Load Data
	content, err := os.ReadFile("examples/portuguese_qa/datasets/history.txt")
	if err != nil {
		log.Fatalf("Failed to read dataset: %v", err)
	}

	lines := strings.Split(string(content), "\n")
	var facts []string
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			facts = append(facts, line)
		}
	}

	// 2. Prepare QA Pairs
	// Simple rule-based QA pair generation from facts for demonstration
	var texts []string
	for _, fact := range facts {
		// Example: "A capital de Portugal é Lisboa."
		// Context: "A capital de Portugal é Lisboa."
		// Question: "Qual é a capital de Portugal?"
		// Answer: "Lisboa"

		// For simplicity in this small example, we'll just train on the facts themselves
		// formatted as sequences to predict.
		// In a real scenario, we'd have a proper (Context, Question, Answer) dataset.
		// Here we'll treat each fact as a sequence: [CTX] fact [EOS]
		// And we'll also add some synthetic questions.

		texts = append(texts, "[CTX] "+fact+" [EOS]")

		if strings.Contains(fact, "capital") {
			texts = append(texts, "[CTX] "+fact+" [Q] Qual é a capital de Portugal? [A] Lisboa [EOS]")
		}
		if strings.Contains(fact, "fundado") {
			texts = append(texts, "[CTX] "+fact+" [Q] Quem fundou Portugal? [A] D. Afonso Henriques [EOS]")
		}
		if strings.Contains(fact, "Vasco da Gama") {
			texts = append(texts, "[CTX] "+fact+" [Q] O que descobriu Vasco da Gama? [A] O caminho marítimo para a Índia [EOS]")
		}
	}

	tokenizer := NewTokenizer()
	tokenizer.BuildVocab(texts)
	vocabSize := tokenizer.VocabSize()
	seqLen := 32
	fmt.Printf("Vocab Size: %d, Sequence Length: %d\n", vocabSize, seqLen)

	// 3. Prepare Training Data
	var trainX [][]float32
	var trainY [][]float32

	// Repeat data to help convergence on small dataset
	for r := 0; r < 20; r++ {
		for _, text := range texts {
			ids := tokenizer.Encode(text, seqLen+1)

			// X is ids[0:seqLen], Y is ids[1:seqLen+1] (next token prediction)
			x := make([]float32, seqLen)
			y := make([]float32, seqLen*vocabSize) // One-hot for CrossEntropy

			for i := 0; i < seqLen; i++ {
				x[i] = float32(ids[i])
				// One-hot for target
				targetID := ids[i+1]
				y[i*vocabSize+targetID] = 1.0
			}

			trainX = append(trainX, x)
			trainY = append(trainY, y)
		}
	}

	// 4. Build Architecture
	dim := 64
	numHeads := 4
	ffDim := 128

	// Check if Metal is available
	var device layer.Device = &layer.CPUDevice{}
	metal := &layer.MetalDevice{}
	if metal.IsAvailable() {
		fmt.Println("Using Metal acceleration")
		device = metal
	} else {
		fmt.Println("Using CPU")
	}

	layers := []layer.Layer{
		layer.NewEmbedding(vocabSize, dim),
		layer.NewPositionalEncoding(seqLen, dim),
		layer.NewTransformerBlock(dim, numHeads, seqLen, ffDim, true), // Causal
		layer.NewSequenceUnroller(layer.NewDense(dim, vocabSize, activations.LogSoftmax{}), seqLen, true),
	}

	model := net.New(layers, loss.NLLLoss{}, opt.NewAdam(0.001))
	model.SetDevice(device)

	// 5. Training
	epochs := 100
	fmt.Printf("Training for %d epochs with %d samples...\n", epochs, len(trainX))

	start := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)
		for i := 0; i < len(trainX); i++ {
			l := model.Train(trainX[i], trainY[i])
			totalLoss += l
		}
		if epoch%10 == 0 || epoch == epochs-1 {
			fmt.Printf("Epoch %d, Loss: %.6f\n", epoch, totalLoss/float32(len(trainX)))
		}
	}
	fmt.Printf("Training completed in %v\n", time.Since(start))

	// 6. QA Functionality
	generate := func(context, question string) string {
		prompt := "[CTX] " + context + " [Q] " + question + " [A]"
		ids := tokenizer.Encode(prompt, seqLen)

		resultIds := make([]int, seqLen)
		copy(resultIds, ids)

		// Find first PAD to start generating
		startIdx := 0
		tokens := tokenizer.tokenize(prompt)
		startIdx = len(tokens)
		if startIdx >= seqLen {
			startIdx = seqLen - 1
		}

		for t := startIdx; t < seqLen; t++ {
			// Forward pass
			input := make([]float32, seqLen)
			for i := 0; i < seqLen; i++ {
				input[i] = float32(resultIds[i])
			}

			output := model.Forward(input)

			// Get probabilities for the position t-1 to predict token at t
			stepOffset := (t - 1) * vocabSize
			stepProbs := output[stepOffset : stepOffset+vocabSize]

			// Simple greedy search (skip [PAD] and [UNK] during generation if possible)
			maxID := 0
			maxProb := float32(-1e10)
			for i, p := range stepProbs {
				if i == tokenizer.wordToID["[PAD]"] || i == tokenizer.wordToID["[UNK]"] {
					continue
				}
				if p > maxProb {
					maxProb = p
					maxID = i
				}
			}

			resultIds[t] = maxID
			if maxID == tokenizer.wordToID["[EOS]"] {
				break
			}
		}

		return tokenizer.Decode(resultIds)
	}

	// 7. Demonstration
	fmt.Println("\n--- Demonstration ---")
	testContext := "A capital de Portugal é Lisboa."
	testQuestion := "Qual é a capital de Portugal?"
	fmt.Printf("Contexto: %s\n", testContext)
	fmt.Printf("Pergunta: %s\n", testQuestion)

	answer := generate(testContext, testQuestion)
	fmt.Printf("Resposta Gerada: %s\n", answer)

	// Interactive QA
	fileInfo, _ := os.Stdin.Stat()
	isTerminal := (fileInfo.Mode() & os.ModeCharDevice) != 0

	if !isTerminal {
		fmt.Println("\nDetetada execução não interativa. Exemplo concluído.")
		return
	}

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("\nQueres testar? (s/n)")
	choice, _ := reader.ReadString('\n')
	if strings.TrimSpace(choice) == "s" {
		for {
			fmt.Print("\nContexto: ")
			ctx, _ := reader.ReadString('\n')
			ctx = strings.TrimSpace(ctx)
			if ctx == "" { break }

			fmt.Print("Pergunta: ")
			q, _ := reader.ReadString('\n')
			q = strings.TrimSpace(q)
			if q == "" { break }

			ans := generate(ctx, q)
			fmt.Printf("Resposta: %s\n", ans)
		}
	}
}
