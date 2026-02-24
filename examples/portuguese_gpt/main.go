package main

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

const (
	seqLen       = 12
	embeddingDim = 32
	numHeads     = 4
	ffDim        = 64
	epochs       = 150
	learningRate = 0.002
	modelFile    = "/Users/flaviocfo/dev/github.com/FlavioCFOliveira/GoNeuron/examples/portuguese_gpt/portuguese_gpt.bin"
)

// SimpleTokenizer for Portuguese with encoding and decoding.
type SimpleTokenizer struct {
	vocab    map[string]int
	revVocab map[int]string
	maxLen   int
}

func NewSimpleTokenizer(corpus []string, maxLen int) *SimpleTokenizer {
	vocab := make(map[string]int)
	revVocab := make(map[int]string)

	vocab["[PAD]"] = 0
	revVocab[0] = "[PAD]"
	vocab["[UNK]"] = 1
	revVocab[1] = "[UNK]"
	vocab["[BOS]"] = 2
	revVocab[2] = "[BOS]"
	vocab["[EOS]"] = 3
	revVocab[3] = "[EOS]"

	count := 4
	for _, line := range corpus {
		words := tokenize(line)
		for _, word := range words {
			if _, ok := vocab[word]; !ok {
				vocab[word] = count
				revVocab[count] = word
				count++
			}
		}
	}

	return &SimpleTokenizer{
		vocab:    vocab,
		revVocab: revVocab,
		maxLen:   maxLen,
	}
}

func tokenize(text string) []string {
	// Simple punctuation handling for PT-PT
	puncs := []string{".", ",", "!", "?", ";", ":", "(", ")", "\"", "-"}
	processed := strings.ToLower(text)
	for _, p := range puncs {
		processed = strings.ReplaceAll(processed, p, " "+p+" ")
	}
	return strings.Fields(processed)
}

func (t *SimpleTokenizer) Encode(text string) []float32 {
	words := tokenize(text)
	tokens := make([]float32, t.maxLen)
	for i := range tokens {
		tokens[i] = 0 // [PAD]
	}

	tokens[0] = 2.0 // [BOS]
	for i := 0; i < len(words) && i < t.maxLen-2; i++ {
		id, ok := t.vocab[words[i]]
		if !ok {
			id = 1 // [UNK]
		}
		tokens[i+1] = float32(id)
	}
	if len(words)+1 < t.maxLen {
		tokens[len(words)+1] = 3.0 // [EOS]
	}

	return tokens
}

func (t *SimpleTokenizer) Decode(tokens []float32) string {
	var words []string
	for _, tok := range tokens {
		id := int(tok)
		if id == 0 || id == 2 {
			continue
		}
		if id == 3 {
			break
		}
		word, ok := t.revVocab[id]
		if !ok {
			word = "[UNK]"
		}
		words = append(words, word)
	}
	return strings.Join(words, " ")
}

func (t *SimpleTokenizer) VocabSize() int {
	return len(t.vocab)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("=== GoNeuron: Portuguese GPT (Mini-GPT) Professional Demo ===")

	// 1. Load Dataset
	corpusPath := "/Users/flaviocfo/dev/github.com/FlavioCFOliveira/GoNeuron/examples/portuguese_gpt/datasets/pt_pt_corpus.txt"
	cleanLines := loadCorpus(corpusPath)

	// 2. Initialize Tokenizer
	tokenizer := NewSimpleTokenizer(cleanLines, seqLen)
	vocabSize := tokenizer.VocabSize()
	fmt.Printf("Vocab Size: %d | Dataset size: %d lines\n", vocabSize, len(cleanLines))

	// 3. Model Architecture
	device := layer.GetDefaultDevice()
	fmt.Printf("Using Device: %v\n", device.Type())

	model := buildModel(vocabSize, device)

	// 4. Persistence: Try to load existing model
	if _, err := os.Stat(modelFile); err == nil {
		fmt.Printf("Loading existing model from %s...\n", modelFile)
		if err := loadModel(model, modelFile); err != nil {
			fmt.Printf("Error loading model: %v. Starting fresh training.\n", err)
			trainModel(model, tokenizer, cleanLines)
			saveModel(model, modelFile)
		} else {
			fmt.Println("Model loaded successfully.")
		}
	} else {
		fmt.Println("No pre-trained model found. Starting training...")
		trainModel(model, tokenizer, cleanLines)
		saveModel(model, modelFile)
	}

	// 5. Evaluation
	fmt.Println("\nRunning Evaluation...")
	acc, avgLoss := evaluate(model, tokenizer, cleanLines)
	fmt.Printf("Final Evaluation - Loss: %.4f | Accuracy: %.2f%%\n", avgLoss, acc*100)

	// 6. Interactive Generation
	interactiveLoop(model, tokenizer)

	// 7. Performance Report
	generateReport(model, device)
}

func loadCorpus(path string) []string {
	content, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("Error reading corpus: %v\n", err)
		os.Exit(1)
	}
	lines := strings.Split(string(content), "\n")
	var cleanLines []string
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			cleanLines = append(cleanLines, line)
		}
	}
	return cleanLines
}

func buildModel(vocabSize int, device layer.Device) *goneuron.Model {
	model := goneuron.NewSequential(
		goneuron.Embedding(vocabSize, embeddingDim),
		goneuron.PositionalEncoding(seqLen),
		goneuron.TransformerBlock(numHeads, seqLen, ffDim, true),
		goneuron.SequenceUnroller(
			goneuron.Dense(vocabSize, goneuron.Softmax),
			seqLen,
			true,
		),
	)

	model.Compile(goneuron.Adam(learningRate), goneuron.CrossEntropy)
	model.SetDevice(device)
	return model
}

func trainModel(model *goneuron.Model, tokenizer *SimpleTokenizer, lines []string) {
	inputs, targets := prepareData(tokenizer, lines)

	fmt.Println("Starting training loop...")
	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := float32(0)
		correct := 0
		total := 0

		for i := 0; i < len(inputs); i++ {
			// Training step
			l := model.Train(inputs[i], targets[i])
			if math.IsNaN(float64(l)) {
				fmt.Println("Warning: NaN loss detected")
				return
			}
			totalLoss += l

			// Compute accuracy for this sample
			pred := model.Forward(inputs[i])
			vSize := tokenizer.VocabSize()
			for s := 0; s < seqLen-1; s++ {
				targetID := -1
				for t := 0; t < vSize; t++ {
					if targets[i][s*vSize+t] == 1.0 {
						targetID = t
						break
					}
				}

				if targetID <= 0 { continue }

				logits := pred[s*vSize : (s+1)*vSize]
				predID := 0
				maxLogit := logits[0]
				for j, logit := range logits {
					if logit > maxLogit {
						maxLogit = logit
						predID = j
					}
				}

				if predID == targetID {
					correct++
				}
				total++
			}
		}

		if epoch%10 == 0 || epoch == 1 {
			acc := float32(0)
			if total > 0 {
				acc = float32(correct) / float32(total)
			}
			fmt.Printf("Epoch %d/%d - Loss: %.4f | Acc: %.2f%%\n", epoch, epochs, totalLoss/float32(len(inputs)), acc*100)
		}
	}
}

func evaluate(model *net.Network, tokenizer *SimpleTokenizer, lines []string) (float32, float32) {
	inputs, targets := prepareData(tokenizer, lines)
	var totalLoss float32
	var correct, total int
	vSize := tokenizer.VocabSize()

	for i := 0; i < len(inputs); i++ {
		pred := model.Forward(inputs[i])
		totalLoss += model.Loss().Forward(pred, targets[i])

		for s := 0; s < seqLen-1; s++ {
			targetID := -1
			for t := 0; t < vSize; t++ {
				if targets[i][s*vSize+t] == 1.0 {
					targetID = t
					break
				}
			}
			if targetID <= 0 { continue }

			logits := pred[s*vSize : (s+1)*vSize]
			predID := 0
			maxLogit := logits[0]
			for j, logit := range logits {
				if logit > maxLogit {
					maxLogit = logit
					predID = j
				}
			}
			if predID == targetID {
				correct++
			}
			total++
		}
	}
	acc := float32(0)
	if total > 0 {
		acc = float32(correct) / float32(total)
	}
	return acc, totalLoss / float32(len(inputs))
}

func prepareData(tokenizer *SimpleTokenizer, lines []string) ([][]float32, [][]float32) {
	vocabSize := tokenizer.VocabSize()
	inputs := make([][]float32, len(lines))
	targets := make([][]float32, len(lines))

	for i, line := range lines {
		tokens := tokenizer.Encode(line)
		inputs[i] = tokens

		tar := make([]float32, seqLen*vocabSize)
		for s := 0; s < seqLen-1; s++ {
			targetID := int(tokens[s+1])
			if targetID > 0 && targetID < vocabSize {
				tar[s*vocabSize+targetID] = 1.0
			} else {
				tar[s*vocabSize+0] = 1.0 // PAD
			}
		}
		tar[(seqLen-1)*vocabSize+0] = 1.0
		targets[i] = tar
	}
	return inputs, targets
}

func saveModel(model *net.Network, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(model.Params())
}

func loadModel(model *net.Network, filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	var params []float32
	if err := gob.NewDecoder(f).Decode(&params); err != nil {
		return err
	}
	model.SetParams(params)
	return nil
}

func sample(logits []float32, temperature float32, topK int) int {
	if temperature <= 0 {
		bestID := 0
		maxLogit := logits[0]
		for i, l := range logits {
			if l > maxLogit {
				maxLogit = l
				bestID = i
			}
		}
		return bestID
	}

	scaled := make([]float32, len(logits))
	for i, l := range logits {
		logit := float32(math.Log(float64(l + 1e-10)))
		scaled[i] = logit / temperature
	}

	var sumExp float64
	maxLogit := scaled[0]
	for _, l := range scaled {
		if l > maxLogit { maxLogit = l }
	}
	for i, l := range scaled {
		scaled[i] = float32(math.Exp(float64(l - maxLogit)))
		sumExp += float64(scaled[i])
	}
	for i := range scaled {
		scaled[i] /= float32(sumExp)
	}

	type probIdx struct {
		prob float32
		idx  int
	}
	probs := make([]probIdx, len(scaled))
	for i, p := range scaled {
		probs[i] = probIdx{p, i}
	}
	sort.Slice(probs, func(i, j int) bool {
		return probs[i].prob > probs[j].prob
	})

	if topK > 0 && topK < len(probs) {
		probs = probs[:topK]
		var newSum float32
		for _, p := range probs {
			newSum += p.prob
		}
		for i := range probs {
			probs[i].prob /= newSum
		}
	}

	r := rand.Float32()
	var cumulative float32
	for _, p := range probs {
		cumulative += p.prob
		if r <= cumulative {
			return p.idx
		}
	}
	return probs[0].idx
}

func generateText(model *net.Network, tokenizer *SimpleTokenizer, seed string, maxNewTokens int, temp float32, topK int) string {
	words := tokenize(seed)
	seedTokens := []float32{2.0} // [BOS]
	for _, w := range words {
		id, ok := tokenizer.vocab[w]
		if !ok { id = 1 }
		seedTokens = append(seedTokens, float32(id))
	}

	for n := 0; n < maxNewTokens; n++ {
		input := make([]float32, seqLen)
		copy(input, seedTokens)

		for _, l := range model.Layers() {
			if resettable, ok := l.(interface{ Reset() }); ok {
				resettable.Reset()
			}
		}
		pred := model.Forward(input)

		lastIdx := len(seedTokens) - 1
		if lastIdx >= seqLen { break }

		vocabSize := tokenizer.VocabSize()
		logits := pred[lastIdx*vocabSize : (lastIdx+1)*vocabSize]

		bestID := sample(logits, temp, topK)

		if bestID == 3 || bestID == 0 { break }
		seedTokens = append(seedTokens, float32(bestID))
		if len(seedTokens) >= seqLen { break }
	}

	return tokenizer.Decode(seedTokens)
}

func interactiveLoop(model *net.Network, tokenizer *SimpleTokenizer) {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("\n--- Modo Interativo (PT-PT) ---")
	fmt.Println("Escreve um início de frase (ou 'sair' para terminar):")
	fmt.Println("Exemplos: 'Lisboa é', 'O Supremo', 'O governo'")

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()
		if strings.ToLower(input) == "sair" {
			break
		}

		gen := generateText(model, tokenizer, input, 8, 0.8, 10)
		fmt.Printf("Gerado: %s\n\n", gen)
	}
}

func generateReport(model *net.Network, device layer.Device) {
	report := `
# Architecture Overview
- Layers: Embedding -> PositionalEncoding -> TransformerBlock (Causal) -> SequenceUnroller(Dense+Softmax)
- Hyper-parameters:
  - Embedding Dim: 32
  - Attention Heads: 4
  - FF Dim: 64
  - Seq Len: 12
  - Optimizer: Adam (LR=0.002)

# Dataset Description
- Source: Small corpus of Portuguese (PT-PT) sentences.
- Task: Next Token Prediction.

# Training Results
- Final Loss: Evaluation results stored in logs.
- Accuracy: Tracked during training and final evaluation.
- Device: ` + fmt.Sprintf("%v", device.Type()) + `

# Inference Performance
- Latency per sequence: Optimized with ` + fmt.Sprintf("%v", device.Type()) + `
- Sampling: Temperature-based and Top-K for diverse generation.

# Benchmarking
- Memory Layout: Contiguous float32
- Hardware Acceleration: ` + fmt.Sprintf("%v", device.Type() == layer.GPU) + `
`
	err := os.WriteFile("/Users/flaviocfo/dev/github.com/FlavioCFOliveira/GoNeuron/examples/portuguese_gpt/report.md", []byte(report), 0644)
	if err != nil {
		fmt.Printf("Error writing report: %v\n", err)
	}
	fmt.Println("\nReport generated: examples/portuguese_gpt/report.md")
}
