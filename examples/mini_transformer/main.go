// Mini-Transformer for Sentiment Analysis example
package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// Vocabulary holds the word-to-index mapping
type Vocabulary struct {
	wordToIdx map[string]int
	nextIdx   int
}

func NewVocabulary() *Vocabulary {
	v := &Vocabulary{
		wordToIdx: make(map[string]int),
		nextIdx:   2, // 0 = <PAD>, 1 = <UNK>
	}
	v.wordToIdx["<PAD>"] = 0
	v.wordToIdx["<UNK>"] = 1
	return v
}

func (v *Vocabulary) AddWord(word string) int {
	if idx, exists := v.wordToIdx[word]; exists {
		return idx
	}
	idx := v.nextIdx
	v.wordToIdx[word] = idx
	v.nextIdx++
	return idx
}

func (v *Vocabulary) GetIdx(word string) int {
	if idx, exists := v.wordToIdx[word]; exists {
		return idx
	}
	return 1 // <UNK>
}

func Tokenize(sentence string, vocab *Vocabulary, maxLength int) []float32 {
	words := strings.Fields(strings.ToLower(sentence))
	tokens := make([]float32, maxLength)
	for i := 0; i < maxLength; i++ {
		if i < len(words) {
			tokens[i] = float32(vocab.GetIdx(words[i]))
		} else {
			tokens[i] = 0 // <PAD>
		}
	}
	return tokens
}

func main() {
	fmt.Println("=== Mini-Transformer Sentiment Analysis ===")

	// 1. Data Preparation
	posSentences := []string{
		"this movie was great",
		"i loved the acting",
		"fantastic storyline and characters",
		"brilliant direction and cinematography",
		"highly recommend this film",
	}
	negSentences := []string{
		"this movie was terrible",
		"i hated the plot",
		"poor acting and weak script",
		"boring and predictable story",
		"waste of time and money",
	}

	vocab := NewVocabulary()
	allSentences := append(posSentences, negSentences...)
	for _, s := range allSentences {
		for _, w := range strings.Fields(strings.ToLower(s)) {
			vocab.AddWord(w)
		}
	}

	maxLen := 10
	x := make([][]float32, len(allSentences))
	y := make([][]float32, len(allSentences))

	for i, s := range allSentences {
		x[i] = Tokenize(s, vocab, maxLen)
		if i < len(posSentences) {
			y[i] = []float32{1, 0} // Positive
		} else {
			y[i] = []float32{0, 1} // Negative
		}
	}

	// Shuffle
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(x), func(i, j int) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	})

	// 2. Model Architecture
	const (
		dModel   = 32
		numHeads = 4
		ffDim    = 64
	)

	// Architecture:
	// 1. Embedding
	// 2. Positional Encoding
	// 3. Transformer Block
	// 4. Global Average Pooling
	// 5. Dense -> LogSoftmax

	layers := []layer.Layer{
		layer.NewEmbedding(vocab.nextIdx, dModel),
		layer.NewPositionalEncoding(maxLen, dModel),
		layer.NewTransformerBlock(dModel, numHeads, maxLen, ffDim),
		layer.NewGlobalAveragePooling1D(maxLen, dModel),
		layer.NewDense(dModel, 2, activations.LogSoftmax{}),
	}

	optimizer := opt.NewAdam(0.001)
	network := net.New(layers, loss.CrossEntropy{}, optimizer)

	// 3. Training (Simplified for demonstration)
	fmt.Println("Starting training...")
	network.Fit(x, y, 10, 2, net.Logger{Interval: 2})

	// 4. Inference
	testSentence := "i loved the movie"
	testTokens := Tokenize(testSentence, vocab, maxLen)
	prediction := network.Forward(testTokens)

	fmt.Printf("\nTest Sentence: \"%s\"\n", testSentence)
	if prediction[0] > prediction[1] {
		fmt.Println("Prediction: Positive")
	} else {
		fmt.Println("Prediction: Negative")
	}
}
