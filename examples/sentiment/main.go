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

func main() {
	// 1. Synthetic Sentiment Dataset (Small for demonstration)
	positive := []string{
		"i love this movie",
		"great film very happy",
		"excellent acting and story",
		"it was fantastic and amazing",
		"highly recommended for everyone",
		"the best movie of the year",
		"i really enjoyed this masterpiece",
		"wonderful experience and great plot",
		"superb performance by the cast",
		"this is a must watch film",
	}

	negative := []string{
		"i hate this movie",
		"terrible film very sad",
		"bad acting and story",
		"it was boring and awful",
		"not recommended for anyone",
		"the worst movie of the year",
		"i really disliked this garbage",
		"horrible experience and bad plot",
		"poor performance by the cast",
		"this is a waste of time",
	}

	// 2. Vocabulary and Tokenization
	vocab := make(map[string]int)
	vocab["<PAD>"] = 0
	vocab["<UNK>"] = 1
	idx := 2

	allSentences := append(positive, negative...)
	for _, s := range allSentences {
		words := strings.Fields(strings.ToLower(s))
		for _, w := range words {
			if _, ok := vocab[w]; !ok {
				vocab[w] = idx
				idx++
			}
		}
	}

	vocabSize := len(vocab)
	maxLen := 5

	tokenize := func(s string) []float64 {
		words := strings.Fields(strings.ToLower(s))
		tokens := make([]float64, maxLen)
		for i := 0; i < maxLen; i++ {
			if i < len(words) {
				if id, ok := vocab[words[i]]; ok {
					tokens[i] = float64(id)
				} else {
					tokens[i] = 1.0 // <UNK>
				}
			} else {
				tokens[i] = 0.0 // <PAD>
			}
		}
		return tokens
	}

	// 3. Prepare Training Data
	x := make([][]float64, 0)
	y := make([][]float64, 0)

	for _, s := range positive {
		x = append(x, tokenize(s))
		y = append(y, []float64{1.0, 0.0}) // One-hot: [Pos, Neg]
	}
	for _, s := range negative {
		x = append(x, tokenize(s))
		y = append(y, []float64{0.0, 1.0}) // One-hot: [Pos, Neg]
	}

	// Shuffle
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(x), func(i, j int) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	})

	fmt.Printf("Vocab Size: %d, Samples: %d\n", vocabSize, len(x))

	// 4. Define Architecture: Embedding -> LSTM -> Dense
	layers := []layer.Layer{
		layer.NewEmbedding(vocabSize, 16),
		layer.NewLSTM(16, 32),
		layer.NewDense(32, 2, activations.LogSoftmax{}),
	}

	optimizer := opt.NewAdam(0.01)
	model := net.New(layers, loss.CrossEntropy{}, optimizer)

	// 5. Training
	fmt.Println("Starting training...")
	model.Fit(x, y, 50, 4, net.Logger{Interval: 10})

	// 6. Test on new sentences
	testSentences := []string{
		"it was a great movie",
		"i hated the film",
		"amazing story",
		"waste of time",
	}

	fmt.Println("\nTesting Model:")
	for _, s := range testSentences {
		tokens := tokenize(s)
		pred := model.Forward(tokens)
		sentiment := "Positive"
		if pred[1] > pred[0] {
			sentiment = "Negative"
		}
		fmt.Printf("Sentence: '%s' -> Predicted: %s (Scores: %.2f, %.2f)\n", s, sentiment, pred[0], pred[1])
	}
}
