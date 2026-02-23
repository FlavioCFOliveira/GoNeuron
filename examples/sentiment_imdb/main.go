// IMDB Sentiment Analysis example upgraded with BiLSTM and Attention
package main

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

// Vocabulary holds the word-to-index mapping
type Vocabulary struct {
	wordToIdx map[string]int
	idxToWord map[int]string
	nextIdx   int
}

func NewVocabulary() *Vocabulary {
	vocab := &Vocabulary{
		wordToIdx: make(map[string]int),
		idxToWord: make(map[int]string),
		nextIdx:   2, // 0 = <PAD>, 1 = <UNK>
	}
	vocab.wordToIdx["<PAD>"] = 0
	vocab.wordToIdx["<UNK>"] = 1
	vocab.idxToWord[0] = "<PAD>"
	vocab.idxToWord[1] = "<UNK>"
	return vocab
}

func (v *Vocabulary) AddWord(word string) int {
	if idx, exists := v.wordToIdx[word]; exists {
		return idx
	}
	idx := v.nextIdx
	v.wordToIdx[word] = idx
	v.idxToWord[idx] = word
	v.nextIdx++
	return idx
}

func (v *Vocabulary) GetIdx(word string) int {
	if idx, exists := v.wordToIdx[word]; exists {
		return idx
	}
	return 1 // <UNK>
}

func BuildVocabulary(sentences []string, maxWords int) *Vocabulary {
	wordFreq := make(map[string]int)
	for _, s := range sentences {
		words := strings.Fields(strings.ToLower(s))
		for _, w := range words {
			w = strings.Trim(w, ".,!?:;\"'()[]{}<>")
			if w != "" {
				wordFreq[w]++
			}
		}
	}
	type wordCount struct {
		word  string
		count int
	}
	var wc []wordCount
	for w, c := range wordFreq {
		wc = append(wc, wordCount{w, c})
	}
	sort.Slice(wc, func(i, j int) bool {
		return wc[i].count > wc[j].count
	})
	vocab := NewVocabulary()
	for i := 0; i < min(len(wc), maxWords-2); i++ {
		vocab.AddWord(wc[i].word)
	}
	return vocab
}

func Tokenize(sentence string, vocab *Vocabulary, maxLength int) []float32 {
	words := strings.Fields(strings.ToLower(sentence))
	tokens := make([]float32, maxLength)
	for i := 0; i < maxLength; i++ {
		if i < len(words) {
			w := strings.Trim(words[i], ".,!?:;\"'()[]{}<>")
			if w != "" {
				tokens[i] = float32(vocab.GetIdx(w))
			} else {
				tokens[i] = 0
			}
		} else {
			tokens[i] = 0
		}
	}
	return tokens
}

func main() {
	fmt.Println("=== IMDB Sentiment Analysis using BiLSTM + Attention ===")

	posSentences := []string{
		"this movie was absolutely fantastic and wonderful",
		"i loved every moment of this brilliant film",
		"amazing storyline with outstanding performances",
		"truly a masterpiece of modern cinema",
		"excellent acting and wonderful direction",
		"the best movie i have ever seen",
		"highly recommend this incredible film",
		"beautiful cinematography and touching story",
		"gorgeous film with heartfelt performances",
		"an utter joy from start to finish",
		"superb entertainment throughout",
		"wonderful characters and great dialogue",
		"truly inspiring and uplifting movie",
		"the perfect film experience",
		"absolutely delightful and entertaining",
		"brilliant writing and direction",
		"outstanding cast with perfect chemistry",
		"cinematic excellence at its finest",
		"a true gem of a movie",
		"masterful storytelling and acting",
	}

	negSentences := []string{
		"this movie was terrible and boring",
		"i hated every minute of this waste of time",
		"poor acting and weak storyline",
		"disappointing and poorly made film",
		"awful direction and bad script",
		"the worst movie of the year",
		"not worth watching or recommending",
		"dull and uninspired cinematography",
		"horrible performances throughout",
		"a complete disappointment",
		"utter waste of money",
		"boring characters and weak plot",
		"painfully slow and annoying",
		"the most awful film i have seen",
		"terrible entertainment value",
		"poorly written and directed",
		"weak cast with no chemistry",
		"terrible cinematic experience",
		"a disaster of a movie",
		"avoid at all costs",
	}

	allSentences := append(posSentences, negSentences...)
	labels := make([]float32, len(allSentences))
	for i := len(posSentences); i < len(labels); i++ {
		labels[i] = 1.0
	}

	maxVocabSize := 1000
	vocab := BuildVocabulary(allSentences, maxVocabSize)
	maxLen := 20

	x := make([][]float32, len(allSentences))
	for i, s := range allSentences {
		x[i] = Tokenize(s, vocab, maxLen)
	}

	y := make([][]float32, len(allSentences))
	for i := range y {
		if labels[i] == 0 {
			y[i] = []float32{1.0, 0.0}
		} else {
			y[i] = []float32{0.0, 1.0}
		}
	}

	rand.Shuffle(len(x), func(i, j int) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	})

	split := int(float32(len(x)) * 0.8)
	xTrain, xTest := x[:split], x[split:]
	yTrain, yTest := y[:split], y[split:]

	const (
		embeddingDim = 32
		lstmUnits    = 32
	)

	// Build Advanced Architecture:
	// 1. Embedding
	// 2. Bidirectional LSTM
	// 3. Global Attention
	// 4. Dense Classifier

	model := goneuron.NewSequential(
		goneuron.Embedding(vocab.nextIdx, embeddingDim),
		goneuron.Flatten(),
		goneuron.SequenceUnroller(goneuron.Bidirectional(goneuron.LSTM(embeddingDim, lstmUnits)), maxLen, true),
		goneuron.GlobalAttention(lstmUnits*2),
		goneuron.Dense(lstmUnits*2, 2, goneuron.LogSoftmax),
	)

	optimizer := goneuron.Adam(0.001)
	model.Compile(optimizer, goneuron.CrossEntropy)

	fmt.Println("\nStarting BiLSTM + Attention training...")
	start := time.Now()
	epochs := 50
	batchSize := 4

	scheduler := goneuron.ReduceLROnPlateau(optimizer, 0.5, 5, 0.01, 1e-6)

	callbacks := []goneuron.Callback{
		goneuron.Logger(10),
		goneuron.SchedulerCallback(scheduler),
	}

	model.Fit(xTrain, yTrain, epochs, batchSize, callbacks...)
	fmt.Printf("\nTraining finished in %v\n", time.Since(start))

	correct := 0
	for i := 0; i < len(xTest); i++ {
		pred := model.Forward(xTest[i])
		if argmax(pred) == argmax(yTest[i]) {
			correct++
		}
	}
	fmt.Printf("Test Accuracy: %.2f%%\n", float32(correct)/float32(len(xTest))*100)

	// Save model
	model.Save("sentiment_imdb_bilstm_attention.gob")
}

func argmax(v []float32) int {
	maxIdx := 0
	maxVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > maxVal {
			maxVal = v[i]
			maxIdx = i
		}
	}
	return maxIdx
}

func min(a, b int) int {
	if a < b { return a }
	return b
}
