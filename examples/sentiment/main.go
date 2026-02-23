package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

func main() {
	// 1. Expanded Synthetic Sentiment Dataset
	posSentences := []string{
		"i love this movie", "great film very happy", "excellent acting and story",
		"it was fantastic and amazing", "highly recommended for everyone",
		"the best movie of the year", "i really enjoyed this masterpiece",
		"wonderful experience and great plot", "superb performance by the cast",
		"this is a must watch film", "absolutely brilliant cinematography",
		"a truly touching and beautiful story", "i could watch this every day",
		"the direction was flawless", "pure joy to watch", "captivating from start to finish",
		"a landmark in cinema history", "unforgettable characters and dialogue",
		"the soundtrack was perfect", "it exceeded all my expectations",
		"a gem of a film with heart", "truly inspiring and uplifting",
		"one of the greatest stories ever told", "the acting was top notch",
		"beautifully filmed and acted",
	}

	negSentences := []string{
		"i hate this movie", "terrible film very sad", "bad acting and story",
		"it was boring and awful", "not recommended for anyone",
		"the worst movie of the year", "i really disliked this garbage",
		"horrible experience and bad plot", "poor performance by the cast",
		"this is a waste of time", "dull and uninspired writing",
		"complete disaster of a movie", "i wanted to leave the theater",
		"the logic makes no sense", "painfully slow and annoying",
		"cliched and predictable throughout", "save your money and skip it",
		"the special effects were cheap", "worst acting i have ever seen",
		"it was a total disappointment",
		"poorly directed and confusing", "avoid at all costs",
		"not worth the price of admission", "a boring mess of a movie",
		"everything about it was bad",
	}

	// 2. Vocabulary and Tokenization
	vocab := make(map[string]int)
	vocab["<PAD>"] = 0
	vocab["<UNK>"] = 1
	idx := 2

	allSentences := append(posSentences, negSentences...)
	for _, s := range allSentences {
		words := strings.Fields(strings.ToLower(s))
		for _, w := range words {
			w = strings.Trim(w, ".,!?:;")
			if _, ok := vocab[w]; !ok {
				vocab[w] = idx
				idx++
			}
		}
	}

	vocabSize := len(vocab)
	maxLen := 10

	tokenize := func(s string) []float32 {
		words := strings.Fields(strings.ToLower(s))
		tokens := make([]float32, maxLen)
		for i := 0; i < maxLen; i++ {
			if i < len(words) {
				w := strings.Trim(words[i], ".,!?:;")
				if id, ok := vocab[w]; ok {
					tokens[i] = float32(id)
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
	x := make([][]float32, 0)
	y := make([][]float32, 0)

	for _, s := range posSentences {
		x = append(x, tokenize(s))
		y = append(y, []float32{1.0, 0.0}) // One-hot: [Pos, Neg]
	}
	for _, s := range negSentences {
		x = append(x, tokenize(s))
		y = append(y, []float32{0.0, 1.0}) // One-hot: [Pos, Neg]
	}

	// Shuffle
	rand.Shuffle(len(x), func(i, j int) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	})

	fmt.Printf("Vocab Size: %d, Samples: %d, Max Length: %d\n", vocabSize, len(x), maxLen)

	// 4. Define Architecture: Embedding -> Unrolled GRU -> Dense
	model := goneuron.NewSequential(
		goneuron.Embedding(vocabSize, 16), // [maxLen, 16]
		goneuron.Flatten(),                // SequenceUnroller expects [T * InSize]
		goneuron.SequenceUnroller(goneuron.GRU(16, 32), maxLen, false), // Returns [32]
		goneuron.Dense(32, 2, goneuron.LogSoftmax),
	)

	optimizer := goneuron.Adam(0.001)
	model.Compile(optimizer, goneuron.NLLLoss)

	// 5. Training with Callbacks
	fmt.Println("\nStarting training...")
	start := time.Now()

	scheduler := goneuron.ReduceLROnPlateau(optimizer, 0.5, 5, 0.0001, 0.00001)
	callbacks := []goneuron.Callback{
		goneuron.Logger(10),
		goneuron.EarlyStopping(10, 0.0001),
		goneuron.SchedulerCallback(scheduler),
	}

	model.Fit(x, y, 200, 4, callbacks...)
	fmt.Printf("Training finished in %v\n", time.Since(start))

	// 6. Evaluation
	testSentences := []string{
		"it was a great movie",
		"i hated the film",
		"absolutely amazing masterpiece",
		"a waste of my time and money",
		"brilliant acting and direction",
		"boring and predictable plot",
		"truly wonderful and inspiring",
		"terrible waste of talent",
	}

	fmt.Println("\nTesting Model:")
	for _, s := range testSentences {
		tokens := tokenize(s)
		pred := model.Forward(tokens)
		sentiment := "Positive"
		if pred[1] > pred[0] {
			sentiment = "Negative"
		}
		pPos := float32(math.Exp(float64(pred[0])))
		pNeg := float32(math.Exp(float64(pred[1])))
		confidence := float32(math.Max(float64(pPos), float64(pNeg))) * 100
		fmt.Printf("Sentence: '%-30s' -> Predicted: %-8s (Confidence: %.2f%%)\n",
			s, sentiment, confidence)
	}
}
