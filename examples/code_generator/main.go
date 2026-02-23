package main

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

type CharTokenizer struct {
	charToIdx map[rune]int
	idxToChar map[int]rune
	vocabSize int
}

func NewCharTokenizer(text string) *CharTokenizer {
	chars := make(map[rune]bool)
	for _, r := range text {
		chars[r] = true
	}
	var sortedChars []rune
	for r := range chars {
		sortedChars = append(sortedChars, r)
	}
	sort.Slice(sortedChars, func(i, j int) bool { return sortedChars[i] < sortedChars[j] })
	charToIdx := make(map[rune]int)
	idxToChar := make(map[int]rune)
	for i, r := range sortedChars {
		charToIdx[r] = i
		idxToChar[i] = r
	}
	return &CharTokenizer{charToIdx: charToIdx, idxToChar: idxToChar, vocabSize: len(sortedChars)}
}

func (t *CharTokenizer) Encode(text string) []int {
	res := make([]int, len(text))
	for i, r := range text {
		idx, ok := t.charToIdx[r]
		if !ok { idx = 0 }
		res[i] = idx
	}
	return res
}

func (t *CharTokenizer) Decode(indices []int) string {
	var sb strings.Builder
	for _, idx := range indices {
		r, ok := t.idxToChar[idx]
		if !ok { r = '?' }
		sb.WriteRune(r)
	}
	return sb.String()
}

func main() {
	content, err := ioutil.ReadFile("examples/code_generator/datasets/go_code.txt")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	text := string(content)
	tokenizer := NewCharTokenizer(text)
	encoded := tokenizer.Encode(text)

	seqLen := 48
	embeddingDim := 64
	numHeads := 4
	ffDim := 128
	learningRate := float32(0.001)
	epochs := 300

	emb := layer.NewEmbedding(tokenizer.vocabSize, embeddingDim)
	pos := layer.NewPositionalEncoding(seqLen, embeddingDim)
	block := layer.NewTransformerBlock(embeddingDim, numHeads, seqLen, ffDim, true)
	head := layer.NewDense(embeddingDim, tokenizer.vocabSize, activations.Linear{})

	device := layer.GetDefaultDevice()
	emb.SetDevice(device)
	pos.SetDevice(device)
	block.SetDevice(device)
	head.SetDevice(device)

	criterion := loss.CrossEntropy{}
	optimizer := opt.NewAdam(learningRate)

	rand.Seed(time.Now().UnixNano())
	fmt.Println("Training Code Generator...")
	startTime := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)
		numBatches := 0
		for i := 0; i < len(encoded)-seqLen-1; i += seqLen {
			inputIndices := encoded[i : i+seqLen]
			targetIndices := encoded[i+1 : i+seqLen+1]
			input := make([]float32, seqLen)
			for j, idx := range inputIndices { input[j] = float32(idx) }

			x := emb.Forward(input)
			x = pos.Forward(x)
			x = block.Forward(x)

			gradsHead := make([]float32, seqLen*embeddingDim)
			allPredProbs := make([][]float32, seqLen)
			for t := 0; t < seqLen; t++ {
				logits := head.Forward(x[t*embeddingDim : (t+1)*embeddingDim])
				allPredProbs[t] = activations.Softmax{}.ActivateBatch(logits)
			}

			lossSum := float32(0.0)
			for t := seqLen - 1; t >= 0; t-- {
				targetOneHot := make([]float32, tokenizer.vocabSize)
				targetOneHot[targetIndices[t]] = 1.0
				lossSum += criterion.Forward(allPredProbs[t], targetOneHot)
				gradOut := criterion.Backward(allPredProbs[t], targetOneHot)
				gradIn := head.AccumulateBackward(gradOut)
				copy(gradsHead[t*embeddingDim:(t+1)*embeddingDim], gradIn)
			}
			totalLoss += lossSum / float32(seqLen)
			numBatches++

			grad := block.Backward(gradsHead)
			grad = pos.Backward(grad)
			emb.Backward(grad)

			optimizer.NewStep()
			updateParams(head, optimizer)
			updateParams(block, optimizer)
			updateParams(pos, optimizer)
			updateParams(emb, optimizer)

			head.ClearGradients()
			block.ClearGradients()
			pos.ClearGradients()
			emb.ClearGradients()
		}
		if epoch%50 == 0 || epoch == epochs-1 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float32(numBatches))
		}
	}

	fmt.Println("\nGenerated Go Code:")
	seed := "func "
	generated := seed
	currIndices := tokenizer.Encode(seed)
	input := make([]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		if i < len(currIndices) { input[i] = float32(currIndices[i]) } else { input[i] = 0 }
	}

	for i := 0; i < 150; i++ {
		x := emb.Forward(input)
		x = pos.Forward(x)
		x = block.Forward(x)
		lastIdx := len(currIndices) - 1
		if lastIdx < 0 { lastIdx = 0 }
		if lastIdx >= seqLen { lastIdx = seqLen - 1 }
		logits := head.Forward(x[lastIdx*embeddingDim : (lastIdx+1)*embeddingDim])
		nextIdx := sample(activations.Softmax{}.ActivateBatch(logits))
		nextChar := tokenizer.idxToChar[nextIdx]
		generated += string(nextChar)

		if len(currIndices) < seqLen {
			currIndices = append(currIndices, nextIdx)
			input[len(currIndices)-1] = float32(nextIdx)
		} else {
			copy(currIndices, currIndices[1:])
			currIndices[seqLen-1] = nextIdx
			for j := 0; j < seqLen; j++ { input[j] = float32(currIndices[j]) }
		}
	}
	fmt.Println(generated)
	fmt.Printf("\nTraining speed: %.2f tokens/sec\n", float64(epochs*len(encoded))/time.Since(startTime).Seconds())
}

func updateParams(l layer.Layer, o opt.Optimizer) {
	p := l.Params()
	g := l.Gradients()
	if p == nil || g == nil { return }
	l.SetParams(o.Step(p, g))
}

func sample(probs []float32) int {
	r := rand.Float32()
	var cumulative float32
	for i, p := range probs {
		cumulative += p
		if r <= cumulative { return i }
	}
	return len(probs) - 1
}
