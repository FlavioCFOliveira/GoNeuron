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
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// CharTokenizer handles character-to-index mapping.
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
	sort.Slice(sortedChars, func(i, j int) bool {
		return sortedChars[i] < sortedChars[j]
	})

	charToIdx := make(map[rune]int)
	idxToChar := make(map[int]rune)
	for i, r := range sortedChars {
		charToIdx[r] = i
		idxToChar[i] = r
	}

	return &CharTokenizer{
		charToIdx: charToIdx,
		idxToChar: idxToChar,
		vocabSize: len(sortedChars),
	}
}

func (t *CharTokenizer) Encode(text string) []int {
	res := make([]int, len(text))
	for i, r := range text {
		idx, ok := t.charToIdx[r]
		if !ok {
			idx = 0 // Default
		}
		res[i] = idx
	}
	return res
}

func (t *CharTokenizer) Decode(indices []int) string {
	var sb strings.Builder
	for _, idx := range indices {
		r, ok := t.idxToChar[idx]
		if !ok {
			r = '?'
		}
		sb.WriteRune(r)
	}
	return sb.String()
}

func main() {
	// 1. Load Dataset
	content, err := ioutil.ReadFile("examples/poetry_generator/datasets/camoes.txt")
	if err != nil {
		fmt.Printf("Error reading dataset: %v\n", err)
		os.Exit(1)
	}
	text := string(content)
	tokenizer := NewCharTokenizer(text)
	encoded := tokenizer.Encode(text)

	fmt.Printf("Vocab size: %d\n", tokenizer.vocabSize)
	fmt.Printf("Dataset size: %d characters\n", len(encoded))

	// 2. Hyper-parameters
	seqLen := 32
	embeddingDim := 64
	numHeads := 4
	ffDim := 128
	learningRate := float32(0.001)
	epochs := 200

	// 3. Architecture
	emb := layer.NewEmbedding(tokenizer.vocabSize, embeddingDim)
	pos := layer.NewPositionalEncoding(seqLen, embeddingDim)
	block := layer.NewTransformerBlock(embeddingDim, numHeads, seqLen, ffDim, true)
	head := layer.NewDense(embeddingDim, tokenizer.vocabSize, activations.Linear{})

	// Check for Metal support
	device := layer.GetDefaultDevice()
	fmt.Printf("Using device: %T\n", device)
	emb.SetDevice(device)
	pos.SetDevice(device)
	block.SetDevice(device)
	head.SetDevice(device)

	layers := []layer.Layer{emb, pos, block}
	criterion := loss.CrossEntropy{}
	optimizer := opt.NewAdam(learningRate)

	// 4. Training Loop
	rand.Seed(time.Now().UnixNano())
	fmt.Println("Starting training...")
	startTime := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)
		numBatches := 0

		// Simple sequential sampling for this example
		for i := 0; i < len(encoded)-seqLen-1; i += seqLen {
			// Prepare input and target
			inputIndices := encoded[i : i+seqLen]
			targetIndices := encoded[i+1 : i+seqLen+1]

			input := make([]float32, seqLen)
			for j, idx := range inputIndices {
				input[j] = float32(idx)
			}

			// Forward Pass
			x := input
			x = emb.Forward(x)
			x = pos.Forward(x)
			x = block.Forward(x)

			// The Transformer output is [seqLen * embeddingDim]
			// We need to apply the head to each position
			lossSum := float32(0.0)
			gradsHead := make([]float32, seqLen*embeddingDim)

			// Multi-step forward for the head
			allPredProbs := make([][]float32, seqLen)
			for t := 0; t < seqLen; t++ {
				start := t * embeddingDim
				end := start + embeddingDim
				predLogits := head.Forward(x[start:end])

				softmax := activations.Softmax{}
				allPredProbs[t] = softmax.ActivateBatch(predLogits)
			}

			// Multi-step backward for the head
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

			// Backward Pass through other layers
			grad := gradsHead
			grad = block.Backward(grad)
			grad = pos.Backward(grad)
			emb.Backward(grad)

			// Optimizer Step
			optimizer.NewStep()
			updateParams(head, optimizer)
			updateParams(block, optimizer)
			updateParams(pos, optimizer)
			updateParams(emb, optimizer)

			// Reset gradients
			head.ClearGradients()
			block.ClearGradients()
			pos.ClearGradients()
			emb.ClearGradients()
		}

		if epoch%20 == 0 || epoch == epochs-1 {
			avgLoss := totalLoss / float32(numBatches)
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss)
		}
	}

	trainTime := time.Since(startTime)
	fmt.Printf("Training completed in %v\n", trainTime)

	// 5. Inference
	fmt.Println("\nGenerated Poetry:")
	seed := "Amor é "
	generated := seed
	currIndices := tokenizer.Encode(seed)

	// Pad or truncate to seqLen
	if len(currIndices) > seqLen {
		currIndices = currIndices[len(currIndices)-seqLen:]
	}
	input := make([]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		if i < len(currIndices) {
			input[i] = float32(currIndices[i])
		} else {
			input[i] = 0 // Padding
		}
	}

	for i := 0; i < 100; i++ {
		// Forward
		x := emb.Forward(input)
		x = pos.Forward(x)
		x = block.Forward(x)

		// Predict next char from the LAST non-padding position or simply the last position if we use causal mask
		lastIdx := len(currIndices) - 1
		if lastIdx < 0 { lastIdx = 0 }
		if lastIdx >= seqLen { lastIdx = seqLen - 1 }

		lastPosStart := lastIdx * embeddingDim
		logits := head.Forward(x[lastPosStart : lastPosStart+embeddingDim])

		// Sample from distribution
		softmax := activations.Softmax{}
		probs := softmax.ActivateBatch(logits)
		nextIdx := sample(probs)

		nextChar := tokenizer.idxToChar[nextIdx]
		generated += string(nextChar)

		// Update input for next step (sliding window)
		if len(currIndices) < seqLen {
			currIndices = append(currIndices, nextIdx)
			input[len(currIndices)-1] = float32(nextIdx)
		} else {
			copy(currIndices, currIndices[1:])
			currIndices[seqLen-1] = nextIdx
			for j := 0; j < seqLen; j++ {
				input[j] = float32(currIndices[j])
			}
		}

		if nextChar == '\n' && strings.HasSuffix(generated, "\n\n") {
			break
		}
	}
	fmt.Println(generated)

	// 6. Benchmarking
	fmt.Println("\nPerformance Metrics:")
	fmt.Printf("Total parameters: %d\n", countParams(layers, head))
	fmt.Printf("Training through-put: %.2f tokens/sec\n", float64(epochs*len(encoded))/trainTime.Seconds())

	// 7. Export Weights
	fmt.Println("\nExporting weights to weights.bin...")
	fullLayers := append(layers, head)
	netModel := net.New(fullLayers, criterion, optimizer)
	err = netModel.SaveBinaryWeights("weights.bin")
	if err != nil {
		fmt.Printf("Error exporting weights: %v\n", err)
	} else {
		fmt.Println("Weights exported successfully to weights.bin")
	}
}

func updateParams(l layer.Layer, o opt.Optimizer) {
	params := l.Params()
	grads := l.Gradients()
	if params == nil || grads == nil {
		return
	}
	newParams := o.Step(params, grads)
	l.SetParams(newParams)
}

func sample(probs []float32) int {
	r := rand.Float32()
	var cumulative float32
	for i, p := range probs {
		cumulative += p
		if r <= cumulative {
			return i
		}
	}
	return len(probs) - 1
}

func countParams(layers []layer.Layer, head layer.Layer) int {
	count := 0
	for _, l := range layers {
		count += len(l.Params())
	}
	count += len(head.Params())
	return count
}

func (t *CharTokenizer) VocabSize() int {
	return t.vocabSize
}
