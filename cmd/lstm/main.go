package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// Sequence type for LSTM training data
type Sequence struct {
	Input  []float64
	Output []float64
}

// LSTM examples demonstrating sequence modeling
func main() {
	rand.Seed(42)

	fmt.Println("=== LSTM Examples ===")

	// Example 1: Simple sequence prediction (y = x + previous output)
	fmt.Println("Example 1: Sequential prediction (y_t = x_t + 0.5*y_{t-1})")
	exampleSequentialPrediction()

	// Example 2: XOR-like sequence (needs memory)
	fmt.Println("\nExample 2: XOR sequence (needs memory)")
	exampleXORSequence()

	// Example 3: Sinusoidal sequence prediction
	fmt.Println("\nExample 3: Sinusoidal sequence prediction")
	exampleSinusoidalPrediction()
}

func exampleSequentialPrediction() {
	// LSTM for sequential prediction
	// Input: scalar, Output: scalar
	in := 1
	hidden := 8
	out := 1

	lstm := layer.NewLSTM(in, hidden)
	output := layer.NewDense(hidden, out, activations.Sigmoid{})

	network := net.New(
		[]layer.Layer{lstm, output},
		loss.MSE{},
		opt.NewAdam(0.05),
	)

	sequences := generateSequentialData(50, 10)

	fmt.Println("  Training...")
	for epoch := 0; epoch < 200; epoch++ {
		totalLoss := 0.0
		count := 0

		for _, seq := range sequences {
			lstm.Reset()
			for t := 0; t < len(seq.Input); t++ {
				_ = lstm.Forward([]float64{seq.Input[t]})
				if t > 0 {
					_ = network.Train([]float64{0}, []float64{seq.Output[t]})
					totalLoss += math.Abs(seq.Output[t] - network.Forward([]float64{0})[0])
					count++
				}
			}
		}

		if epoch%50 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(count))
		}
	}

	fmt.Println("  Training completed!")
}

func exampleXORSequence() {
	in := 1
	hidden := 4
	out := 1

	lstm := layer.NewLSTM(in, hidden)
	output := layer.NewDense(hidden, out, activations.Sigmoid{})

	network := net.New(
		[]layer.Layer{lstm, output},
		loss.MSE{},
		opt.NewAdam(0.1),
	)

	sequences := generateXORSequences(30, 8)

	fmt.Println("  Training...")
	for epoch := 0; epoch < 300; epoch++ {
		totalLoss := 0.0
		count := 0

		for _, seq := range sequences {
			lstm.Reset()
			for t := 0; t < len(seq.Input); t++ {
				_ = lstm.Forward([]float64{seq.Input[t]})
				if t > 0 {
					_ = network.Train([]float64{0}, []float64{seq.Output[t]})
					totalLoss += math.Abs(seq.Output[t] - network.Forward([]float64{0})[0])
					count++
				}
			}
		}

		if epoch%100 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(count))
		}
	}

	fmt.Println("  Testing XOR sequence:")
	testSeq := []float64{0, 1, 0, 1, 0, 1, 0, 1}
	lstm.Reset()
	fmt.Print("  Input: ")
	for _, x := range testSeq {
		fmt.Printf("%.0f ", x)
	}
	fmt.Println()

	fmt.Print("  Predicted: ")
	for _, x := range testSeq {
		out := lstm.Forward([]float64{x})
		pred := output.Forward(out)
		fmt.Printf("%.3f ", pred[0])
	}
	fmt.Println()
}

func exampleSinusoidalPrediction() {
	in := 1
	hidden := 16
	out := 1

	lstm := layer.NewLSTM(in, hidden)
	output := layer.NewDense(hidden, out, activations.Sigmoid{})

	network := net.New(
		[]layer.Layer{lstm, output},
		loss.MSE{},
		opt.NewAdam(0.02),
	)

	sequences := generateSineSequences(50, 20)

	fmt.Println("  Training...")
	for epoch := 0; epoch < 500; epoch++ {
		totalLoss := 0.0
		count := 0

		for _, seq := range sequences {
			lstm.Reset()
			for t := 0; t < len(seq.Input); t++ {
				_ = lstm.Forward([]float64{seq.Input[t]})
				if t > 0 {
					_ = network.Train([]float64{0}, []float64{seq.Output[t]})
					totalLoss += math.Abs(seq.Output[t] - network.Forward([]float64{0})[0])
					count++
				}
			}
		}

		if epoch%100 == 0 {
			fmt.Printf("  Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(count))
		}
	}

	fmt.Println("  Testing sine prediction:")
	lstm.Reset()
	inputs := []float64{0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5}
	fmt.Println("  Input -> Predicted (Target)")

	for _, x := range inputs {
		out := lstm.Forward([]float64{x})
		pred := output.Forward(out)
		target := math.Sin(x*2*math.Pi)/2 + 0.5
		fmt.Printf("  %.2f -> %.4f (%.4f)\n", x, pred[0], target)
	}
}

func generateSequentialData(nSequences, seqLength int) []Sequence {
	sequences := make([]Sequence, nSequences)

	for i := 0; i < nSequences; i++ {
		xSeq := make([]float64, seqLength)
		ySeq := make([]float64, seqLength)

		yPrev := 0.0
		for t := 0; t < seqLength; t++ {
			xSeq[t] = rand.Float64()
			ySeq[t] = xSeq[t] + 0.5*yPrev
			yPrev = ySeq[t]
		}

		sequences[i] = Sequence{Input: xSeq, Output: ySeq}
	}

	return sequences
}

func generateXORSequences(nSequences, seqLength int) []Sequence {
	sequences := make([]Sequence, nSequences)

	for i := 0; i < nSequences; i++ {
		xSeq := make([]float64, seqLength)
		ySeq := make([]float64, seqLength)

		onesCount := 0
		for t := 0; t < seqLength; t++ {
			if rand.Float64() < 0.5 {
				xSeq[t] = 1
				onesCount++
			} else {
				xSeq[t] = 0
			}
			ySeq[t] = float64(onesCount % 2)
		}

		sequences[i] = Sequence{Input: xSeq, Output: ySeq}
	}

	return sequences
}

func generateSineSequences(nSequences, seqLength int) []Sequence {
	sequences := make([]Sequence, nSequences)

	for i := 0; i < nSequences; i++ {
		xSeq := make([]float64, seqLength)
		ySeq := make([]float64, seqLength)

		for t := 0; t < seqLength; t++ {
			xSeq[t] = float64(t) / float64(seqLength-1) * 2 * math.Pi
			ySeq[t] = math.Sin(xSeq[t])/2 + 0.5
		}

		sequences[i] = Sequence{Input: xSeq, Output: ySeq}
	}

	return sequences
}
