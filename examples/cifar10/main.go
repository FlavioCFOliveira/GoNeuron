package main

import (
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

const (
	ImageSize   = 32
	Channels    = 3
	NumClasses  = 10
	RecordSize  = 1 + ImageSize*ImageSize*Channels // 1 byte label + 3072 bytes image
	TrainImages = 50000
	TestImages  = 10000
)

// CIFAR10 represents the dataset
type CIFAR10 struct {
	TrainInputs  [][]float32
	TrainTargets [][]float32
	TestInputs   [][]float32
	TestTargets  [][]float32
}

func loadBatch(filename string) ([][]float32, [][]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	var inputs [][]float32
	var targets [][]float32

	buf := make([]byte, RecordSize)
	for {
		_, err := io.ReadFull(file, buf)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}

		label := int(buf[0])
		target := make([]float32, NumClasses)
		target[label] = 1.0

		input := make([]float32, ImageSize*ImageSize*Channels)
		for i := 0; i < ImageSize*ImageSize*Channels; i++ {
			// Normalize to [0, 1]
			input[i] = float32(buf[i+1]) / 255.0
		}

		inputs = append(inputs, input)
		targets = append(targets, target)
	}

	return inputs, targets, nil
}

func LoadCIFAR10(dataDir string) (*CIFAR10, error) {
	cifar := &CIFAR10{}

	// Load training batches
	for i := 1; i <= 5; i++ {
		filename := filepath.Join(dataDir, fmt.Sprintf("data_batch_%d.bin", i))
		inputs, targets, err := loadBatch(filename)
		if err != nil {
			return nil, err
		}
		cifar.TrainInputs = append(cifar.TrainInputs, inputs...)
		cifar.TrainTargets = append(cifar.TrainTargets, targets...)
	}

	// Load test batch
	filename := filepath.Join(dataDir, "test_batch.bin")
	inputs, targets, err := loadBatch(filename)
	if err != nil {
		return nil, err
	}
	cifar.TestInputs = inputs
	cifar.TestTargets = targets

	return cifar, nil
}

func main() {
	// 1. Data Loading
	dataDir := "examples/cifar10/datasets"
	fmt.Println("Loading CIFAR-10 dataset...")
	dataset, err := LoadCIFAR10(dataDir)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}
	fmt.Printf("Loaded %d training images and %d test images\n", len(dataset.TrainInputs), len(dataset.TestInputs))

	// 2. Architecture Design
	// Conv2D (32 filters, 3x3) -> BatchNorm2D -> Conv2D (32 filters, 3x3) -> BatchNorm2D -> MaxPool2D (2x2)
	// Conv2D (64 filters, 3x3) -> BatchNorm2D -> Conv2D (64 filters, 3x3) -> BatchNorm2D -> MaxPool2D (2x2)
	// Flatten -> Dense (512) -> Dropout (0.5) -> Dense (10) -> LogSoftmax

	l0 := layer.NewConv2D(3, 32, 3, 1, 1, activations.ReLU{})
	bn0 := layer.NewBatchNorm2D(32, 1e-5, 0.1, true)
	l1 := layer.NewConv2D(32, 32, 3, 1, 1, activations.ReLU{})
	bn1 := layer.NewBatchNorm2D(32, 1e-5, 0.1, true)
	mp0 := layer.NewMaxPool2D(32, 2, 2, 0) // 32x32 -> 16x16

	l2 := layer.NewConv2D(32, 64, 3, 1, 1, activations.ReLU{})
	bn2 := layer.NewBatchNorm2D(64, 1e-5, 0.1, true)
	l3 := layer.NewConv2D(64, 64, 3, 1, 1, activations.ReLU{})
	bn3 := layer.NewBatchNorm2D(64, 1e-5, 0.1, true)
	mp1 := layer.NewMaxPool2D(64, 2, 2, 0) // 16x16 -> 8x8

	flatten := layer.NewFlatten()
	// Output of mp1 is 64 channels * 8 * 8 = 4096
	dense1 := layer.NewDense(4096, 512, activations.ReLU{})
	dropout := layer.NewDropout(0.5, 512)
	dense2 := layer.NewDense(512, 10, activations.LogSoftmax{})

	layers := []layer.Layer{
		l0, bn0, l1, bn1, mp0,
		l2, bn2, l3, bn3, mp1,
		flatten, dense1, dropout, dense2,
	}

	network := net.New(layers, loss.NLLLoss{}, opt.NewAdam(0.001))

	// Support for MetalDevice if available
	metal := layer.NewMetalDevice()
	if metal.IsAvailable() {
		fmt.Println("Using Metal device for acceleration")
		network.SetDevice(metal)
	} else {
		fmt.Println("Using CPU device")
	}

	// 3. Training Loop
	epochs := 10
	batchSize := 64
	fmt.Printf("Starting training for %d epochs (batch size %d)...\n", epochs, batchSize)

	for epoch := 0; epoch < epochs; epoch++ {
		start := time.Now()

		// Shuffle training data
		indices := rand.Perm(len(dataset.TrainInputs))
		totalLoss := float32(0)

		for i := 0; i < len(dataset.TrainInputs); i += batchSize {
			end := i + batchSize
			if end > len(dataset.TrainInputs) {
				end = len(dataset.TrainInputs)
			}

			batchX := make([][]float32, end-i)
			batchY := make([][]float32, end-i)
			for j := 0; j < end-i; j++ {
				batchX[j] = dataset.TrainInputs[indices[i+j]]
				batchY[j] = dataset.TrainTargets[indices[i+j]]
			}

			// Use training mode for dropout
			dropout.SetTraining(true)
			lossVal := network.TrainBatch(batchX, batchY)
			totalLoss += lossVal
		}

		duration := time.Since(start)
		avgLoss := totalLoss / float32(len(dataset.TrainInputs)/batchSize)

		// 4. Evaluation
		accuracy := evaluate(network, dataset.TestInputs, dataset.TestTargets, dropout)
		fmt.Printf("Epoch %d/%d - Loss: %.4f - Test Accuracy: %.2f%% - Duration: %v\n",
			epoch+1, epochs, avgLoss, accuracy*100, duration)
	}

	// Final Results
	fmt.Println("Training complete!")
}

func evaluate(network *net.Network, inputs, targets [][]float32, dropout *layer.Dropout) float32 {
	correct := 0
	dropout.SetTraining(false)

	for i := range inputs {
		pred := network.Forward(inputs[i])
		if argmax(pred) == argmax(targets[i]) {
			correct++
		}
	}

	return float32(correct) / float32(len(inputs))
}

func argmax(probs []float32) int {
	maxIdx := 0
	maxVal := probs[0]
	for i := 1; i < len(probs); i++ {
		if probs[i] > maxVal {
			maxVal = probs[i]
			maxIdx = i
		}
	}
	return maxIdx
}
