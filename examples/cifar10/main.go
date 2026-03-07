package main

import (
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
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

	// Pre-allocate contiguous buffers for zero-allocation pattern
	// Each batch has 10000 images
	const batchSize = 10000
	inputBuffer := make([]float32, 0, batchSize*ImageSize*ImageSize*Channels)
	targetBuffer := make([]float32, 0, batchSize*NumClasses)
	inputViews := make([][]float32, 0, batchSize)
	targetViews := make([][]float32, 0, batchSize)

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
		// Validate label is in valid range [0, 9] for CIFAR-10
		if label < 0 || label >= NumClasses {
			return nil, nil, fmt.Errorf("invalid label value at record %d: %d (must be in range [0, %d])", len(inputViews)+1, label, NumClasses-1)
		}

		// Append target to contiguous buffer and create view
		targetStart := len(targetBuffer)
		targetBuffer = append(targetBuffer, make([]float32, NumClasses)...)
		targetBuffer[targetStart+label] = 1.0
		targetViews = append(targetViews, targetBuffer[targetStart:targetStart+NumClasses])

		// Append input to contiguous buffer and create view
		inputStart := len(inputBuffer)
		inputBuffer = append(inputBuffer, make([]float32, ImageSize*ImageSize*Channels)...)
		for i := 0; i < ImageSize*ImageSize*Channels; i++ {
			// Normalize to [0, 1]
			inputBuffer[inputStart+i] = float32(buf[i+1]) / 255.0
		}
		inputViews = append(inputViews, inputBuffer[inputStart:inputStart+ImageSize*ImageSize*Channels])
	}

	return inputViews, targetViews, nil
}

func LoadCIFAR10(dataDir string) (*CIFAR10, error) {
	cifar := &CIFAR10{}

	// Load training batches (each should have 10000 images)
	for i := 1; i <= 5; i++ {
		filename := filepath.Join(dataDir, fmt.Sprintf("data_batch_%d.bin", i))
		inputs, targets, err := loadBatch(filename)
		if err != nil {
			return nil, fmt.Errorf("failed to load training batch %d: %w", i, err)
		}
		// Validate batch size: each training batch should have 10000 images
		if len(inputs) != 10000 {
			return nil, fmt.Errorf("training batch %d has invalid size: expected 10000 images, got %d", i, len(inputs))
		}
		// Validate that inputs and targets match
		if len(targets) != len(inputs) {
			return nil, fmt.Errorf("training batch %d: mismatch between inputs (%d) and targets (%d)", i, len(inputs), len(targets))
		}
		cifar.TrainInputs = append(cifar.TrainInputs, inputs...)
		cifar.TrainTargets = append(cifar.TrainTargets, targets...)
	}

	// Load test batch (should have 10000 images)
	filename := filepath.Join(dataDir, "test_batch.bin")
	inputs, targets, err := loadBatch(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to load test batch: %w", err)
	}
	// Validate test batch size
	if len(inputs) != 10000 {
		return nil, fmt.Errorf("test batch has invalid size: expected 10000 images, got %d", len(inputs))
	}
	if len(targets) != len(inputs) {
		return nil, fmt.Errorf("test batch: mismatch between inputs (%d) and targets (%d)", len(inputs), len(targets))
	}
	cifar.TestInputs = inputs
	cifar.TestTargets = targets

	// Final validation: ensure total training images is 50000
	if len(cifar.TrainInputs) != TrainImages {
		return nil, fmt.Errorf("total training images mismatch: expected %d, got %d", TrainImages, len(cifar.TrainInputs))
	}
	if len(cifar.TrainTargets) != TrainImages {
		return nil, fmt.Errorf("total training targets mismatch: expected %d, got %d", TrainImages, len(cifar.TrainTargets))
	}

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

	// 2. Architecture Design
	model := goneuron.NewSequential(
		goneuron.Conv2D(32, 3, 1, 1, goneuron.ReLU),
		goneuron.BatchNorm2D(),
		goneuron.Conv2D(32, 3, 1, 1, goneuron.ReLU),
		goneuron.BatchNorm2D(),
		goneuron.MaxPool2D(2, 2, 0), // 32x32 -> 16x16

		goneuron.Conv2D(64, 3, 1, 1, goneuron.ReLU),
		goneuron.BatchNorm2D(),
		goneuron.Conv2D(64, 3, 1, 1, goneuron.ReLU),
		goneuron.BatchNorm2D(),
		goneuron.MaxPool2D(2, 2, 0), // 16x16 -> 8x8

		goneuron.Flatten(),
		goneuron.Dense(512, goneuron.ReLU),
		goneuron.Dropout(0.5, 512),
		goneuron.Dense(10, goneuron.LogSoftmax),
	)

	// Set input dimensions for the first layer (CIFAR-10 is 3x32x32)
	if l, ok := model.Layers()[0].(interface{ SetInputDimensions(int, int) }); ok {
		l.SetInputDimensions(ImageSize, ImageSize)
	}
	model.Build(Channels * ImageSize * ImageSize)

	model.Compile(goneuron.Adam(0.001), goneuron.NLLLoss)

	fmt.Println("Using CPU device")

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
			model.SetTraining(true)
			lossVal := model.TrainBatch(batchX, batchY)
			totalLoss += lossVal
		}

		duration := time.Since(start)
		avgLoss := totalLoss / float32(len(dataset.TrainInputs)/batchSize)

		// 4. Evaluation
		accuracy := evaluate(model, dataset.TestInputs, dataset.TestTargets)
		fmt.Printf("Epoch %d/%d - Loss: %.4f - Test Accuracy: %.2f%% - Duration: %v\n",
			epoch+1, epochs, avgLoss, accuracy*100, duration)
	}

	// Final Results
	fmt.Println("Training complete!")
}

func evaluate(model *goneuron.Model, inputs, targets [][]float32) float32 {
	model.SetTraining(false)

	predictions, err := model.PredictBatch(inputs)
	if err != nil {
		fmt.Printf("Error during prediction: %v\n", err)
		return 0
	}

	correct := 0
	for i := 0; i < len(predictions); i++ {
		if argmax(predictions[i]) == argmax(targets[i]) {
			correct++
		}
	}

	return float32(correct) / float32(len(inputs))
}

func argmax(probs []float32) int {
	if len(probs) == 0 {
		return 0
	}
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
