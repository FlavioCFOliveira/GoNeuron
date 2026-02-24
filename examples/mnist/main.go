package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
)

const (
	mnistBaseURL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
	trainImages  = "train-images-idx3-ubyte.gz"
	trainLabels  = "train-labels-idx1-ubyte.gz"
	testImages   = "t10k-images-idx3-ubyte.gz"
	testLabels   = "t10k-labels-idx1-ubyte.gz"
)

func downloadFile(url, dest string) error {
	if _, err := os.Stat(dest); err == nil {
		fmt.Printf("File already exists: %s\n", dest)
		return nil
	}

	fmt.Printf("Downloading %s...\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func loadMNISTImages(filename string) ([][]float32, int, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, 0, 0, err
	}
	defer gz.Close()

	var magic, numImages, rows, cols int32
	binary.Read(gz, binary.BigEndian, &magic)
	binary.Read(gz, binary.BigEndian, &numImages)
	binary.Read(gz, binary.BigEndian, &rows)
	binary.Read(gz, binary.BigEndian, &cols)

	if magic != 2051 {
		return nil, 0, 0, fmt.Errorf("invalid magic number: %d", magic)
	}

	images := make([][]float32, numImages)
	pixelCount := int(rows * cols)
	for i := 0; i < int(numImages); i++ {
		pixels := make([]uint8, pixelCount)
		gz.Read(pixels)
		floatPixels := make([]float32, pixelCount)
		for j := 0; j < pixelCount; j++ {
			floatPixels[j] = float32(pixels[j]) / 255.0
		}
		images[i] = floatPixels
	}

	return images, int(rows), int(cols), nil
}

func loadMNISTLabels(filename string) ([][]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var magic, numLabels int32
	binary.Read(gz, binary.BigEndian, &magic)
	binary.Read(gz, binary.BigEndian, &numLabels)

	if magic != 2049 {
		return nil, fmt.Errorf("invalid magic number: %d", magic)
	}

	labels := make([][]float32, numLabels)
	for i := 0; i < int(numLabels); i++ {
		var label uint8
		binary.Read(gz, binary.BigEndian, &label)
		oneHot := make([]float32, 10)
		oneHot[label] = 1.0
		labels[i] = oneHot
	}

	return labels, nil
}

func main() {
	// 1. Download/Load Data
	fmt.Println("Checking MNIST dataset...")
	destDir := "datasets"
	os.MkdirAll(destDir, 0755)

	files := []string{trainImages, trainLabels, testImages, testLabels}
	for _, f := range files {
		url := mnistBaseURL + f
		dest := filepath.Join(destDir, f)
		if err := downloadFile(url, dest); err != nil {
			fmt.Printf("Error downloading %s: %v\n", f, err)
			os.Exit(1)
		}
	}

	trainImagesPath := filepath.Join(destDir, trainImages)
	trainLabelsPath := filepath.Join(destDir, trainLabels)
	testImagesPath := filepath.Join(destDir, testImages)
	testLabelsPath := filepath.Join(destDir, testLabels)

	xTrain, rows, cols, err := loadMNISTImages(trainImagesPath)
	if err != nil {
		fmt.Printf("Error loading training images: %v\n", err)
		return
	}
	yTrain, err := loadMNISTLabels(trainLabelsPath)
	if err != nil {
		fmt.Printf("Error loading training labels: %v\n", err)
		return
	}

	xTest, _, _, err := loadMNISTImages(testImagesPath)
	if err != nil {
		fmt.Printf("Error loading test images: %v\n", err)
		return
	}
	yTest, err := loadMNISTLabels(testLabelsPath)
	if err != nil {
		fmt.Printf("Error loading test labels: %v\n", err)
		return
	}

	fmt.Printf("Loaded %d training samples and %d test samples (%dx%d images)\n", len(xTrain), len(xTest), rows, cols)

	// 2. Define Architecture using High-Level API
	model := goneuron.NewSequential(
		goneuron.Conv2D(16, 3, 1, 1, goneuron.ReLU), // Inferred inChannels=1
		goneuron.MaxPool2D(2, 2, 0),
		goneuron.Conv2D(32, 3, 1, 1, goneuron.ReLU),
		goneuron.MaxPool2D(2, 2, 0),
		goneuron.Flatten(),
		goneuron.Dense(128, goneuron.ReLU),
		goneuron.Dropout(0.2, 128),
		goneuron.Dense(10, goneuron.LogSoftmax),
	)

	// Set input dimensions for the first layer (MNIST is 1x28x28)
	if l, ok := model.Layers()[0].(interface{ SetInputDimensions(int, int) }); ok {
		l.SetInputDimensions(28, 28)
	}
	// Initializing model with 1*28*28 input
	model.Build(1 * 28 * 28)

	// 3. Compile Model
	device := layer.NewMetalDevice()
	if device.IsAvailable() {
		fmt.Println("Using Metal acceleration")
		model.SetDevice(device)
	}

	optimizer := goneuron.Adam(0.001)
	model.Compile(optimizer, goneuron.NLLLoss)

	// 4. Training
	fmt.Println("Starting training...")
	model.Summary()
	start := time.Now()

	scheduler := goneuron.ReduceLROnPlateau(optimizer, 0.5, 1, 0.01, 1e-6)

	callbacks := []goneuron.Callback{
		goneuron.Logger(1),
		goneuron.ModelCheckpoint("mnist_best.gob"),
		goneuron.EarlyStopping(3, 0.001),
		goneuron.SchedulerCallback(scheduler),
	}

	model.Fit(xTrain, yTrain, 10, 64, callbacks...)
	fmt.Printf("Training finished in %v\n", time.Since(start))

	// 5. Evaluation
	fmt.Println("\nEvaluating on test set...")
	correct := 0
	for i := 0; i < len(xTest); i++ {
		pred := model.Predict(xTest[i])
		if argmax(pred) == argmax(yTest[i]) {
			correct++
		}
	}

	accuracy := float32(correct) / float32(len(xTest))
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy*100)

	// Save final model
	model.Save("mnist_final.gob")
	fmt.Println("Model saved to mnist_final.gob")
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
