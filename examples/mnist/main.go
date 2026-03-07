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
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read magic number: %w", err)
	}
	if err := binary.Read(gz, binary.BigEndian, &numImages); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read number of images: %w", err)
	}
	if err := binary.Read(gz, binary.BigEndian, &rows); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read rows: %w", err)
	}
	if err := binary.Read(gz, binary.BigEndian, &cols); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read columns: %w", err)
	}

	if magic != 2051 {
		return nil, 0, 0, fmt.Errorf("invalid magic number: %d", magic)
	}

	// Validate dimensions
	if rows <= 0 || cols <= 0 || numImages <= 0 {
		return nil, 0, 0, fmt.Errorf("invalid dimensions: images=%d, rows=%d, cols=%d", numImages, rows, cols)
	}
	if rows > 10000 || cols > 10000 || numImages > 1000000 {
		return nil, 0, 0, fmt.Errorf("dimensions exceed maximum: images=%d, rows=%d, cols=%d", numImages, rows, cols)
	}

	// Pre-allocate contiguous buffer for all images (zero-allocation pattern)
	pixelCount := int(rows * cols)
	totalPixels := int(numImages) * pixelCount
	buffer := make([]float32, totalPixels)
	images := make([][]float32, numImages)

	for i := 0; i < int(numImages); i++ {
		pixels := make([]uint8, pixelCount)
		n, err := io.ReadFull(gz, pixels)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("failed to read image %d: %w", i, err)
		}
		if n != pixelCount {
			return nil, 0, 0, fmt.Errorf("incomplete read for image %d: got %d bytes, expected %d", i, n, pixelCount)
		}
		// Slice into pre-allocated buffer
		floatPixels := buffer[i*pixelCount : (i+1)*pixelCount]
		for j := 0; j < pixelCount; j++ {
			floatPixels[j] = float32(pixels[j]) / 255.0
		}
		images[i] = floatPixels
	}

	// Verify no extra data remains after reading all images
	var extraByte [1]byte
	n, err := gz.Read(extraByte[:])
	if err != io.EOF {
		if err != nil {
			return nil, 0, 0, fmt.Errorf("unexpected error after reading images: %w", err)
		}
		if n > 0 {
			return nil, 0, 0, fmt.Errorf("extra data found after reading %d images", numImages)
		}
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
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic number: %w", err)
	}
	if err := binary.Read(gz, binary.BigEndian, &numLabels); err != nil {
		return nil, fmt.Errorf("failed to read number of labels: %w", err)
	}

	if magic != 2049 {
		return nil, fmt.Errorf("invalid magic number for labels: expected 2049, got %d", magic)
	}

	// Validate number of labels
	if numLabels <= 0 {
		return nil, fmt.Errorf("invalid number of labels: %d", numLabels)
	}
	if numLabels > 1000000 {
		return nil, fmt.Errorf("number of labels exceeds maximum: %d", numLabels)
	}

	labels := make([][]float32, numLabels)
	for i := 0; i < int(numLabels); i++ {
		var label uint8
		if err := binary.Read(gz, binary.BigEndian, &label); err != nil {
			return nil, fmt.Errorf("failed to read label %d: %w", i, err)
		}
		// Validate label is in valid range [0, 9] for MNIST
		if label > 9 {
			return nil, fmt.Errorf("invalid label value at index %d: %d (must be in range [0, 9])", i, label)
		}
		oneHot := make([]float32, 10)
		oneHot[label] = 1.0
		labels[i] = oneHot
	}

	// Verify no extra data remains (optional check for file integrity)
	var extraByte uint8
	if err := binary.Read(gz, binary.BigEndian, &extraByte); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("extra data found after reading %d labels", numLabels)
		}
		// EOF is expected, any other error is problematic
		if err != io.EOF {
			return nil, fmt.Errorf("unexpected error after reading labels: %w", err)
		}
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
	device := &layer.CPUDevice{}
	fmt.Println("Using CPU for training")
	model.SetDevice(device)

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
	predictions, err := model.PredictBatch(xTest)
	if err != nil {
		fmt.Printf("Error during prediction: %v\n", err)
		return
	}
	correct := 0
	for i := 0; i < len(predictions); i++ {
		if argmax(predictions[i]) == argmax(yTest[i]) {
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
	if len(v) == 0 {
		return 0
	}
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
