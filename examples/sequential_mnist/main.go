// Sequential MNIST example: Treats each image as a sequence of rows
// and uses LSTM to classify digits based on the sequential pattern.
package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
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

	pixelCount := int(rows * cols)
	totalPixels := int(numImages) * pixelCount

	// Pre-allocate contiguous buffer for zero-allocation pattern
	buffer := make([]float32, totalPixels)
	images := make([][]float32, numImages)

	// Temporary buffer for reading raw pixels
	pixels := make([]uint8, pixelCount)

	for i := 0; i < int(numImages); i++ {
		if _, err := gz.Read(pixels); err != nil {
			return nil, 0, 0, fmt.Errorf("failed to read image %d: %w", i, err)
		}

		// Create slice view into contiguous buffer
		floatPixels := buffer[i*pixelCount : (i+1)*pixelCount]
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
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic number: %w", err)
	}
	if err := binary.Read(gz, binary.BigEndian, &numLabels); err != nil {
		return nil, fmt.Errorf("failed to read number of labels: %w", err)
	}

	if magic != 2049 {
		return nil, fmt.Errorf("invalid magic number: %d", magic)
	}

	const numClasses = 10
	totalElements := int(numLabels) * numClasses

	// Pre-allocate contiguous buffer for zero-allocation pattern
	buffer := make([]float32, totalElements)
	labels := make([][]float32, numLabels)

	for i := 0; i < int(numLabels); i++ {
		var label uint8
		if err := binary.Read(gz, binary.BigEndian, &label); err != nil {
			return nil, fmt.Errorf("failed to read label %d: %w", i, err)
		}

		// Create slice view into contiguous buffer
		oneHot := buffer[i*numClasses : (i+1)*numClasses]
		if label < numClasses {
			oneHot[label] = 1.0
		}
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

	fmt.Printf("Loaded %d training samples and %d test samples (%dx%d images)\n",
		len(xTrain), len(xTest), rows, cols)

	// Shuffle training data
	rand.Shuffle(len(xTrain), func(i, j int) {
		xTrain[i], xTrain[j] = xTrain[j], xTrain[i]
		yTrain[i], yTrain[j] = yTrain[j], yTrain[i]
	})

	// 2. Define Architecture using High-Level API
	lstmUnits := 64
	model := goneuron.NewSequential(
		goneuron.SequenceUnroller(goneuron.LSTM(lstmUnits, cols), rows, false),
		goneuron.Dense(lstmUnits, 128, goneuron.Tanh),
		goneuron.Dense(128, 10, goneuron.LogSoftmax),
	)

	// 3. Compile Model
	optimizer := goneuron.Adam(0.001)
	model.Compile(optimizer, goneuron.NLLLoss)

	// 4. Training
	fmt.Println("\nStarting Sequential MNIST training...")
	model.Summary()

	start := time.Now()
	scheduler := goneuron.ReduceLROnPlateau(optimizer, 0.5, 2, 0.01, 1e-8)

	callbacks := []goneuron.Callback{
		goneuron.Logger(1),
		goneuron.ModelCheckpoint("sequential_mnist_best.gob"),
		goneuron.EarlyStopping(5, 0.001),
		goneuron.SchedulerCallback(scheduler),
	}

	model.Fit(xTrain, yTrain, 1000, 64, callbacks...)
	fmt.Printf("\nTraining finished in %v\n", time.Since(start))

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
	model.Save("sequential_mnist_final.gob")
	fmt.Println("Model saved to sequential_mnist_final.gob")
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
