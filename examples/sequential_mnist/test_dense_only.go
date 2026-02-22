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

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
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
		return nil
	}
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

func loadMNISTImages(filename string) ([][]float64, int, int, error) {
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
	images := make([][]float64, numImages)
	pixelCount := int(rows * cols)
	for i := 0; i < int(numImages); i++ {
		pixels := make([]uint8, pixelCount)
		gz.Read(pixels)
		floatPixels := make([]float64, pixelCount)
		for j := 0; j < pixelCount; j++ {
			floatPixels[j] = float64(pixels[j]) / 255.0
		}
		images[i] = floatPixels
	}
	return images, int(rows), int(cols), nil
}

func loadMNISTLabels(filename string) ([][]float64, error) {
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
	labels := make([][]float64, numLabels)
	for i := 0; i < int(numLabels); i++ {
		var label uint8
		binary.Read(gz, binary.BigEndian, &label)
		oneHot := make([]float64, 10)
		oneHot[label] = 1.0
		labels[i] = oneHot
	}
	return labels, nil
}

func argmax(v []float64) int {
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

func main() {
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
	xTrain, rows, cols, err := loadMNISTImages(filepath.Join(destDir, trainImages))
	if err != nil {
		fmt.Printf("Error loading training images: %v\n", err)
		return
	}
	yTrain, err := loadMNISTLabels(filepath.Join(destDir, trainLabels))
	if err != nil {
		fmt.Printf("Error loading training labels: %v\n", err)
		return
	}
	xTest, _, _, err := loadMNISTImages(filepath.Join(destDir, testImages))
	if err != nil {
		fmt.Printf("Error loading test images: %v\n", err)
		return
	}
	yTest, err := loadMNISTLabels(filepath.Join(destDir, testLabels))
	if err != nil {
		fmt.Printf("Error loading test labels: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d training samples and %d test samples (%dx%d images)\n",
		len(xTrain), len(xTest), rows, cols)

	// Test with just Dense layers (no LSTM)
	layers := []layer.Layer{
		layer.NewDense(rows*cols, 128, activations.Tanh{}),
		layer.NewDense(128, 10, activations.LogSoftmax{}),
	}
	optimizer := &opt.SGD{LearningRate: 0.1}
	network := net.New(layers, loss.NLLLoss{}, optimizer)
	fmt.Println("\nStarting Dense-only training...")
	start := time.Now()
	callbacks := []net.Callback{net.Logger{Interval: 1}}
	network.Fit(xTrain, yTrain, 10, 32, callbacks...)
	fmt.Printf("\nTraining finished in %v\n", time.Since(start))
	correct := 0
	for i := 0; i < len(xTest); i++ {
		pred := network.Forward(xTest[i])
		if argmax(pred) == argmax(yTest[i]) {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(xTest))
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy*100)
}
