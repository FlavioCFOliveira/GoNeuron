package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math/rand"
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
	noiseSize    = 100
	imgSize      = 784
)

func downloadFile(url, dest string) error {
	if _, err := os.Stat(dest); err == nil {
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

func loadMNISTImages(filename string) ([][]float32, error) {
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
	var magic, numImages, rows, cols int32
	binary.Read(gz, binary.BigEndian, &magic)
	binary.Read(gz, binary.BigEndian, &numImages)
	binary.Read(gz, binary.BigEndian, &rows)
	binary.Read(gz, binary.BigEndian, &cols)
	if magic != 2051 {
		return nil, fmt.Errorf("invalid magic number: %d", magic)
	}
	images := make([][]float32, numImages)
	pixelCount := int(rows * cols)
	for i := 0; i < int(numImages); i++ {
		pixels := make([]uint8, pixelCount)
		gz.Read(pixels)
		floatPixels := make([]float32, pixelCount)
		for j := 0; j < pixelCount; j++ {
			// Normalize to [-1, 1] to match Tanh output
			floatPixels[j] = (float32(pixels[j])/255.0)*2.0 - 1.0
		}
		images[i] = floatPixels
	}
	return images, nil
}

func saveImage(pixels []float32, filename string) error {
	img := image.NewGray(image.Rect(0, 0, 28, 28))
	for i := 0; i < 784; i++ {
		// Denormalize from [-1, 1] to [0, 255]
		val := (pixels[i] + 1.0) / 2.0 * 255.0
		if val < 0 {
			val = 0
		}
		if val > 255 {
			val = 255
		}
		img.SetGray(i%28, i/28, color.Gray{Y: uint8(val)})
	}
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func generateNoise(batchSize int) [][]float32 {
	noise := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		n := make([]float32, noiseSize)
		for j := 0; j < noiseSize; j++ {
			n[j] = rand.Float32()*2.0 - 1.0
		}
		noise[i] = n
	}
	return noise
}

func main() {
	rand.Seed(time.Now().UnixNano())
	device := layer.GetDefaultDevice()
	fmt.Printf("Using device: %T\n", device)

	// 1. Load Data
	destDir := filepath.Join("examples", "gan_mnist", "datasets")
	os.MkdirAll(destDir, 0755)
	trainPath := filepath.Join(destDir, trainImages)
	if err := downloadFile(mnistBaseURL+trainImages, trainPath); err != nil {
		fmt.Printf("Error downloading: %v\n", err)
		return
	}
	xTrain, err := loadMNISTImages(trainPath)
	if err != nil {
		fmt.Printf("Error loading: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d training samples\n", len(xTrain))

	// 2. Define Generator using High-Level API
	gen := goneuron.NewSequential(
		goneuron.Dense(noiseSize, 256, goneuron.ReLU),
		goneuron.Dense(256, 512, goneuron.ReLU),
		goneuron.Dense(512, 1024, goneuron.ReLU),
		goneuron.Dense(1024, imgSize, goneuron.Tanh),
	)
	gen.Compile(goneuron.Adam(0.0002), goneuron.MSE) // Loss not used directly for G
	gen.SetDevice(device)

	// 3. Define Discriminator using High-Level API
	disc := goneuron.NewSequential(
		goneuron.Dense(imgSize, 1024, goneuron.LeakyReLU(0.2)),
		goneuron.Dropout(0.3, 1024),
		goneuron.Dense(1024, 512, goneuron.LeakyReLU(0.2)),
		goneuron.Dropout(0.3, 512),
		goneuron.Dense(512, 256, goneuron.LeakyReLU(0.2)),
		goneuron.Dense(256, 1, goneuron.Sigmoid),
	)
	disc.Compile(goneuron.Adam(0.0002), goneuron.BCELoss)
	disc.SetDevice(device)

	// 4. Training loop
	epochs := 100
	batchSize := 128
	numBatches := len(xTrain) / batchSize

	realLabels := []float32{1.0}
	fakeLabels := []float32{0.0}

	fmt.Println("Starting GAN training (High-Level API)...")
	for epoch := 0; epoch < epochs; epoch++ {
		start := time.Now()
		totalDLoss := float32(0)
		totalGLoss := float32(0)

		for b := 0; b < numBatches; b++ {
			// --- Train Discriminator ---
			disc.ClearGradients()

			// Real images
			for i := 0; i < batchSize; i++ {
				idx := rand.Intn(len(xTrain))
				yPred := disc.Predict(xTrain[idx])
				totalDLoss += disc.Loss().Forward(yPred, realLabels)
				grad := disc.Loss().Backward(yPred, realLabels)
				disc.Backward(grad)
			}

			// Fake images
			noise := generateNoise(batchSize)
			for i := 0; i < batchSize; i++ {
				fakeImg := gen.Predict(noise[i])
				yPred := disc.Predict(fakeImg)
				totalDLoss += disc.Loss().Forward(yPred, fakeLabels)
				grad := disc.Loss().Backward(yPred, fakeLabels)
				disc.Backward(grad)
			}

			// Average gradients and step
			for _, l := range disc.Layers() {
				grads := l.Gradients()
				for i := range grads {
					grads[i] /= float32(batchSize * 2)
				}
				l.SetGradients(grads)
			}
			disc.Step()

			// --- Train Generator ---
			gen.ClearGradients()
			noise = generateNoise(batchSize)
			for i := 0; i < batchSize; i++ {
				fakeImg := gen.Predict(noise[i])
				yPred := disc.Predict(fakeImg) // D(G(noise))

				totalGLoss += disc.Loss().Forward(yPred, realLabels)
				grad := disc.Loss().Backward(yPred, realLabels)

				// Backprop through Discriminator to get grad w.r.t fakeImg
				gradFakeImg := disc.Backward(grad)
				gen.Backward(gradFakeImg)
			}

			// Average gradients and step
			for _, l := range gen.Layers() {
				grads := l.Gradients()
				for i := range grads {
					grads[i] /= float32(batchSize)
				}
				l.SetGradients(grads)
			}
			gen.Step()
		}

		fmt.Printf("Epoch %d/%d - D Loss: %.4f, G Loss: %.4f - %v\n",
			epoch+1, epochs, totalDLoss/float32(numBatches*batchSize*2),
			totalGLoss/float32(numBatches*batchSize), time.Since(start))

		// Save sample images
		if (epoch+1)%10 == 0 || epoch == 0 {
			noise := generateNoise(1)
			sample := gen.Predict(noise[0])
			samplePath := filepath.Join("examples", "gan_mnist", fmt.Sprintf("sample_epoch_%d.png", epoch+1))
			saveImage(sample, samplePath)
			fmt.Printf("Saved sample to %s\n", samplePath)
		}
	}

	fmt.Println("Training complete.")
}
