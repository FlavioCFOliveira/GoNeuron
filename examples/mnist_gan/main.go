package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

const (
	mnistBaseURL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
	trainImages  = "train-images-idx3-ubyte.gz"
	noiseSize    = 100
	imgSize      = 784 // 28x28
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

	images := make([][]float32, numImages)
	pixelCount := int(rows * cols)
	for i := 0; i < int(numImages); i++ {
		pixels := make([]uint8, pixelCount)
		gz.Read(pixels)
		floatPixels := make([]float32, pixelCount)
		for j := 0; j < pixelCount; j++ {
			// Normalize to [-1, 1] for Tanh activation in Generator
			floatPixels[j] = (float32(pixels[j])/255.0)*2.0 - 1.0
		}
		images[i] = floatPixels
	}
	return images, nil
}

func saveGeneratedImage(pixels []float32, filename string) error {
	img := image.NewGray(image.Rect(0, 0, 28, 28))
	for i := 0; i < 784; i++ {
		// Denormalize from [-1, 1] to [0, 255]
		val := (pixels[i] + 1.0) / 2.0 * 255.0
		if val < 0 { val = 0 }
		if val > 255 { val = 255 }
		img.SetGray(i%28, i/28, color.Gray{Y: uint8(val)})
	}
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func main() {
	// 1. Setup Data
	destDir := "datasets"
	os.MkdirAll(destDir, 0755)
	trainImagesPath := filepath.Join(destDir, trainImages)
	if err := downloadFile(mnistBaseURL+trainImages, trainImagesPath); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	xTrain, err := loadMNISTImages(trainImagesPath)
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d training images\n", len(xTrain))

	// 2. Define Generator
	genNet := goneuron.NewSequential(
		goneuron.Dense(256, goneuron.LeakyReLU(0.2)),
		goneuron.Dense(512, goneuron.LeakyReLU(0.2)),
		goneuron.Dense(1024, goneuron.LeakyReLU(0.2)),
		goneuron.Dense(imgSize, goneuron.Tanh),
	)
	genNet.Compile(goneuron.Adam(0.0002), goneuron.MSE)
	genNet.Build(noiseSize)

	// 3. Define Discriminator
	discNet := goneuron.NewSequential(
		goneuron.Dense(512, goneuron.LeakyReLU(0.2)),
		goneuron.Dense(256, goneuron.LeakyReLU(0.2)),
		goneuron.Dense(1, goneuron.Linear), // Output logits
	)
	// Use BCEWithLogitsLoss for numerical stability
	discNet.Compile(goneuron.Adam(0.0002), goneuron.BCEWithLogitsLoss)
	discNet.Build(imgSize)

	// 4. Training Loop
	epochs := 50
	batchSize := 128
	numBatches := len(xTrain) / batchSize
	rng := layer.NewRNG(42)

	ones := []float32{1.0}
	zeros := []float32{0.0}

	fmt.Println("Starting GAN training...")
	os.MkdirAll("output", 0755)

	for epoch := 0; epoch < epochs; epoch++ {
		var epochDRealLoss, epochDFakeLoss, epochGLoss float32

		for b := 0; b < numBatches; b++ {
			// --- Train Discriminator ---
			discNet.ClearGradients()

			// 1. Real Images
			realIdx := int(rng.RandFloat() * float32(len(xTrain)))
			if realIdx >= len(xTrain) { realIdx = len(xTrain) - 1 }
			realImg := xTrain[realIdx]

			dRealPred := discNet.Forward(realImg)
			dRealLoss := discNet.Loss().Forward(dRealPred, ones)
			epochDRealLoss += dRealLoss

			dRealGrad := discNet.Loss().Backward(dRealPred, ones)
			discNet.Backward(dRealGrad)

			// 2. Fake Images
			noise := make([]float32, noiseSize)
			for i := range noise {
				noise[i] = rng.RandFloat()*2.0 - 1.0 // Latent space [-1, 1]
			}
			fakeImg := genNet.Forward(noise)

			dFakePred := discNet.Forward(fakeImg)
			dFakeLoss := discNet.Loss().Forward(dFakePred, zeros)
			epochDFakeLoss += dFakeLoss

			dFakeGrad := discNet.Loss().Backward(dFakePred, zeros)
			discNet.Backward(dFakeGrad)

			// Update Discriminator weights
			discNet.Step()

			// --- Train Generator ---
			genNet.ClearGradients()
			discNet.ClearGradients() // Don't let D grads interfere

			// We want D(G(z)) to be 1
			fakeImgForG := genNet.Forward(noise)
			gDiscPred := discNet.Forward(fakeImgForG)
			gLoss := discNet.Loss().Forward(gDiscPred, ones)
			epochGLoss += gLoss

			// Backward through D to get gradients w.r.t input (fake image)
			gDiscGrad := discNet.Loss().Backward(gDiscPred, ones)
			gInputGrad := discNet.Backward(gDiscGrad)

			// Backward through G using gradients from D
			genNet.Backward(gInputGrad)

			// Update Generator weights
			genNet.Step()

			// Clean up Discriminator gradients (we don't want to update D here)
			discNet.ClearGradients()
		}

		fmt.Printf("Epoch %d/%d | D Real Loss: %.4f | D Fake Loss: %.4f | G Loss: %.4f\n",
			epoch+1, epochs, epochDRealLoss/float32(numBatches),
			epochDFakeLoss/float32(numBatches), epochGLoss/float32(numBatches))

		// Save a sample image every epoch
		noise := make([]float32, noiseSize)
		for i := range noise {
			noise[i] = rng.RandFloat()*2.0 - 1.0
		}
		sample := genNet.Forward(noise)
		saveGeneratedImage(sample, fmt.Sprintf("output/epoch_%d.png", epoch+1))
	}

	fmt.Println("Training complete. Check output/ directory for generated images.")
}
