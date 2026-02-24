package main

import (
	"fmt"
	"math/rand"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

func main() {
	fmt.Println("=== Autoencoder for Synthetic Data Compression ===")

	// 1. Generate synthetic data (clusters in 10D space, mostly 2D)
	numSamples := 1000
	dim := 10
	data := make([][]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		data[i] = make([]float32, dim)
		// Intrinsic 2D information
		x := rand.Float32()
		y := rand.Float32()
		for j := 0; j < dim; j++ {
			// Each dimension is a linear combination of x, y plus noise
			data[i][j] = x*rand.Float32() + y*rand.Float32() + rand.Float32()*0.1
		}
	}

	// 2. Create architecture (10 -> 2 -> 10)
	model := goneuron.NewSequential(
		// Encoder
		goneuron.Dense(2, goneuron.ReLU),
		// Decoder
		goneuron.Dense(10, goneuron.Sigmoid), // Using Sigmoid assuming input normalized to [0,1]
	)

	model.Compile(goneuron.Adam(0.01), goneuron.MSE)

	// 3. Train (self-supervised: target == input)
	epochs := 100
	fmt.Printf("Training for %d epochs...\n", epochs)
	model.Fit(data, data, epochs, 32, goneuron.Logger(20))

	// 4. Demonstrate compression
	fmt.Println("\n=== Sample Reconstruction ===")
	sample := data[0]
	reconstructed := model.Forward(sample)

	fmt.Printf("Original:      %.3f\n", sample[:5])
	fmt.Printf("Reconstructed: %.3f\n", reconstructed[:5])

	// Latent space (encoder output)
	latent := model.Layers()[0].Forward(sample)
	fmt.Printf("Latent (2D):   %.3f\n", latent)
}
