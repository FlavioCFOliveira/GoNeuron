package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

func main() {
	fmt.Println("=== Autoencoder for Synthetic Data Compression ===")
	rand.Seed(time.Now().UnixNano())

	// 1. Generate synthetic data (clusters in 10D space, mostly 2D)
	numSamples := 1000
	dim := 10
	data := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		data[i] = make([]float64, dim)
		// Intrinsic 2D information
		x := rand.Float64()
		y := rand.Float64()
		for j := 0; j < dim; j++ {
			// Each dimension is a linear combination of x, y plus noise
			data[i][j] = x*rand.Float64() + y*rand.Float64() + rand.Float64()*0.1
		}
	}

	// 2. Create architecture (10 -> 2 -> 10)
	layers := []layer.Layer{
		// Encoder
		layer.NewDense(10, 2, activations.ReLU{}),
		// Decoder
		layer.NewDense(2, 10, activations.Sigmoid{}), // Using Sigmoid assuming input normalized to [0,1]
	}

	optimizer := opt.NewAdam(0.01)
	network := net.New(layers, loss.MSE{}, optimizer)

	// 3. Train (self-supervised: target == input)
	epochs := 100
	fmt.Printf("Training for %d epochs...\n", epochs)
	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := 0.0
		for i := range data {
			totalLoss += network.Train(data[i], data[i])
		}
		if epoch%20 == 0 {
			fmt.Printf("Epoch %d - Avg Loss: %.6f\n", epoch, totalLoss/float64(numSamples))
		}
	}

	// 4. Demonstrate compression
	fmt.Println("\n=== Sample Reconstruction ===")
	sample := data[0]
	reconstructed := network.Forward(sample)

	fmt.Printf("Original:      %.3f\n", sample[:5])
	fmt.Printf("Reconstructed: %.3f\n", reconstructed[:5])

	// Latent space (encoder output)
	latent := layers[0].Forward(sample)
	fmt.Printf("Latent (2D):   %.3f\n", latent)
}
