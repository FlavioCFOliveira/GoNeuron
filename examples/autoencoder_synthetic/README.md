# Autoencoder for Synthetic Data Compression

This example demonstrates a simple Autoencoder architecture for data compression and reconstruction.

## Architecture
- **Encoder**: Dense(10 -> 2) with ReLU activation.
- **Decoder**: Dense(2 -> 10) with Sigmoid activation.

## How it works
The Autoencoder learns a compressed representation (latent space) of the input data by attempting to reconstruct the input at its output. In this example, 10-dimensional synthetic data is compressed into a 2-dimensional representation and then reconstructed back to 10 dimensions.

## Running the example
```bash
go run examples/autoencoder_synthetic/main.go
```
