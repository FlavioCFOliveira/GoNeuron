# GAN for MNIST - Architecture and Results

This example implements a Generative Adversarial Network (GAN) to generate MNIST-like digits.

## Architecture Overview

### Generator
The generator takes a noise vector of size 100 (sampled from a uniform distribution [-1, 1]) and transforms it into a 28x28 image (784 pixels).
- **Input**: 100 (Noise)
- **Layer 1**: Dense (100 -> 256), ReLU
- **Layer 2**: Dense (256 -> 512), ReLU
- **Layer 3**: Dense (512 -> 1024), ReLU
- **Layer 4**: Dense (1024 -> 784), Tanh

The output pixels are in the range [-1, 1], which matches the normalized MNIST data.

### Discriminator
The discriminator is a binary classifier that distinguishes between real MNIST images and fake images produced by the generator.
- **Input**: 784 (Image)
- **Layer 1**: Dense (784 -> 1024), LeakyReLU (alpha=0.2)
- **Layer 2**: Dropout (0.3)
- **Layer 3**: Dense (1024 -> 512), LeakyReLU (alpha=0.2)
- **Layer 4**: Dropout (0.3)
- **Layer 5**: Dense (512 -> 256), LeakyReLU (alpha=0.2)
- **Layer 6**: Dense (256 -> 1), Sigmoid

## Dataset Description
- **Source**: MNIST handwritten digits.
- **Pre-processing**:
  - Pixels normalized from [0, 255] to [0, 1].
  - Further normalized to [-1, 1] using `(x * 2.0) - 1.0` to match the Generator's Tanh output.

## Training Procedure
- **Optimizer**: Adam (learning rate = 0.0002) for both networks.
- **Loss Function**: Binary Cross Entropy (BCE).
- **Batch Size**: 128.
- **Steps per Batch**:
  1. Train Discriminator:
     - Real images with label 1.0.
     - Fake images with label 0.0.
     - Gradients are averaged over the batch (2 * batchSize).
  2. Train Generator:
     - Fake images with label 1.0 (to fool the discriminator).
     - Gradients are backpropagated through the Discriminator to the Generator.
     - Only Generator weights are updated.

## Results and Performance

### Training Results
- **Epochs**: 100
- **Convergence**: The loss for both Generator and Discriminator typically stabilizes, with the Generator's loss decreasing as it learns to produce more realistic images.
- **Visual Progress**: Samples saved every 10 epochs show the transition from random noise to recognizable digit shapes.

### Performance
- **Hardware Acceleration**: Uses Metal/MPS on macOS (Darwin/arm64) via `MetalDevice`.
- **Zero-Allocation**: Leverages pre-allocated buffers in `Dense` and `Dropout` layers for high-speed training.
- **Inference Latency**: Sub-millisecond generation time per image.

## Benchmarking
The implementation uses optimized nested loops for CPU fallback and Metal kernels for GPU acceleration, ensuring efficient execution even for deep architectures like this GAN.
