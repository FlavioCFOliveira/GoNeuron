# MNIST GAN with Binary Cross-Entropy

An alternative implementation of the MNIST GAN, focusing on precise loss formulation and visualization of the generative process.

## Technical Objectives

1.  **BCE Loss Optimization**: Utilizing Binary Cross-Entropy for both the discriminator and the generator objectives.
2.  **Latent Space Exploration**: Demonstrating how random noise vectors $z$ evolve into coherent digit representations.
3.  **Visualization Pipeline**: Automated saving of synthetic samples for qualitative assessment.

## Key Implementation Details

### 1. Loss Formulation
```go
criterion := goneuron.BCELoss
// ... manual loop updates for D and G ...
```
*   **Design**: Highlights the flexibility of the GoNeuron core in supporting manual training loops beyond the standard `Fit` method, which is necessary for architectures where multiple models interact.

### 2. Generative Quality
The model utilizes `LeakyReLU` activations in both networks to maintain gradient flow even for "fake" samples that the discriminator easily identifies, preventing generator saturation.

## Execution

```bash
go run main.go
```
