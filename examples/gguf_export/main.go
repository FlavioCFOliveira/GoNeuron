package main

import (
	"fmt"
	"os"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
)

func main() {
	fmt.Println("GGUF Export Demonstration")
	fmt.Println("-------------------------")

	// 1. Create a simple model
	// This model represents a small MLP that could be part of a larger system
	model := goneuron.NewSequential(
		goneuron.Dense(10, 20, goneuron.ReLU),
		goneuron.Dense(20, 5, goneuron.Softmax),
	)

	// 2. Compile (required to initialize buffers if not training)
	model.Compile(goneuron.SGD(0.1), goneuron.MSE)

	// 3. Optional: Build with a sample input size if using lazy inference
	// In this case, we already provided input sizes (10 and 20)

	filenameF32 := "model_f32.gguf"
	filenameF16 := "model_f16.gguf"

	// 4. Save as standard GGUF (Float32)
	fmt.Printf("Saving model to %s...\n", filenameF32)
	if err := model.SaveGGUF(filenameF32); err != nil {
		fmt.Printf("Error saving GGUF: %v\n", err)
		return
	}

	// 5. Save as compressed GGUF (Float16)
	// This uses internal/net constants for specifying the tensor type
	fmt.Printf("Saving model to %s (F16)...\n", filenameF16)
	if err := model.SaveGGUFExt(filenameF16, net.GGMLTypeF16); err != nil {
		fmt.Printf("Error saving GGUF F16: %v\n", err)
		return
	}

	// 6. Verify files
	for _, f := range []string{filenameF32, filenameF16} {
		info, err := os.Stat(f)
		if err != nil {
			fmt.Printf("Error stating file %s: %v\n", f, err)
			continue
		}
		fmt.Printf("File: %-15s Size: %d bytes\n", f, info.Size())
	}

	fmt.Println("\nSuccess! You can now use these files with tools compatible with GGUF v3.")

	// Clean up for this demo (optional)
	// os.Remove(filenameF32)
	// os.Remove(filenameF16)
}
