package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/FlavioCFOliveira/GoNeuron/goneuron"
)

func main() {
	// 1. Generate a synthetic CSV dataset (XOR-like)
	csvFile := "xor_data.csv"
	generateXORCSV(csvFile)
	defer os.Remove(csvFile)

	// 2. Load the dataset using the new LoadCSV utility
	// Columns: f1, f2, label
	dataset, err := goneuron.LoadCSV(csvFile, []int{2}, true)
	if err != nil {
		log.Fatalf("Failed to load CSV: %v", err)
	}

	fmt.Printf("Loaded %d samples with %d features each.\n", len(dataset.Samples), len(dataset.Samples[0]))

	// 3. Create a model
	model := goneuron.NewSequential(
		goneuron.Dense(8, goneuron.Tanh),
		goneuron.Dense(1, goneuron.Sigmoid),
	)

	model.Compile(goneuron.Adam(0.01), goneuron.MSE)

	// 4. Train with CSVLogger
	logFile := "training_log.csv"
	defer os.Remove(logFile)

	fmt.Println("Starting training...")
	model.Fit(dataset.Samples, dataset.Labels, 100, 4,
		goneuron.Logger(20),
		goneuron.CSVLogger(logFile, false),
	)

	fmt.Printf("Training complete. Log saved to %s\n", logFile)

	// 5. Verify log file exists and has content
	if _, err := os.Stat(logFile); err == nil {
		fmt.Println("Verified: CSV log file exists.")
	}

	// 6. Test prediction
	testInput := []float32{1.0, 0.0}
	prediction := model.Predict(testInput)
	fmt.Printf("Prediction for [1, 0]: %.4f (Expected near 1.0)\n", prediction[0])
}

func generateXORCSV(filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Header
	writer.Write([]string{"feat1", "feat2", "target"})

	// Data
	data := [][]string{
		{"0", "0", "0"},
		{"0", "1", "1"},
		{"1", "0", "1"},
		{"1", "1", "0"},
	}

	for _, record := range data {
		writer.Write(record)
	}
}
