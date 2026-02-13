// Package main - LSTM Training and Inference Example
// Demonstrates training an LSTM neural network from CSV data for time series prediction.
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// DataRecord represents a normalized data record for training
type DataRecord struct {
	Value float64 // Normalized value
}

// SequencedData represents a sequence with input and output
type SequencedData struct {
	Input  float64 // Input value (single timestamp)
	Target float64 // Target value (next timestamp)
}

func main() {
	fmt.Println("=============================================================")
	fmt.Println("  GoNeuron - LSTM Time Series Prediction")
	fmt.Println("=============================================================")
	fmt.Println()

	// Step 1: Generate time series data
	fmt.Println("--- Step 1: Generating Time Series Data ---")
	records := generateTimeSeriesData(50000)
	saveToCSV(records, "time_series.csv")
	fmt.Printf("Generated %d records saved to time_series.csv\n", len(records))
	fmt.Println()

	// Step 2: Load and prepare sequences
	fmt.Println("--- Step 2: Loading and Preparing Data ---")
	records, err := loadFromCSV("time_series.csv")
	if err != nil {
		log.Fatal("Error loading CSV:", err)
	}
	fmt.Printf("Loaded %d records\n", len(records))
	fmt.Printf("Value range: [%.4f, %.4f]\n", records[0].Value, records[len(records)-1].Value)
	fmt.Println()

	// Step 3: Create sequences for LSTM training
	// Each sample: single input value -> predict next value
	fmt.Println("--- Step 3: Creating Training Samples ---")
	trainSamples, testSamples := createSamples(records, 0.8)
	fmt.Printf("Training samples: %d\n", len(trainSamples))
	fmt.Printf("Testing samples: %d\n", len(testSamples))
	fmt.Println()

	// Step 4: Train LSTM network
	fmt.Println("--- Step 4: Training LSTM Network ---")
	fmt.Println("Architecture: LSTM(1->32) -> Dense(32->1)")
	fmt.Println("  Input: 1 feature (normalized value)")
	fmt.Println("  LSTM: 32 hidden units (Sigmoid gates, Tanh cell)")
	fmt.Println("  Output: 1 neuron (Tanh activation)")
	fmt.Println("Loss: MSE")
	fmt.Println("Optimizer: Adam (lr=0.001)")
	fmt.Println()

	network := createLSTMNetwork()

	// Extract layers for manual training
	lstmLayer := network.Layers()[0].(*layer.LSTM)
	outputLayer := network.Layers()[1].(*layer.Dense)

	epochs := 50
	fmt.Println("Training progress:")
	lastLoss := 1.0

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		sampleCount := 0

		// Shuffle training data
		shuffleData(trainSamples)

		// Train on samples - each sample is a single input->target pair
		for _, sample := range trainSamples {
			// Reset LSTM state for each sample (treat each as independent)
			lstmLayer.Reset()

			// Forward through LSTM with single input
			hidden := lstmLayer.Forward([]float64{sample.Input})

			// Forward through output layer
			pred := outputLayer.Forward(hidden)

			// Train the network
			loss := network.Train(pred, []float64{sample.Target})
			totalLoss += loss
			sampleCount++
		}

		avgLoss := totalLoss / float64(sampleCount)

		if epoch%10 == 0 {
			fmt.Printf("  Epoch %3d: Loss = %.6f\n", epoch, avgLoss)
		}
		// Track best loss
		if avgLoss < lastLoss-0.00001 {
			lastLoss = avgLoss
		}
	}

	fmt.Println("Training complete!")
	fmt.Printf("Final training loss: %.6f\n", lastLoss)
	fmt.Println()

	// Step 5: Evaluate model
	fmt.Println("--- Step 5: Evaluating Model ---")
	trainMSE, trainMAE := evaluateLSTM(network, trainSamples)
	fmt.Printf("Training MSE: %.6f\n", trainMSE)
	fmt.Printf("Training MAE: %.6f\n", trainMAE)

	testMSE, testMAE := evaluateLSTM(network, testSamples)
	fmt.Printf("Testing MSE: %.6f\n", testMSE)
	fmt.Printf("Testing MAE: %.6f\n", testMAE)
	fmt.Println()

	// Step 6: Sample predictions
	fmt.Println("--- Step 6: Sample Predictions ---")
	fmt.Println("Actual vs Predicted Values:")
	fmt.Println("----------------------------------------")

	sampleAcc10, sampleAcc20, sampleAcc30 := 0, 0, 0
	for i := 0; i < min(10, len(testSamples)); i++ {
		sample := testSamples[i]
		lstmLayer.Reset()
		hidden := lstmLayer.Forward([]float64{sample.Input})
		pred := outputLayer.Forward(hidden)[0]

		errorPct := math.Abs(sample.Target-pred) * 100 // Values are already normalized [0,1]

		if errorPct < 10 {
			sampleAcc10++
		}
		if errorPct < 20 {
			sampleAcc20++
		}
		if errorPct < 30 {
			sampleAcc30++
		}

		status := ""
		if errorPct < 10 {
			status = " [EXCELLENT]"
		} else if errorPct < 20 {
			status = " [GOOD]"
		} else if errorPct < 30 {
			status = " [FAIR]"
		}

		fmt.Printf("Sample %d: Actual=%.4f, Predicted=%.4f (%.1f%%)%s\n",
			i+1, sample.Target, pred, errorPct, status)
	}

	// Full accuracy
	fullAcc10, fullAcc20, fullAcc30 := 0, 0, 0
	for _, sample := range testSamples {
		lstmLayer.Reset()
		hidden := lstmLayer.Forward([]float64{sample.Input})
		pred := outputLayer.Forward(hidden)[0]

		errorPct := math.Abs(sample.Target-pred) * 100

		if errorPct < 10 {
			fullAcc10++
		}
		if errorPct < 20 {
			fullAcc20++
		}
		if errorPct < 30 {
			fullAcc30++
		}
	}

	fmt.Printf("\nFull Accuracy on Test Set (%d samples):\n", len(testSamples))
	fmt.Printf("  Within 10%%: %d/%d (%.1f%%)\n", fullAcc10, len(testSamples), float64(fullAcc10)/float64(len(testSamples))*100)
	fmt.Printf("  Within 20%%: %d/%d (%.1f%%)\n", fullAcc20, len(testSamples), float64(fullAcc20)/float64(len(testSamples))*100)
	fmt.Printf("  Within 30%%: %d/%d (%.1f%%)\n", fullAcc30, len(testSamples), float64(fullAcc30)/float64(len(testSamples))*100)
	fmt.Println()

	// Step 7: Save trained model
	fmt.Println("--- Step 7: Saving Trained Model ---")
	err = network.Save("lstm_model.bin")
	if err != nil {
		log.Fatal("Error saving model:", err)
	}
	fmt.Println("Model saved to lstm_model.bin")
	fmt.Println()

	// Step 8: Verify saved model
	fmt.Println("--- Step 8: Verifying Saved Model ---")
	loadedNetwork, _, err := net.Load("lstm_model.bin")
	if err != nil {
		log.Fatal("Error loading model:", err)
	}
	fmt.Println("Model loaded successfully!")

	loadedLSTM := loadedNetwork.Layers()[0].(*layer.LSTM)
	loadedOutput := loadedNetwork.Layers()[1].(*layer.Dense)

	allMatch := true
	for i := 0; i < 5; i++ {
		sample := testSamples[i]
		lstmLayer.Reset()
		hidden := lstmLayer.Forward([]float64{sample.Input})
		originalPred := outputLayer.Forward(hidden)[0]

		loadedLSTM.Reset()
		loadedHidden := loadedLSTM.Forward([]float64{sample.Input})
		loadedPred := loadedOutput.Forward(loadedHidden)[0]

		match := "MATCH"
		if math.Abs(originalPred-loadedPred) > 1e-6 {
			match = "MISMATCH"
			allMatch = false
		}
		fmt.Printf("  Sample %d: Original=%.6f, Loaded=%.6f [%s]\n", i+1, originalPred, loadedPred, match)
	}
	fmt.Println()

	// Step 9: Demonstrate prediction from new sequence
	fmt.Println("--- Step 9: Prediction from New Sequence ---")
	fmt.Println("Generating prediction from a new sequence...")
	newSequence := generateNewSequence()
	fmt.Printf("Input sequence: [")
	for i, v := range newSequence {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%.4f", v)
	}
	fmt.Println("]")

	loadedLSTM.Reset()
	for _, val := range newSequence {
		loadedLSTM.Forward([]float64{val})
	}
	prediction := loadedOutput.Forward(loadedLSTM.Hidden())[0]
	fmt.Printf("Predicted next value: %.4f\n", prediction)
	fmt.Println()

	fmt.Println("=============================================================")
	if allMatch {
		fmt.Println("  SUCCESS: Model verification passed!")
	}
	fmt.Println("  LSTM Training and Inference Complete!")
	fmt.Println("=============================================================")
	fmt.Println()
	fmt.Printf("Summary:\n")
	fmt.Printf("  - Training samples: %d\n", len(trainSamples))
	fmt.Printf("  - Testing samples: %d\n", len(testSamples))
	fmt.Printf("  - Training MSE: %.6f\n", trainMSE)
	fmt.Printf("  - Testing MSE: %.6f\n", testMSE)
	fmt.Printf("  - Testing MAE: %.6f\n", testMAE)
	fmt.Printf("  - Model saved: lstm_model.bin\n")
}

// generateTimeSeriesData generates synthetic time series data
func generateTimeSeriesData(count int) []DataRecord {
	rand.Seed(42) // Fixed seed for reproducibility

	records := make([]DataRecord, count)

	// Generate a sine wave with noise
	for i := 0; i < count; i++ {
		t := float64(i) / float64(count-1) * 4 * math.Pi

		// Base signal: sine wave
		value := math.Sin(t)

		// Add trend
		trend := 0.1 * math.Sin(t/2)

		// Add noise
		noise := (rand.Float64() - 0.5) * 0.1

		// Combine components
		rawValue := value + trend + noise

		// Normalize to [0, 1]
		normalized := (rawValue + 1) / 2 // Map from [-1, 1] to [0, 1]

		records[i] = DataRecord{
			Value: normalized,
		}
	}

	return records
}

// saveToCSV saves records to a normalized CSV file
func saveToCSV(records []DataRecord, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Error creating file:", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// CSV with normalized values for optimal training
	writer.Write([]string{"Value"})
	for _, rec := range records {
		writer.Write([]string{
			strconv.FormatFloat(rec.Value, 'f', 6, 64),
		})
	}
}

// loadFromCSV loads normalized records from CSV
func loadFromCSV(filename string) ([]DataRecord, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data []DataRecord
	for i, record := range records {
		if i == 0 {
			continue // Skip header
		}
		value, _ := strconv.ParseFloat(record[0], 64)

		data = append(data, DataRecord{
			Value: value,
		})
	}
	return data, nil
}

// createSamples creates input->target pairs for LSTM training
func createSamples(records []DataRecord, trainRatio float64) ([]SequencedData, []SequencedData) {
	var samples []SequencedData

	// Create samples where each input predicts the next value
	for i := 0; i < len(records)-1; i++ {
		samples = append(samples, SequencedData{
			Input:  records[i].Value,
			Target: records[i+1].Value,
		})
	}

	// Shuffle samples
	shuffleData(samples)

	// Split into train and test
	splitIdx := int(float64(len(samples)) * trainRatio)
	trainSamples := samples[:splitIdx]
	testSamples := samples[splitIdx:]

	return trainSamples, testSamples
}

// createLSTMNetwork creates an LSTM network for time series prediction
func createLSTMNetwork() *net.Network {
	// LSTM layer: 1 input feature -> 32 hidden units
	lstm := layer.NewLSTM(1, 32)

	// Output layer: 32 -> 1 with Tanh activation
	output := layer.NewDense(32, 1, activations.Tanh{})

	network := net.New(
		[]layer.Layer{lstm, output},
		loss.MSE{},
		opt.NewAdam(0.001),
	)

	return network
}

// evaluateLSTM evaluates the LSTM network
func evaluateLSTM(network *net.Network, samples []SequencedData) (mse, mae float64) {
	sumSquaredError := 0.0
	sumAbsoluteError := 0.0

	lstmLayer := network.Layers()[0].(*layer.LSTM)
	outputLayer := network.Layers()[1].(*layer.Dense)

	for _, sample := range samples {
		// Reset LSTM state
		lstmLayer.Reset()

		// Process input
		hidden := lstmLayer.Forward([]float64{sample.Input})

		// Get prediction
		pred := outputLayer.Forward(hidden)[0]

		// Calculate error
		error := sample.Target - pred
		sumSquaredError += error * error
		sumAbsoluteError += math.Abs(error)
	}

	n := float64(len(samples))
	mse = sumSquaredError / n
	mae = sumAbsoluteError / n

	return mse, mae
}

// shuffleData shuffles a slice of SequencedData
func shuffleData(data []SequencedData) {
	for i := len(data) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
}

// generateNewSequence generates a new sequence for prediction demo
func generateNewSequence() []float64 {
	rand.Seed(42) // Fixed seed for reproducibility
	sequence := make([]float64, 10)
	for i := range sequence {
		sequence[i] = rand.Float64()
	}
	return sequence
}

// min returns the smaller of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
