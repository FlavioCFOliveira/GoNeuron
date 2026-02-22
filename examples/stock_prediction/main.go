// Stock Prediction example using LSTM architecture.
//
// This example demonstrates how to use the GoNeuron library to predict stock
// prices using an LSTM neural network. The model predicts the 'Close' price
// based on a window of previous days.
//
// Architecture:
//   - Input: Window of 10 days of normalized Close prices
//   - LSTM: 32 hidden units with pre-allocated buffers for performance
//   - Dense: 1 output neuron with Linear activation for regression
//
// Data preprocessing:
//   - Min-Max scaling for normalization
//   - Sliding window approach for sequence creation
//   - 80/20 train/test split
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// ============================================================================
// Data structures and constants
// ============================================================================

// StockData represents a single row from the CSV file
type StockData struct {
	Date       string
	Open       float64
	High       float64
	Low        float64
	Close      float64
	AdjClose   float64
	Volume     int64
}

// SequenceSample represents a single training/sample with input window and target
type SequenceSample struct {
	Input  []float64 // Window of previous days' prices
	Target float64   // Next day's price to predict
}

// NormalizationParams holds min/max values for Min-Max scaling
type NormalizationParams struct {
	Min float64
	Max float64
}

// ============================================================================
// Data loading and preprocessing functions
// ============================================================================

// loadStockData reads the CSV file and returns a slice of StockData
func loadStockData(filepath string) ([]StockData, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %w", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("CSV file is empty or has only header")
	}

	var data []StockData
	for i, record := range records[1:] { // Skip header row
		if len(record) < 6 {
			return nil, fmt.Errorf("row %d has insufficient columns", i+2)
		}

		var sd StockData
		sd.Date = record[0]

		if err := parseFloat(&record[1], &sd.Open); err != nil {
			return nil, fmt.Errorf("failed to parse Open at row %d: %w", i+2, err)
		}
		if err := parseFloat(&record[2], &sd.High); err != nil {
			return nil, fmt.Errorf("failed to parse High at row %d: %w", i+2, err)
		}
		if err := parseFloat(&record[3], &sd.Low); err != nil {
			return nil, fmt.Errorf("failed to parse Low at row %d: %w", i+2, err)
		}
		if err := parseFloat(&record[4], &sd.Close); err != nil {
			return nil, fmt.Errorf("failed to parse Close at row %d: %w", i+2, err)
		}
		if err := parseFloat(&record[5], &sd.AdjClose); err != nil {
			return nil, fmt.Errorf("failed to parse AdjClose at row %d: %w", i+2, err)
		}
		if err := parseInt(&record[6], &sd.Volume); err != nil {
			return nil, fmt.Errorf("failed to parse Volume at row %d: %w", i+2, err)
		}

		data = append(data, sd)
	}

	return data, nil
}

// parseFloat is a helper to parse float64 from string
func parseFloat(str *string, val *float64) error {
	var v float64
	_, err := fmt.Sscanf(*str, "%f", &v)
	if err != nil {
		return err
	}
	*val = v
	return nil
}

// parseInt is a helper to parse int64 from string
func parseInt(str *string, val *int64) error {
	var v int64
	_, err := fmt.Sscanf(*str, "%d", &v)
	if err != nil {
		return err
	}
	*val = v
	return nil
}

// extractClosePrices returns just the Close prices from stock data
func extractClosePrices(data []StockData) []float64 {
	prices := make([]float64, len(data))
	for i, d := range data {
		prices[i] = d.Close
	}
	return prices
}

// minMaxScale performs Min-Max scaling on price data
// Returns normalized prices and the normalization parameters
func minMaxScale(prices []float64) ([]float64, NormalizationParams) {
	if len(prices) == 0 {
		return nil, NormalizationParams{}
	}

	minVal := prices[0]
	maxVal := prices[0]
	for _, p := range prices {
		if p < minVal {
			minVal = p
		}
		if p > maxVal {
			maxVal = p
		}
	}

	// Avoid division by zero
	if maxVal == minVal {
		maxVal = minVal + 1.0
	}

	scaled := make([]float64, len(prices))
	for i, p := range prices {
		scaled[i] = (p - minVal) / (maxVal - minVal)
	}

	return scaled, NormalizationParams{Min: minVal, Max: maxVal}
}

// inverseScale reverses the Min-Max scaling
func inverseScale(scaled float64, params NormalizationParams) float64 {
	return scaled*(params.Max-params.Min) + params.Min
}

// createSequences creates input sequences using a sliding window approach
// Each input is a window of 'lookback' days, and the target is the next day's price
func createSequences(prices []float64, lookback int) []SequenceSample {
	if len(prices) <= lookback {
		return nil
	}

	var sequences []SequenceSample
	for i := 0; i <= len(prices)-lookback-1; i++ {
		sequence := SequenceSample{
			Input:  make([]float64, lookback),
			Target: prices[i+lookback],
		}
		copy(sequence.Input, prices[i:i+lookback])
		sequences = append(sequences, sequence)
	}

	return sequences
}

// splitData splits sequences into training and testing sets
func splitData(sequences []SequenceSample, trainRatio float64) ([]SequenceSample, []SequenceSample) {
	if len(sequences) == 0 {
		return nil, nil
	}

	splitIdx := int(float64(len(sequences)) * trainRatio)
	if splitIdx == 0 {
		splitIdx = 1
	}
	if splitIdx >= len(sequences) {
		splitIdx = len(sequences) - 1
	}

	return sequences[:splitIdx], sequences[splitIdx:]
}

// ============================================================================
// Neural network creation and training
// ============================================================================

// createLSTMNetwork creates an LSTM-based network for stock price prediction
//
// Architecture:
//   - Input layer: accepts a window of normalized prices
//   - LSTM layer: 32 hidden units with pre-allocated buffers
//   - Dense layer: single output with Linear activation (for regression)
//
// The LSTM processes the sequence one timestep at a time.
func createLSTMNetwork(lookback int, lstmUnits int) *net.Network {
	layers := []layer.Layer{
		// LSTM layer processes one input at a time
		layer.NewLSTM(lookback, lstmUnits),

		// Dense layer converts LSTM output to single price prediction
		// Linear activation for regression (no bounds on output)
		layer.NewDense(lstmUnits, 1, activations.Linear{}),
	}

	// Use MSE loss for regression
	// SGD optimizer with appropriate learning rate
	network := net.New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.01})

	return network
}

// trainEpoch trains the network on all sequences for one epoch
// Returns the average loss over the epoch
func trainEpoch(network *net.Network, sequences []SequenceSample, verbose bool) float64 {
	var totalLoss float64

	for i, seq := range sequences {
		// Reset LSTM state for each new sequence
		// This is critical - each stock price sequence is independent
		for _, l := range network.Layers() {
			if resettable, ok := l.(interface{ Reset() }); ok {
				resettable.Reset()
			}
		}

		// Train on this sequence
		loss := network.Train(seq.Input, []float64{seq.Target})
		totalLoss += loss

		// Progress logging
		if verbose && (i+1)%10 == 0 {
			avgLoss := totalLoss / float64(i+1)
			fmt.Printf("  Sample %d/%d, avg loss: %.6f\n", i+1, len(sequences), avgLoss)
		}
	}

	return totalLoss / float64(len(sequences))
}

// evaluateNetwork evaluates the network on a set of sequences
// Returns predictions, actual values, and MSE
func evaluateNetwork(network *net.Network, sequences []SequenceSample, normParams NormalizationParams) ([]float64, []float64, float64) {
	var predictions []float64
	var actuals []float64
	var mse float64

	for _, seq := range sequences {
		// Reset LSTM state for each sequence
		for _, l := range network.Layers() {
			if resettable, ok := l.(interface{ Reset() }); ok {
				resettable.Reset()
			}
		}

		// Forward pass
		output := network.Forward(seq.Input)
		predScaled := output[0]

		// Inverse scale to get actual price
		predActual := inverseScale(predScaled, normParams)
		actualActual := inverseScale(seq.Target, normParams)

		predictions = append(predictions, predActual)
		actuals = append(actuals, actualActual)

		// Compute squared error
		diff := predActual - actualActual
		mse += diff * diff
	}

	mse /= float64(len(sequences))
	return predictions, actuals, mse
}

// calculateMAPE calculates Mean Absolute Percentage Error
func calculateMAPE(predictions, actuals []float64) float64 {
	if len(predictions) != len(actuals) {
		return -1
	}

	var totalError float64
	for i := range predictions {
		if actuals[i] != 0 {
			error := math.Abs(predictions[i]-actuals[i]) / math.Abs(actuals[i])
			totalError += error
		}
	}

	return totalError / float64(len(predictions)) * 100 // Return as percentage
}

// ============================================================================
// Benchmarking
// ============================================================================

// benchmarkTraining measures training time per epoch
func benchmarkTraining(network *net.Network, sequences []SequenceSample, epochs int) []time.Duration {
	durations := make([]time.Duration, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		start := time.Now()

		trainEpoch(network, sequences, false)

		duration := time.Since(start)
		durations[epoch] = duration
	}

	return durations
}

// printBenchmarkStats prints training benchmark statistics
func printBenchmarkStats(durations []time.Duration) {
	if len(durations) == 0 {
		fmt.Println("No benchmark data")
		return
	}

	var total time.Duration
	for _, d := range durations {
		total += d
	}

	avg := total / time.Duration(len(durations))
	minVal := durations[0]
	maxVal := durations[0]
	for _, d := range durations {
		if d < minVal {
			minVal = d
		}
		if d > maxVal {
			maxVal = d
		}
	}

	fmt.Printf("\n--- Training Benchmark ---\n")
	fmt.Printf("Epochs: %d\n", len(durations))
	fmt.Printf("Average time per epoch: %v\n", avg)
	fmt.Printf("Min time per epoch: %v\n", minVal)
	fmt.Printf("Max time per epoch: %v\n", maxVal)
}

// ============================================================================
// Main execution
// ============================================================================

func main() {
	fmt.Println("=== Stock Price Prediction using LSTM ===\n")

	// Configuration
	const (
		lookback     = 10   // Number of previous days to use as input
		lstmUnits    = 32   // Number of LSTM hidden units
		epochs       = 100  // Number of training epochs
		trainRatio   = 0.8  // Training data ratio
	)

	// csvPath is relative to the current working directory
	csvPath := "examples/stock_prediction/datasets/GOOG.csv"

	// Load data
	fmt.Println("Loading stock data...")
	data, err := loadStockData(csvPath)
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %d data points\n", len(data))

	// Extract and normalize prices
	fmt.Println("\nPreprocessing data...")
	closePrices := extractClosePrices(data)
	fmt.Printf("Price range: $%.2f - $%.2f\n", closePrices[0], closePrices[len(closePrices)-1])

	scaledPrices, normParams := minMaxScale(closePrices)
	fmt.Printf("Normalized price range: %.4f - %.4f\n", scaledPrices[0], scaledPrices[len(scaledPrices)-1])

	// Create sequences
	fmt.Println("\nCreating sequences...")
	sequences := createSequences(scaledPrices, lookback)
	fmt.Printf("Created %d sequences\n", len(sequences))

	// Split data
	trainSeq, testSeq := splitData(sequences, trainRatio)
	fmt.Printf("Training samples: %d\n", len(trainSeq))
	fmt.Printf("Testing samples: %d\n", len(testSeq))

	// Create network
	fmt.Println("\nCreating LSTM network...")
	fmt.Printf("Architecture: %d (input) -> LSTM(%d) -> Dense(1)\n", lookback, lstmUnits)

	network := createLSTMNetwork(lookback, lstmUnits)

	// Training loop with progress logging
	fmt.Printf("\n=== Training for %d epochs ===\n", epochs)
	startTime := time.Now()

	for epoch := 1; epoch <= epochs; epoch++ {
		// Train with verbose logging for first and last epochs
		verbose := epoch == 1 || epoch == epochs || epoch%20 == 0
		if verbose {
			fmt.Printf("\nEpoch %d/%d\n", epoch, epochs)
		}
		avgLoss := trainEpoch(network, trainSeq, verbose)

		if verbose {
			fmt.Printf("Epoch %d - Average Loss: %.6f\n", epoch, avgLoss)
		}

		// Periodic evaluation
		if epoch%20 == 0 || epoch == epochs {
			_, _, testMSE := evaluateNetwork(network, testSeq, normParams)
			testRMSE := math.Sqrt(testMSE)
			fmt.Printf("  Test RMSE: $%.2f\n", testRMSE)
		}
	}

	totalTrainingTime := time.Since(startTime)
	fmt.Printf("\n=== Training Complete ===\n")
	fmt.Printf("Total training time: %v\n", totalTrainingTime)
	fmt.Printf("Average time per epoch: %v\n", totalTrainingTime/time.Duration(epochs))

	// Final evaluation on test set
	fmt.Println("\n=== Final Evaluation on Test Set ===")
	predictions, actuals, testMSE := evaluateNetwork(network, testSeq, normParams)
	testRMSE := math.Sqrt(testMSE)
	testMAPE := calculateMAPE(predictions, actuals)

	fmt.Printf("Test MSE:  %.6f\n", testMSE)
	fmt.Printf("Test RMSE: $%.2f\n", testRMSE)
	fmt.Printf("Test MAPE: %.2f%%\n", testMAPE)

	// Show sample predictions
	fmt.Println("\n--- Sample Predictions (first 5 test samples) ---")
	fmt.Printf("%-15s %-15s %-15s %-10s\n", "Actual", "Predicted", "Error", "Error %")
	displayLimit := 5
	if len(actuals) < displayLimit {
		displayLimit = len(actuals)
	}
	for i := 0; i < displayLimit; i++ {
		error := predictions[i] - actuals[i]
		errorPct := 0.0
		if actuals[i] != 0 {
			errorPct = (error / actuals[i]) * 100
		}
		fmt.Printf("$%-14.2f $%-14.2f $%-14.2f %.2f%%\n",
			actuals[i], predictions[i], error, errorPct)
	}

	// Benchmark training speed
	fmt.Println("\n=== Training Performance Benchmark ===")
	benchmarkSeq := trainSeq
	if len(benchmarkSeq) > 200 {
		benchmarkSeq = trainSeq[:200] // Use subset for faster benchmark
	}

	benchmarkNetwork := createLSTMNetwork(lookback, lstmUnits)
	benchmarkDurations := benchmarkTraining(benchmarkNetwork, benchmarkSeq, 10)
	printBenchmarkStats(benchmarkDurations)

	// Memory statistics
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("\n--- Memory Usage ---\n")
	fmt.Printf("Alloc: %v KB\n", m.Alloc/1024)
	fmt.Printf("Total Alloc: %v KB\n", m.TotalAlloc/1024)
	fmt.Printf("Heap Alloc: %v KB\n", m.HeapAlloc/1024)
	fmt.Printf("Goroutines: %d\n", runtime.NumGoroutine())

	fmt.Println("\n=== Example Complete ===")
}
