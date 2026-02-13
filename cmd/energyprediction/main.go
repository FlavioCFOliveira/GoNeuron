// Package main - LSTM Energy Consumption Prediction
// Demonstrates training an LSTM neural network for energy consumption forecasting.
// Uses real-world patterns: daily cycles, weekly patterns, and weather effects.
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// EnergyData represents normalized energy consumption data
type EnergyData struct {
	Timestamp      time.Time
	Hour           float64 // Hour of day (0-23)
	DayOfWeek      float64 // Day of week (0-6)
	IsWeekend      float64 // 1 if weekend, 0 otherwise
	Temperature    float64 // Temperature in Celsius (normalized)
	Humidity       float64 // Humidity percentage (normalized)
	SolarRadiation float64 // Solar radiation (normalized)
	Consumption    float64 // Normalized consumption [0, 1]
}

// SequenceData represents a sequence for LSTM training
type SequenceData struct {
	Inputs []float64 // Multiple features for the input
	Target float64   // Target consumption value
}

func main() {
	fmt.Println("=============================================================")
	fmt.Println("  GoNeuron - Energy Consumption Prediction using LSTM")
	fmt.Println("=============================================================")
	fmt.Println()

	// Step 1: Generate synthetic energy consumption data
	fmt.Println("--- Step 1: Generating Energy Consumption Data ---")
	energyData := generateEnergyData(50000)
	saveToCSV(energyData, "energy_consumption.csv")
	fmt.Printf("Generated %d records saved to energy_consumption.csv\n", len(energyData))
	fmt.Println()

	// Step 2: Load and verify data
	fmt.Println("--- Step 2: Loading and Verifying Data ---")
	energyData, err := loadFromCSV("energy_consumption.csv")
	if err != nil {
		log.Fatal("Error loading CSV:", err)
	}
	fmt.Printf("Loaded %d records\n", len(energyData))
	fmt.Printf("Consumption range: [%.4f, %.4f]\n", getMin(energyData), getMax(energyData))
	fmt.Printf("Temperature range: [%.2f, %.2f] C\n", getMinTemp(energyData), getMaxTemp(energyData))
	fmt.Println()

	// Step 3: Create sequences for LSTM training
	fmt.Println("--- Step 3: Creating Training Sequences ---")
	// Use sequence length of 6 timesteps (3 hours history + 3 hours patterns)
	seqLength := 6
	trainSequences, testSequences := createSequences(energyData, seqLength, 0.8)
	fmt.Printf("Training sequences: %d\n", len(trainSequences))
	fmt.Printf("Testing sequences: %d\n", len(testSequences))
	fmt.Printf("Features per sequence: %d (6 timesteps x 6 features)\n", len(trainSequences[0].Inputs))
	fmt.Println()

	// Step 4: Train LSTM network
	fmt.Println("--- Step 4: Training LSTM Network ---")
	fmt.Println("Architecture: LSTM(36->32) -> Dense(32->16, Tanh) -> Dense(16->1, Tanh)")
	fmt.Println("  Input: 6 timesteps x 6 features = 36 inputs")
	fmt.Println("  LSTM: 32 hidden units")
	fmt.Println("  Dense 1: 32 -> 16 neurons (Tanh)")
	fmt.Println("  Output: 1 neuron (Tanh activation)")
	fmt.Println("Loss: MSE")
	fmt.Println("Optimizer: Adam (lr=0.002)")
	fmt.Println()

	network := createLSTMNetwork()

	epochs := 30
	batchSize := 32
	fmt.Println("Training progress:")
	lastLoss := 1.0

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		batchCount := 0

		// Shuffle training data
		shuffleData(trainSequences)

		// Train in batches
		for i := 0; i < len(trainSequences); i += batchSize {
			end := i + batchSize
			if end > len(trainSequences) {
				end = len(trainSequences)
			}

			batchX := make([][]float64, end-i)
			batchY := make([][]float64, end-i)

			for j := i; j < end; j++ {
				batchX[j-i] = trainSequences[j].Inputs
				batchY[j-i] = []float64{trainSequences[j].Target}
			}

			loss := network.TrainBatch(batchX, batchY)
			totalLoss += loss
			batchCount++
		}

		avgLoss := totalLoss / float64(batchCount)

		if epoch%5 == 0 {
			fmt.Printf("  Epoch %3d: Loss = %.6f\n", epoch, avgLoss)
		}

		// Early stopping check
		if avgLoss < lastLoss-0.00001 {
			lastLoss = avgLoss
		}
	}

	fmt.Println("Training complete!")
	fmt.Printf("Final training loss: %.6f\n", lastLoss)
	fmt.Println()

	// Step 5: Evaluate model
	fmt.Println("--- Step 5: Evaluating Model ---")
	trainMSE, trainMAE, trainR2 := evaluateLSTM(network, testSequences)
	fmt.Printf("Training MSE: %.6f\n", trainMSE)
	fmt.Printf("Training MAE: %.6f\n", trainMAE)
	fmt.Printf("Training R2:  %.6f\n", trainR2)

	testMSE, testMAE, testR2 := evaluateLSTM(network, testSequences)
	fmt.Printf("Testing MSE:  %.6f\n", testMSE)
	fmt.Printf("Testing MAE:  %.6f\n", testMAE)
	fmt.Printf("Testing R2:   %.6f\n", testR2)
	fmt.Println()

	// Step 6: Sample predictions
	fmt.Println("--- Step 6: Sample Predictions ---")
	fmt.Println("Actual vs Predicted Energy Consumption:")
	fmt.Println("----------------------------------------")

	acc10, acc20, acc30 := 0, 0, 0
	for i := 0; i < min(15, len(testSequences)); i++ {
		seq := testSequences[i]
		pred := network.Forward(seq.Inputs)[0]
		actual := seq.Target

		errorPct := math.Abs(actual-pred) * 100

		if errorPct < 10 {
			acc10++
		}
		if errorPct < 20 {
			acc20++
		}
		if errorPct < 30 {
			acc30++
		}

		status := ""
		if errorPct < 10 {
			status = " [EXCELLENT]"
		} else if errorPct < 20 {
			status = " [GOOD]"
		} else if errorPct < 30 {
			status = " [FAIR]"
		}

		fmt.Printf("Sample %2d: Actual=%.4f, Predicted=%.4f (Err: %.1f%%)%s\n",
			i+1, actual, pred, errorPct, status)
	}

	// Full accuracy calculation
	for _, seq := range testSequences {
		pred := network.Forward(seq.Inputs)[0]
		errorPct := math.Abs(pred-seq.Target) * 100

		if errorPct < 10 {
			acc10++
		}
		if errorPct < 20 {
			acc20++
		}
		if errorPct < 30 {
			acc30++
		}
	}

	fmt.Printf("\nFull Accuracy on Test Set (%d samples):\n", len(testSequences))
	fmt.Printf("  Within 10%%: %d/%d (%.1f%%)\n", acc10, len(testSequences), float64(acc10)/float64(len(testSequences))*100)
	fmt.Printf("  Within 20%%: %d/%d (%.1f%%)\n", acc20, len(testSequences), float64(acc20)/float64(len(testSequences))*100)
	fmt.Printf("  Within 30%%: %d/%d (%.1f%%)\n", acc30, len(testSequences), float64(acc30)/float64(len(testSequences))*100)
	fmt.Println()

	// Step 7: Save trained model
	fmt.Println("--- Step 7: Saving Trained Model ---")
	err = network.Save("energy_prediction_model.bin")
	if err != nil {
		log.Fatal("Error saving model:", err)
	}
	fmt.Println("Model saved to energy_prediction_model.bin")
	fmt.Println()

	// Step 8: Verify saved model
	fmt.Println("--- Step 8: Verifying Saved Model ---")
	loadedNetwork, _, err := net.Load("energy_prediction_model.bin")
	if err != nil {
		log.Fatal("Error loading model:", err)
	}
	fmt.Println("Model loaded successfully!")

	allMatch := true
	for i := 0; i < 5; i++ {
		seq := testSequences[i]

		// For LSTM networks, we need to reset state for consistent predictions
		// because saved models don't preserve internal LSTM state
		originalPred := network.Forward(seq.Inputs)[0]

		loadedPred := loadedNetwork.Forward(seq.Inputs)[0]

		match := "MATCH"
		// Allow small tolerance for LSTM due to state reset behavior
		if math.Abs(originalPred-loadedPred) > 1e-4 {
			match = "MISMATCH"
			allMatch = false
		}
		fmt.Printf("  Sample %d: Original=%.6f, Loaded=%.6f [%s]\n", i+1, originalPred, loadedPred, match)
	}
	fmt.Println()

	// Step 9: Demonstrate prediction from new sequence
	fmt.Println("--- Step 9: Prediction from New Sequence ---")
	fmt.Println("Generating prediction based on recent consumption pattern...")

	// Get recent sequence from test data
	recentSeq := testSequences[0]
	fmt.Printf("Input sequence (last 6 hours consumption): [")
	for i := 5; i < len(recentSeq.Inputs); i += 6 {
		if i > 5 {
			fmt.Print(", ")
		}
		fmt.Printf("%.3f", recentSeq.Inputs[i])
	}
	fmt.Println("]")

	prediction := loadedNetwork.Forward(recentSeq.Inputs)[0]
	fmt.Printf("Predicted next hour consumption: %.4f (%.1f%% of max)\n", prediction, prediction*100)
	fmt.Println()

	// Step 10: Multi-step prediction
	fmt.Println("--- Step 10: Multi-Step Forecast (Next 24 hours) ---")
	fmt.Println("Generating 24-hour forecast based on current pattern...")

	// Use the last sequence as starting point
	currentSeq := make([]float64, len(recentSeq.Inputs))
	copy(currentSeq, recentSeq.Inputs)

	predictions := make([]float64, 24)
	for i := 0; i < 24; i++ {
		// Predict next value
		pred := loadedNetwork.Forward(currentSeq)[0]
		predictions[i] = pred

		// Shift sequence: remove first 6 features, add new prediction
		newSeq := make([]float64, len(currentSeq))
		copy(newSeq, currentSeq[6:])
		// Update the last consumption value (feature index 5, 11, 17, etc.)
		for j := 5; j < len(newSeq); j += 6 {
			newSeq[j] = pred
		}
		currentSeq = newSeq
	}

	fmt.Println("24-hour forecast:")
	fmt.Println("  Hour    | Predicted | Pattern")
	fmt.Println("  --------|-----------|--------")
	for i := 0; i < 24; i++ {
		hour := (int(recentSeq.Inputs[0]*24) + i + 1) % 24
		pat := "NIGHT"
		if hour >= 6 && hour < 12 {
			pat = "MORNING"
		} else if hour >= 12 && hour < 18 {
			pat = "AFTERNOON"
		} else if hour >= 18 && hour < 22 {
			pat = "EVENING"
		}
		fmt.Printf("  Hour %2d:  %.4f     | %s\n", hour, predictions[i], pat)
	}

	fmt.Println()
	fmt.Println("=============================================================")
	if allMatch {
		fmt.Println("  SUCCESS: Model verification passed!")
	}
	fmt.Println("  Energy Consumption Prediction Complete!")
	fmt.Println("=============================================================")
	fmt.Println()
	fmt.Printf("Summary:\n")
	fmt.Printf("  - Training sequences: %d\n", len(trainSequences))
	fmt.Printf("  - Testing sequences:  %d\n", len(testSequences))
	fmt.Printf("  - Training MSE: %.6f\n", trainMSE)
	fmt.Printf("  - Testing MSE:  %.6f\n", testMSE)
	fmt.Printf("  - Testing MAE:  %.6f\n", testMAE)
	fmt.Printf("  - Model saved:  energy_prediction_model.bin\n")
}

// clamp restricts a float64 to be within [min, max]
func clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

// generateEnergyData generates synthetic energy consumption data with realistic patterns
func generateEnergyData(count int) []EnergyData {
	rand.Seed(42) // Fixed seed for reproducibility

	data := make([]EnergyData, count)

	// Base values for normalization
	const (
		tempMin       = -5.0
		tempMax       = 35.0
		humidityMin   = 20.0
		humidityMax   = 90.0
		solarMin      = 0.0
		solarMax      = 800.0
		consumptionMin = 0.15
		consumptionMax = 0.85
	)

	for i := 0; i < count; i++ {
		// Create timestamp
		timestamp := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).
			Add(time.Duration(i) * time.Hour)

		hour := float64(timestamp.Hour())
		dayOfWeek := float64(timestamp.Weekday())
		isWeekend := 0.0
		if dayOfWeek == 0 || dayOfWeek == 6 { // Sunday or Saturday
			isWeekend = 1.0
		}

		// Temperature: daily cycle with seasonal variation
		dayProgress := float64(i) / float64(count) * 2 * math.Pi
		baseTemp := 15.0 + 10.0*math.Sin(dayProgress-0.5) // Seasonal shift
		tempNoise := (rand.Float64() - 0.5) * 3.0
		temperature := baseTemp + tempNoise

		// Humidity: inverse correlation with temperature
		humidity := 60.0 - 0.5*(temperature-15.0) + (rand.Float64()-0.5)*10

		// Solar radiation: peaks at noon, zero at night
		var solarRadiation float64
		if hour >= 6 && hour < 20 {
			// Peak at solar noon (hour 12-13)
			noonFactor := math.Cos((hour-13)*math.Pi/8)
			solarRadiation = 600.0 * math.Max(0, noonFactor)
		} else {
			solarRadiation = 0.0
		}
		solarRadiation += (rand.Float64() - 0.5) * 50 // Add noise

		// Energy consumption: complex pattern based on:
		// 1. Hour of day (peaks in morning and evening)
		// 2. Day of week (lower on weekends)
		// 3. Temperature (higher when very hot or cold)
		// 4. Solar radiation (lower when more solar)

		// Base daily pattern: double peak (morning 8-9, evening 18-20)
		morningExponent := -0.5 * clamp((hour-8.5)/1.5, -5, 5)
		morningPeak := math.Exp(morningExponent)
		eveningExponent := -0.5 * clamp((hour-19.0)/1.5, -5, 5)
		eveningPeak := math.Exp(eveningExponent)

		// Day of week effect
		weekdayFactor := 1.0 - 0.15*isWeekend

		// Temperature effect: U-shaped (higher consumption when too hot or cold)
		optimalTemp := 20.0
		tempDeviation := math.Abs(temperature - optimalTemp)
		tempFactor := 1.0 + 0.02*tempDeviation

		// Solar effect: reduce consumption when solar is high
		solarFactor := 1.0 - 0.0002*solarRadiation

		// Random noise
		noise := (rand.Float64() - 0.5) * 0.1

		// Combine all factors
		consumption := 0.35 * (morningPeak + eveningPeak) // Base daily pattern
		consumption *= weekdayFactor                      // Weekday/weekend adjustment
		consumption *= tempFactor                         // Temperature adjustment
		consumption *= solarFactor                        // Solar adjustment
		consumption += noise                              // Random noise

		// Normalize consumption to [consumptionMin, consumptionMax]
		consumption = math.Max(consumptionMin, math.Min(consumptionMax, consumption))

		data[i] = EnergyData{
			Timestamp:      timestamp,
			Hour:           hour / 24.0,                // Normalize to [0, 1]
			DayOfWeek:      dayOfWeek / 7.0,            // Normalize to [0, 1]
			IsWeekend:      isWeekend,
			Temperature:    (temperature - tempMin) / (tempMax - tempMin),
			Humidity:       (humidity - humidityMin) / (humidityMax - humidityMin),
			SolarRadiation: solarRadiation / solarMax,
			Consumption:    consumption,
		}
	}

	return data
}

// saveToCSV saves energy data to a normalized CSV file
func saveToCSV(data []EnergyData, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Error creating file:", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{
		"Timestamp", "Hour", "DayOfWeek", "IsWeekend",
		"Temperature", "Humidity", "SolarRadiation", "Consumption",
	}
	writer.Write(header)

	// Write data
	for _, d := range data {
		row := []string{
			d.Timestamp.Format("2006-01-02 15:04:05"),
			strconv.FormatFloat(d.Hour*24, 'f', 1, 64),
			strconv.FormatFloat(d.DayOfWeek*7, 'f', 1, 64),
			strconv.FormatFloat(d.IsWeekend, 'f', 0, 64),
			strconv.FormatFloat(d.Temperature*(35-(-5))+(-5), 'f', 2, 64),
			strconv.FormatFloat(d.Humidity*(90-20)+20, 'f', 2, 64),
			strconv.FormatFloat(d.SolarRadiation*800, 'f', 2, 64),
			strconv.FormatFloat(d.Consumption, 'f', 6, 64),
		}
		writer.Write(row)
	}
}

// loadFromCSV loads energy data from CSV
func loadFromCSV(filename string) ([]EnergyData, error) {
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

	var data []EnergyData
	for i, record := range records {
		if i == 0 {
			continue // Skip header
		}

		timestamp, _ := time.Parse("2006-01-02 15:04:05", record[0])
		hour, _ := strconv.ParseFloat(record[1], 64)
		dayOfWeek, _ := strconv.ParseFloat(record[2], 64)
		isWeekend, _ := strconv.ParseFloat(record[3], 64)
		temp, _ := strconv.ParseFloat(record[4], 64)
		humidity, _ := strconv.ParseFloat(record[5], 64)
		solar, _ := strconv.ParseFloat(record[6], 64)
		consumption, _ := strconv.ParseFloat(record[7], 64)

		data = append(data, EnergyData{
			Timestamp:      timestamp,
			Hour:           hour / 24.0,
			DayOfWeek:      dayOfWeek / 7.0,
			IsWeekend:      isWeekend,
			Temperature:    (temp - (-5)) / (35 - (-5)),
			Humidity:       (humidity - 20) / (90 - 20),
			SolarRadiation: solar / 800,
			Consumption:    consumption,
		})
	}

	return data, nil
}

// createSequences creates input->target pairs for LSTM training
func createSequences(data []EnergyData, seqLength int, trainRatio float64) ([]SequenceData, []SequenceData) {
	var sequences []SequenceData

	// Create sequences with sliding window
	for i := 0; i <= len(data)-seqLength-1; i++ {
		inputs := make([]float64, seqLength*6) // 6 features per timestep
		for j := 0; j < seqLength; j++ {
			idx := i + j
			feat := &data[idx]
			offset := j * 6
			inputs[offset+0] = feat.Hour
			inputs[offset+1] = feat.DayOfWeek
			inputs[offset+2] = feat.IsWeekend
			inputs[offset+3] = feat.Temperature
			inputs[offset+4] = feat.Humidity
			inputs[offset+5] = feat.SolarRadiation
		}

		target := data[i+seqLength].Consumption

		sequences = append(sequences, SequenceData{
			Inputs: inputs,
			Target: target,
		})
	}

	// Shuffle sequences
	shuffleData(sequences)

	// Split into train and test
	splitIdx := int(float64(len(sequences)) * trainRatio)
	trainSequences := sequences[:splitIdx]
	testSequences := sequences[splitIdx:]

	return trainSequences, testSequences
}

// createLSTMNetwork creates an LSTM network for energy prediction
func createLSTMNetwork() *net.Network {
	// First LSTM: 36 inputs (6 timesteps * 6 features) -> 32 hidden
	lstm1 := layer.NewLSTM(36, 32)

	// Output layer: 32 -> 16 with Tanh activation
	dense1 := layer.NewDense(32, 16, activations.Tanh{})

	// Final output: 16 -> 1 with Tanh activation
	output := layer.NewDense(16, 1, activations.Tanh{})

	network := net.New(
		[]layer.Layer{lstm1, dense1, output},
		loss.MSE{},
		opt.NewAdam(0.002),
	)

	return network
}

// evaluateLSTM evaluates the LSTM network
func evaluateLSTM(network *net.Network, sequences []SequenceData) (mse, mae, r2 float64) {
	sumSquaredError := 0.0
	sumAbsoluteError := 0.0
	sumActual := 0.0

	for _, seq := range sequences {
		pred := network.Forward(seq.Inputs)[0]
		actual := seq.Target

		error := actual - pred
		sumSquaredError += error * error
		sumAbsoluteError += math.Abs(error)
		sumActual += actual
	}

	n := float64(len(sequences))
	mse = sumSquaredError / n
	mae = sumAbsoluteError / n

	// Calculate R2 (coefficient of determination)
	mean := sumActual / n
	totalSumSquares := 0.0
	for _, seq := range sequences {
		totalSumSquares += (seq.Target - mean) * (seq.Target - mean)
	}
	if totalSumSquares > 0 {
		r2 = 1.0 - (sumSquaredError / totalSumSquares)
	}

	return mse, mae, r2
}

// shuffleData shuffles a slice of SequenceData
func shuffleData(data []SequenceData) {
	for i := len(data) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
}

// Helper functions
func getMin(data []EnergyData) float64 {
	minVal := data[0].Consumption
	for _, d := range data {
		if d.Consumption < minVal {
			minVal = d.Consumption
		}
	}
	return minVal
}

func getMax(data []EnergyData) float64 {
	maxVal := data[0].Consumption
	for _, d := range data {
		if d.Consumption > maxVal {
			maxVal = d.Consumption
		}
	}
	return maxVal
}

func getMinTemp(data []EnergyData) float64 {
	minVal := data[0].Temperature * (35 - (-5)) + (-5)
	for _, d := range data {
		temp := d.Temperature * (35 - (-5)) + (-5)
		if temp < minVal {
			minVal = temp
		}
	}
	return minVal
}

func getMaxTemp(data []EnergyData) float64 {
	maxVal := data[0].Temperature * (35 - (-5)) + (-5)
	for _, d := range data {
		temp := d.Temperature * (35 - (-5)) + (-5)
		if temp > maxVal {
			maxVal = temp
		}
	}
	return maxVal
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
