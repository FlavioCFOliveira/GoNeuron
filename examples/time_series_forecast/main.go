// Multivariate Time Series Forecasting example: Uses LSTM to predict
// future values from multivariate time series data.
// This example generates synthetic data for demonstration.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// TimeSeriesGenerator generates synthetic multivariate time series data
type TimeSeriesGenerator struct {
	numFeatures int
	numSamples  int
	noiseLevel  float64
}

// NewTimeSeriesGenerator creates a new data generator
func NewTimeSeriesGenerator(numFeatures, numSamples int, noiseLevel float64) *TimeSeriesGenerator {
	return &TimeSeriesGenerator{
		numFeatures: numFeatures,
		numSamples:  numSamples,
		noiseLevel:  noiseLevel,
	}
}

// Generate creates synthetic multivariate time series data
// Each feature is a combination of sine/cosine waves with different frequencies
func (g *TimeSeriesGenerator) Generate() ([][]float64, error) {
	data := make([][]float64, g.numSamples)
	t := 0.0
	tick := 0.1

	for i := 0; i < g.numSamples; i++ {
		features := make([]float64, g.numFeatures)
		for j := 0; j < g.numFeatures; j++ {
			// Create multiple frequency components
			baseFreq := float64(j+1) * 0.5
			phase := float64(j) * math.Pi / 4.0

			// Combine sine waves with different frequencies
			value := math.Sin(baseFreq*t+phase) +
				0.5*math.Sin(2*baseFreq*t+phase) +
				0.25*math.Sin(0.5*baseFreq*t+phase)

			// Add trend component for some features
			if j%2 == 0 {
				value += 0.01 * float64(i) / float64(g.numSamples)
			}

			// Add noise
			value += g.noiseLevel * (rand.Float64() - 0.5)

			features[j] = value
		}
		data[i] = features
		t += tick
	}

	return data, nil
}

// Normalize normalizes data using min-max scaling
func Normalize(data [][]float64) ([][]float64, []struct{ Min, Max float64 }) {
	numFeatures := len(data[0])
	params := make([]struct{ Min, Max float64 }, numFeatures)

	// Find min/max for each feature
	for j := 0; j < numFeatures; j++ {
		minVal, maxVal := data[0][j], data[0][j]
		for i := 0; i < len(data); i++ {
			if data[i][j] < minVal {
				minVal = data[i][j]
			}
			if data[i][j] > maxVal {
				maxVal = data[i][j]
			}
		}
		params[j] = struct{ Min, Max float64 }{Min: minVal, Max: maxVal}
	}

	// Normalize data
	normalized := make([][]float64, len(data))
	for i := range normalized {
		normalized[i] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			minVal := params[j].Min
			maxVal := params[j].Max
			if maxVal-minVal != 0 {
				normalized[i][j] = (data[i][j] - minVal) / (maxVal - minVal)
			} else {
				normalized[i][j] = 0
			}
		}
	}

	return normalized, params
}

// Denormalize reverses normalization
func Denormalize(data []float64, params []struct{ Min, Max float64 }) []float64 {
	result := make([]float64, len(data))
	for i := 0; i < len(data); i++ {
		if params[i].Max-params[i].Min != 0 {
			result[i] = data[i]*(params[i].Max-params[i].Min) + params[i].Min
		} else {
			result[i] = params[i].Min
		}
	}
	return result
}

// CreateSequences creates input-output pairs for time series forecasting
func CreateSequences(data [][]float64, lookback int, forecastStep int) ([][]float64, [][]float64) {
	var X [][]float64
	var y [][]float64

	numFeatures := len(data[0])

	for i := 0; i < len(data)-lookback-forecastStep; i++ {
		// Input: flattened sequence [lookback * numFeatures]
		input := make([]float64, lookback*numFeatures)
		for t := 0; t < lookback; t++ {
			copy(input[t*numFeatures:(t+1)*numFeatures], data[i+t])
		}
		X = append(X, input)

		// Output: next forecastStep * numFeatures values
		output := make([]float64, forecastStep*numFeatures)
		for t := 0; t < forecastStep; t++ {
			copy(output[t*numFeatures:(t+1)*numFeatures], data[i+lookback+t])
		}
		y = append(y, output)
	}

	return X, y
}

func main() {
	fmt.Println("=== Multivariate Time Series Forecasting using LSTM ===")

	// Configuration
	const (
		numFeatures = 3           // Number of time series variables
		numSamples  = 500         // Total samples
		lookback    = 20          // History window
		forecastStep = 5           // Steps to predict ahead
		trainRatio  = 0.8         // Training ratio
		lstmUnits   = 64          // LSTM hidden units
		epochs      = 50          // Training epochs
		batchSize   = 16          // Batch size
		learningRate = 0.001      // Learning rate
	)

	// 1. Generate synthetic data
	fmt.Printf("\nGenerating synthetic time series data...\n")
	fmt.Printf("Features: %d, Samples: %d, Noise: 0.05\n", numFeatures, numSamples)

	generator := NewTimeSeriesGenerator(numFeatures, numSamples, 0.05)
	data, err := generator.Generate()
	if err != nil {
		fmt.Printf("Error generating data: %v\n", err)
		return
	}

	// 2. Normalize data
	fmt.Println("Normalizing data...")
	normalizedData, normParams := Normalize(data)

	// 3. Create sequences
	fmt.Printf("Creating sequences (lookback=%d, forecast=%d)...\n", lookback, forecastStep)
	X, y := CreateSequences(normalizedData, lookback, forecastStep)

	fmt.Printf("Total sequences: %d\n", len(X))

	// 4. Split into train/test
	splitIdx := int(float64(len(X)) * trainRatio)
	XTrain, XTest := X[:splitIdx], X[splitIdx:]
	yTrain, yTest := y[:splitIdx], y[splitIdx:]

	fmt.Printf("Training sequences: %d, Test sequences: %d\n", len(XTrain), len(XTest))

	// 5. Define Architecture
	// The input is a flattened sequence [lookback * numFeatures]
	// We use an Embedding-like approach: reshape to [lookback, numFeatures]
	// then process with LSTM
	const (
		inputSize  = numFeatures
		sequenceLen = lookback
		denseUnits  = 32
	)

	layers := []layer.Layer{
		// Input: [lookback * numFeatures] -> treated as sequence
		// SequenceUnroller with LSTM processes each time step
		layer.NewSequenceUnroller(
			layer.NewLSTM(inputSize, lstmUnits),
			sequenceLen, false),
		// Output: [lstmUnits]

		// Hidden dense layer
		layer.NewDense(lstmUnits, denseUnits, activations.Tanh{}),

		// Output layer predicting forecastStep * numFeatures values
		layer.NewDense(denseUnits, forecastStep*numFeatures, activations.Linear{}),
	}

	// 6. Initialize Network
	optimizer := opt.NewAdam(learningRate)
	network := net.New(layers, loss.MSE{}, optimizer)

	// 7. Training
	fmt.Println("\nStarting training...")
	fmt.Printf("Architecture: LSTM(%d) -> Dense(%d) -> Dense(%d)\n",
		lstmUnits, denseUnits, forecastStep*numFeatures)

	start := time.Now()

	scheduler := opt.NewReduceLROnPlateau(optimizer, 0.5, 5, 0.01, 1e-6)

	callbacks := []net.Callback{
		net.Logger{Interval: 1},
		net.NewModelCheckpoint("time_series_forecast_best.gob"),
		net.NewEarlyStopping(10, 0.0001),
		net.NewSchedulerCallback(scheduler),
	}

	network.Fit(XTrain, yTrain, epochs, batchSize, callbacks...)
	fmt.Printf("\nTraining finished in %v\n", time.Since(start))

	// 8. Evaluation
	fmt.Println("\nEvaluating on test set...")

	// Calculate RMSE
	var totalMSE float64
	for i := 0; i < len(XTest); i++ {
		pred := network.Forward(XTest[i])
		trueVal := yTest[i]

		for j := 0; j < len(pred); j++ {
			diff := pred[j] - trueVal[j]
			totalMSE += diff * diff
		}
	}

	rmse := math.Sqrt(totalMSE / float64(len(XTest)*forecastStep*numFeatures))
	fmt.Printf("Test RMSE (normalized): %.6f\n", rmse)

	// Denormalize and calculate actual RMSE
	var actualMSE float64
	for i := 0; i < min(len(XTest), 100); i++ {
		pred := network.Forward(XTest[i])

		// Denormalize predictions and actuals
		predDenorm := Denormalize(pred, normParams)
		actualDenorm := Denormalize(yTest[i], normParams)

		for j := 0; j < len(predDenorm); j++ {
			diff := predDenorm[j] - actualDenorm[j]
			actualMSE += diff * diff
		}
	}

	actualRMSE := math.Sqrt(actualMSE / float64(min(len(XTest), 100)*forecastStep*numFeatures))
	fmt.Printf("Test RMSE (actual): %.6f\n", actualRMSE)

	// 9. Sample predictions
	fmt.Println("\nSample Predictions (first 3 test samples):")
	for i := 0; i < 3; i++ {
		pred := network.Forward(XTest[i])
		trueVal := yTest[i]

		fmt.Printf("\nSample %d:\n", i+1)
		fmt.Printf("  True values:   %v\n", formatValues(Denormalize(trueVal, normParams), 10))
		fmt.Printf("  Predicted:     %v\n", formatValues(Denormalize(pred, normParams), 10))
	}

	// Save model
	network.Save("time_series_forecast_final.gob")
	fmt.Println("\nModel saved to time_series_forecast_final.gob")
}

func formatValues(values []float64, maxLen int) string {
	if len(values) <= maxLen {
		result := "["
		for i, v := range values {
			if i > 0 {
				result += ", "
			}
			result += fmt.Sprintf("%.3f", v)
		}
		result += "]"
		return result
	}

	result := "["
	for i := 0; i < maxLen/2; i++ {
		if i > 0 {
			result += ", "
		}
		result += fmt.Sprintf("%.3f", values[i])
	}
	result += ", ..., "
	for i := len(values) - maxLen/2; i < len(values); i++ {
		if i > len(values)-maxLen/2 {
			result += ", "
		}
		result += fmt.Sprintf("%.3f", values[i])
	}
	result += "]"
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Performance Notes for Multivariate Time Series Forecasting Architecture:
//
// Zero-Allocation Pattern:
// 1. LSTM pre-allocates all buffers (inputBuf, outputBuf, preActBuf, etc.)
// 2. SequenceUnroller pre-allocates inputBuf, outputBuf, gradInBuf ONCE
// 3. Dense layers use pre-allocated gradient buffers
// 4. All Forward/Backward passes reuse buffers - no allocations during training
//
// Architecture Design:
// - LSTM input size = numFeatures (each time step processes all features)
// - Hidden size = 64 (balance of capacity and efficiency)
// - Dense hidden = 32 units with Tanh
// - Linear output for regression (direct predictions)
// - MSE loss for regression
//
// Input Format:
// - Each input is flattened sequence [lookback * numFeatures]
// - SequenceUnroller reshapes internally to [lookback, numFeatures]
// - Each time step receives [numFeatures] values
//
// Expected Performance:
// - Good performance on synthetic periodic data
// - Zero allocations in hot path
// - Efficient cache-friendly processing
//
// For production use:
// - Real-world data often benefits from feature engineering
// - Consider adding attention mechanism for long sequences
// - Multiple LSTM layers for complex patterns
// - Dropout for regularization on noisy data
