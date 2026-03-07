// Stock Prediction example using Bidirectional LSTM architecture.
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

type StockData struct {
	Date   string
	Open   float32
	High   float32
	Low    float32
	Close  float32
	Volume float32
}

type NormalizationParams struct {
	Min float32
	Max float32
}

func loadStockData(filepath string) ([]StockData, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data []StockData
	// Skip header
	for _, record := range records[1:] {
		var sd StockData
		sd.Date = record[0]

		val, _ := strconv.ParseFloat(record[1], 32)
		sd.Open = float32(val)
		val, _ = strconv.ParseFloat(record[2], 32)
		sd.High = float32(val)
		val, _ = strconv.ParseFloat(record[3], 32)
		sd.Low = float32(val)
		val, _ = strconv.ParseFloat(record[4], 32)
		sd.Close = float32(val)
		val, _ = strconv.ParseFloat(record[6], 32)
		sd.Volume = float32(val)

		data = append(data, sd)
	}
	return data, nil
}

func multiMinMaxScale(data [][]float32) ([][]float32, []NormalizationParams) {
	if len(data) == 0 {
		return nil, nil
	}
	numFeatures := len(data[0])
	numSamples := len(data)
	params := make([]NormalizationParams, numFeatures)

	// Find min/max for each feature
	for j := 0; j < numFeatures; j++ {
		minVal, maxVal := data[0][j], data[0][j]
		for i := 0; i < numSamples; i++ {
			if data[i][j] < minVal { minVal = data[i][j] }
			if data[i][j] > maxVal { maxVal = data[i][j] }
		}
		params[j] = NormalizationParams{Min: minVal, Max: maxVal}
	}

	// Scale
	scaled := make([][]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		scaled[i] = make([]float32, numFeatures)
		for j := 0; j < numFeatures; j++ {
			if params[j].Max == params[j].Min {
				scaled[i][j] = 0
			} else {
				scaled[i][j] = (data[i][j] - params[j].Min) / (params[j].Max - params[j].Min)
			}
		}
	}
	return scaled, params
}

func main() {
	fmt.Println("=== Improved Stock Price Prediction using Stacked BiLSTM ===")
	const (
		lookback      = 20
		lstmUnits     = 32
		epochs        = 1000
		trainRatio    = 0.8
		batchSize     = 16
		learningRate  = 0.001
	)

	// Try to load a larger dataset if available, fallback to GOOG
	datasetPath := "examples/stock_prediction/datasets/TSLA.csv"
	if _, err := os.Stat(datasetPath); os.IsNotExist(err) {
		datasetPath = "examples/stock_prediction_bilstm/datasets/GOOG.csv"
	}

	fmt.Printf("Loading dataset: %s\n", datasetPath)
	data, err := loadStockData(datasetPath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d samples\n", len(data))

	featureData := make([][]float32, len(data))
	for i, d := range data {
		featureData[i] = []float32{d.Open, d.High, d.Low, d.Close, d.Volume}
	}

	scaledFeatures, normParams := multiMinMaxScale(featureData)
	// Target is Close price (index 3)
	closeIdx := 3

	var x [][]float32
	var y [][]float32
	for i := 0; i <= len(scaledFeatures)-lookback-1; i++ {
		// Flatten lookback sequence: [T-lookback, ..., T-1]
		sequence := make([]float32, 0, lookback*5)
		for t := 0; t < lookback; t++ {
			sequence = append(sequence, scaledFeatures[i+t]...)
		}
		x = append(x, sequence)
		y = append(y, []float32{scaledFeatures[i+lookback][closeIdx]})
	}

	splitIdx := int(float32(len(x)) * trainRatio)
	xTrain, xTest := x[:splitIdx], x[splitIdx:]
	yTrain, yTest := y[:splitIdx], y[splitIdx:]

	fmt.Printf("Training samples: %d, Test samples: %d\n", len(xTrain), len(xTest))

	// Architecture:
	// 1. BiLSTM (5 -> 32) x lookback steps -> Sequence (lookback * 64)
	// 2. Dropout
	// 3. BiLSTM (64 -> 32) x lookback steps -> Summary (64)
	// 4. Dense (64 -> 1)

	bilstm1 := layer.NewBidirectional(layer.NewLSTM(5, lstmUnits))
	unroller1 := layer.NewSequenceUnroller(bilstm1, lookback, true)

	dropout := layer.NewDropout(0.2, lookback*lstmUnits*2)

	bilstm2 := layer.NewBidirectional(layer.NewLSTM(lstmUnits*2, lstmUnits))
	unroller2 := layer.NewSequenceUnroller(bilstm2, lookback, false)

	dense := layer.NewDense(lstmUnits*2, 1, activations.Linear{})

	layers := []layer.Layer{
		unroller1,
		dropout,
		unroller2,
		dense,
	}

	optimizer := opt.NewAdam(learningRate)
	network := net.New(layers, loss.MSE{}, optimizer)

	fmt.Println("Training...")
	network.Fit(xTrain, yTrain, epochs, batchSize, net.Logger{Interval: 100})

	// Evaluation
	var mse float32
	network.SetTraining(false)
	for i := range xTest {
		out, _ := network.Forward(xTest[i])
		pred := out[0]
		// Denormalize
		actualPred := pred*(normParams[closeIdx].Max-normParams[closeIdx].Min) + normParams[closeIdx].Min
		actualTrue := yTest[i][0]*(normParams[closeIdx].Max-normParams[closeIdx].Min) + normParams[closeIdx].Min

		diff := actualPred - actualTrue
		mse += diff * diff
	}
	rmse := float32(math.Sqrt(float64(mse / float32(len(xTest)))))
	fmt.Printf("\nTest RMSE: $%.2f\n", rmse)
}
