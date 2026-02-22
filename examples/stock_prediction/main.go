// Stock Prediction example using LSTM architecture with SequenceUnroller.
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

type StockData struct {
	Date     string
	Open     float32
	High     float32
	Low      float32
	Close    float32
	AdjClose float32
	Volume   int64
}

type NormalizationParams struct {
	Min float32
	Max float32
}

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

	var data []StockData
	for _, record := range records[1:] {
		var sd StockData
		sd.Date = record[0]
		fmt.Sscanf(record[1], "%f", &sd.Open)
		fmt.Sscanf(record[2], "%f", &sd.High)
		fmt.Sscanf(record[3], "%f", &sd.Low)
		fmt.Sscanf(record[4], "%f", &sd.Close)
		fmt.Sscanf(record[5], "%f", &sd.AdjClose)
		fmt.Sscanf(record[6], "%d", &sd.Volume)
		data = append(data, sd)
	}
	return data, nil
}

func minMaxScale(prices []float32) ([]float32, NormalizationParams) {
	minVal, maxVal := prices[0], prices[0]
	for _, p := range prices {
		if p < minVal {
			minVal = p
		}
		if p > maxVal {
			maxVal = p
		}
	}
	scaled := make([]float32, len(prices))
	for i, p := range prices {
		scaled[i] = (p - minVal) / (maxVal - minVal)
	}
	return scaled, NormalizationParams{Min: minVal, Max: maxVal}
}

func main() {
	const (
		lookback  = 10
		lstmUnits = 32
		epochs    = 100
	)

	csvPath := "examples/stock_prediction/datasets/GOOG.csv"
	data, err := loadStockData(csvPath)
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}

	prices := make([]float32, len(data))
	for i, d := range data {
		prices[i] = d.Close
	}

	scaledPrices, normParams := minMaxScale(prices)

	var x [][]float32
	var y [][]float32
	for i := 0; i <= len(scaledPrices)-lookback-1; i++ {
		x = append(x, scaledPrices[i:i+lookback])
		y = append(y, []float32{scaledPrices[i+lookback]})
	}

	splitIdx := int(float32(len(x)) * 0.8)
	xTrain, xTest := x[:splitIdx], x[splitIdx:]
	yTrain, yTest := y[:splitIdx], y[splitIdx:]

	layers := []layer.Layer{
		layer.NewSequenceUnroller(layer.NewLSTM(1, lstmUnits), lookback, false),
		layer.NewDense(lstmUnits, 1, activations.Linear{}),
	}

	optimizer := opt.NewAdam(0.001)
	model := net.New(layers, loss.MSE{}, optimizer)

	fmt.Println("Training...")
	model.Fit(xTrain, yTrain, epochs, 16, net.Logger{Interval: 20})

	fmt.Println("\nEvaluating...")
	var totalSE float32
	for i := range xTest {
		pred := model.Forward(xTest[i])
		actual := yTest[i][0]
		diff := (pred[0] - actual) * (normParams.Max - normParams.Min)
		totalSE += diff * diff

		if i < 5 {
			pActual := actual*(normParams.Max-normParams.Min) + normParams.Min
			pPred := pred[0]*(normParams.Max-normParams.Min) + normParams.Min
			fmt.Printf("Actual: $%.2f, Predicted: $%.2f, Error: $%.2f\n", pActual, pPred, pPred-pActual)
		}
	}
	fmt.Printf("Test RMSE: $%.2f\n", float32(math.Sqrt(float64(totalSE/float32(len(xTest))))))
}
