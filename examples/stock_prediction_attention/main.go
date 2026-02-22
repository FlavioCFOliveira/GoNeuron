// Stock Prediction example using LSTM + Attention architecture with SequenceUnroller.
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
	Date  string
	Close float32
}

type NormalizationParams struct {
	Min float32
	Max float32
}

func loadStockData(filepath string) ([]StockData, error) {
	file, err := os.Open(filepath)
	if err != nil { return nil, err }
	defer file.Close()
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil { return nil, err }
	var data []StockData
	for _, record := range records[1:] {
		var sd StockData
		sd.Date = record[0]
		fmt.Sscanf(record[4], "%f", &sd.Close)
		data = append(data, sd)
	}
	return data, nil
}

func minMaxScale(prices []float32) ([]float32, NormalizationParams) {
	minVal, maxVal := prices[0], prices[0]
	for _, p := range prices {
		if p < minVal { minVal = p }
		if p > maxVal { maxVal = p }
	}
	scaled := make([]float32, len(prices))
	for i, p := range prices {
		scaled[i] = (p - minVal) / (maxVal - minVal)
	}
	return scaled, NormalizationParams{Min: minVal, Max: maxVal}
}

func main() {
	fmt.Println("=== Stock Price Prediction using LSTM + Attention ===")
	const (
		lookback   = 10
		lstmUnits  = 32
		epochs     = 100
		trainRatio = 0.8
	)

	data, err := loadStockData("examples/stock_prediction_attention/datasets/GOOG.csv")
	if err != nil { log.Fatal(err) }

	prices := make([]float32, len(data))
	for i, d := range data { prices[i] = d.Close }

	scaledPrices, normParams := minMaxScale(prices)

	var x [][]float32
	var y [][]float32
	for i := 0; i <= len(scaledPrices)-lookback-1; i++ {
		x = append(x, scaledPrices[i:i+lookback])
		y = append(y, []float32{scaledPrices[i+lookback]})
	}

	splitIdx := int(float32(len(x)) * trainRatio)
	xTrain, xTest := x[:splitIdx], x[splitIdx:]
	yTrain, yTest := y[:splitIdx], y[splitIdx:]

	layers := []layer.Layer{
		layer.NewSequenceUnroller(layer.NewLSTM(1, lstmUnits), lookback, true), // Returns [lookback * lstmUnits]
		layer.NewGlobalAttention(lstmUnits),
		layer.NewDense(lstmUnits, 1, activations.Linear{}),
	}

	optimizer := opt.NewAdam(0.001)
	network := net.New(layers, loss.MSE{}, optimizer)

	fmt.Println("Training...")
	network.Fit(xTrain, yTrain, epochs, 16, net.Logger{Interval: 20})

	// Evaluation
	var mse float32
	for i := range xTest {
		pred := network.Forward(xTest[i])[0]
		diff := (pred - yTest[i][0]) * (normParams.Max - normParams.Min)
		mse += diff * diff
	}
	fmt.Printf("\nTest RMSE: $%.2f\n", float32(math.Sqrt(float64(mse/float32(len(xTest))))))
}
