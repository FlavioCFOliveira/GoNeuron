// Stock Prediction example using Bidirectional LSTM architecture.
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
	Close float64
}

type NormalizationParams struct {
	Min float64
	Max float64
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

func minMaxScale(prices []float64) ([]float64, NormalizationParams) {
	minVal, maxVal := prices[0], prices[0]
	for _, p := range prices {
		if p < minVal { minVal = p }
		if p > maxVal { maxVal = p }
	}
	scaled := make([]float64, len(prices))
	for i, p := range prices {
		scaled[i] = (p - minVal) / (maxVal - minVal)
	}
	return scaled, NormalizationParams{Min: minVal, Max: maxVal}
}

func main() {
	fmt.Println("=== Stock Price Prediction using BiLSTM ===")
	const (
		lookback   = 10
		lstmUnits  = 32
		epochs     = 50
		trainRatio = 0.8
	)

	data, err := loadStockData("examples/stock_prediction_bilstm/datasets/GOOG.csv")
	if err != nil { log.Fatal(err) }

	prices := make([]float64, len(data))
	for i, d := range data { prices[i] = d.Close }

	scaledPrices, normParams := minMaxScale(prices)

	var x [][]float64
	var y [][]float64
	for i := 0; i <= len(scaledPrices)-lookback-1; i++ {
		x = append(x, scaledPrices[i:i+lookback])
		y = append(y, []float64{scaledPrices[i+lookback]})
	}

	splitIdx := int(float64(len(x)) * trainRatio)
	xTrain, xTest := x[:splitIdx], x[splitIdx:]
	yTrain, yTest := y[:splitIdx], y[splitIdx:]

	lstm := layer.NewLSTM(1, lstmUnits)
	bilstm := layer.NewBidirectional(lstm)

	layers := []layer.Layer{
		layer.NewSequenceUnroller(bilstm, lookback, false),
		layer.NewDense(lstmUnits*2, 1, activations.Linear{}),
	}

	optimizer := opt.NewAdam(0.001)
	network := net.New(layers, loss.MSE{}, optimizer)

	fmt.Println("Training...")
	network.Fit(xTrain, yTrain, epochs, 16, net.Logger{Interval: 10})

	// Evaluation
	var mse float64
	for i := range xTest {
		pred := network.Forward(xTest[i])[0]
		diff := (pred - yTest[i][0]) * (normParams.Max - normParams.Min)
		mse += diff * diff
	}
	fmt.Printf("\nTest RMSE: $%.2f\n", math.Sqrt(mse/float64(len(xTest))))
}
