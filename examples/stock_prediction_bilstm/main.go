// Stock Prediction example using BiLSTM architecture.
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"

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

type SequenceSample struct {
	Input  []float64
	Target float64
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

func createSequences(prices []float64, lookback int) []SequenceSample {
	var sequences []SequenceSample
	for i := 0; i <= len(prices)-lookback-1; i++ {
		seq := SequenceSample{
			Input:  make([]float64, lookback),
			Target: prices[i+lookback],
		}
		copy(seq.Input, prices[i:i+lookback])
		sequences = append(sequences, seq)
	}
	return sequences
}

func createBiLSTMNetwork(lookback int, lstmUnits int) *net.Network {
	forward := layer.NewLSTM(lookback, lstmUnits)
	backward := layer.NewLSTM(lookback, lstmUnits)
	bilstm := layer.NewBidirectional(forward, backward)

	layers := []layer.Layer{
		bilstm,
		layer.NewDense(lstmUnits*2, 1, activations.Linear{}),
	}
	return net.New(layers, loss.MSE{}, &opt.SGD{LearningRate: 0.01})
}

func main() {
	fmt.Println("=== Stock Price Prediction using BiLSTM ===")
	const (
		lookback   = 10
		lstmUnits  = 32
		epochs     = 30
		trainRatio = 0.8
	)

	data, err := loadStockData("examples/stock_prediction_bilstm/datasets/GOOG.csv")
	if err != nil { log.Fatal(err) }

	prices := make([]float64, len(data))
	for i, d := range data { prices[i] = d.Close }

	scaledPrices, normParams := minMaxScale(prices)
	sequences := createSequences(scaledPrices, lookback)
	splitIdx := int(float64(len(sequences)) * trainRatio)
	trainSeq, testSeq := sequences[:splitIdx], sequences[splitIdx:]

	network := createBiLSTMNetwork(lookback, lstmUnits)

	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := 0.0
		for _, seq := range trainSeq {
			network.Layers()[0].Reset()
			totalLoss += network.Train(seq.Input, []float64{seq.Target})
		}
		if epoch%5 == 0 || epoch == 1 {
			fmt.Printf("Epoch %d, Loss: %.6f\n", epoch, totalLoss/float64(len(trainSeq)))
		}
	}

	// Evaluation
	var mse float64
	for _, seq := range testSeq {
		network.Layers()[0].Reset()
		pred := network.Forward(seq.Input)[0]
		diff := (pred - seq.Target) * (normParams.Max - normParams.Min)
		mse += diff * diff
	}
	fmt.Printf("\nTest RMSE: $%.2f\n", math.Sqrt(mse/float64(len(testSeq))))

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Memory Alloc: %v KB\n", m.Alloc/1024)
}
