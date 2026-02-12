// Package main - Car Price Prediction using Neural Network
// Demonstrates training a neural network from CSV data and testing inference.
//
// Usage:
//   go run cmd/carprice/main.go
//
// The example:
// 1. Generates synthetic car data (Brand, Model, Year, CC, Price)
// 2. Saves to cars.csv
// 3. Loads and preprocesses the data
// 4. Trains a neural network
// 5. Tests inference on new data
// 6. Saves the trained model
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

// Car represents a car record from CSV
type Car struct {
	Brand   string
	Model   string
	Year    int
	CC      int
	Price   float64
	YearMod float64 // Normalized year
	CCMod   float64 // Normalized cc
	PriceMod float64 // Normalized price
}

func main() {
	fmt.Println("=============================================================")
	fmt.Println("  GoNeuron - Car Price Prediction from CSV")
	fmt.Println("=============================================================")
	fmt.Println()

	// Step 1: Generate and save car data to CSV
	fmt.Println("--- Step 1: Generating Car Data ---")
	cars := generateCarData(500)
	saveToCSV(cars, "cars.csv")
	fmt.Printf("Generated %d car records saved to cars.csv\n", len(cars))
	fmt.Println()

	// Step 2: Load and preprocess data
	fmt.Println("--- Step 2: Loading and Preprocessing Data ---")
	cars, err := loadFromCSV("cars.csv")
	if err != nil {
		log.Fatal("Error loading CSV:", err)
	}
	fmt.Printf("Loaded %d car records\n", len(cars))
	fmt.Printf("Data range - Year: %d-%d, CC: %d-%d, Price: $%.0f-$%.0f\n",
		cars[0].Year, cars[len(cars)-1].Year,
		cars[0].CC, cars[len(cars)-1].CC,
		cars[0].Price, cars[len(cars)-1].Price)
	fmt.Println()

	// Step 3: Normalize data
	fmt.Println("--- Step 3: Normalizing Data ---")
	minYear, maxYear := findRange(cars, func(c Car) int { return c.Year })
	minCC, maxCC := findRange(cars, func(c Car) int { return c.CC })
	minPrice, maxPrice := findRange(cars, func(c Car) float64 { return c.Price })

	normalizeData(cars, float64(minYear), float64(maxYear), float64(minCC), float64(maxCC), minPrice, maxPrice)
	fmt.Printf("Normalized year range: [%.3f, %.3f]\n", cars[0].YearMod, cars[len(cars)-1].YearMod)
	fmt.Printf("Normalized CC range: [%.3f, %.3f]\n", cars[0].CCMod, cars[len(cars)-1].CCMod)
	fmt.Printf("Normalized price range: [%.3f, %.3f]\n", cars[0].PriceMod, cars[len(cars)-1].PriceMod)
	fmt.Println()

	// Step 4: Split data into training and testing
	fmt.Println("--- Step 4: Splitting Data (80%% train, 20%% test) ---")
	trainSize := int(float64(len(cars)) * 0.8)
	trainData := cars[:trainSize]
	testData := cars[trainSize:]

	fmt.Printf("Training samples: %d\n", len(trainData))
	fmt.Printf("Testing samples: %d\n", len(testData))
	fmt.Println()

	// Step 5: Train the neural network
	fmt.Println("--- Step 5: Training Neural Network ---")
	fmt.Println("Architecture: 2 -> 8 -> 6 -> 1")
	fmt.Println("  Inputs: YearNorm, CCNorm")
	fmt.Println("  Hidden: 8 neurons (Tanh), 6 neurons (Tanh)")
	fmt.Println("  Output: 1 neuron (ReLU for regression)")
	fmt.Println("Loss: MSE")
	fmt.Println("Optimizer: Adam (lr=0.001)")
	fmt.Println()

	network := createNetwork()

	epochs := 1000
	fmt.Println("Training progress:")
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for i := range trainData {
			input := []float64{
				trainData[i].YearMod,
				trainData[i].CCMod,
			}
			target := []float64{trainData[i].PriceMod}

			loss := network.Train(input, target)
			totalLoss += loss
		}

		if epoch%100 == 0 {
			avgLoss := totalLoss / float64(len(trainData))
			fmt.Printf("  Epoch %4d: Loss = %.6f\n", epoch, avgLoss)
		}
	}

	fmt.Println("Training complete!")
	fmt.Println()

	// Step 6: Test on training data
	fmt.Println("--- Step 6: Testing on Training Data ---")
	trainMSE := testNetwork(network, trainData)
	fmt.Printf("Training MSE (normalized): %.6f\n", trainMSE)
	fmt.Println()

	// Step 7: Test on new/unseen data
	fmt.Println("--- Step 7: Testing on New Data ---")
	testMSE := testNetwork(network, testData)
	fmt.Printf("Testing MSE (normalized): %.6f\n", testMSE)
	fmt.Println()

	// Step 8: Predict on specific examples
	fmt.Println("--- Step 8: Predictions on Test Examples ---")
	fmt.Println("Actual vs Predicted Prices:")
	fmt.Println("----------------------------------------")

	allCorrect := 0
	for i := 0; i < min(5, len(testData)); i++ {
		car := testData[i]
		input := []float64{car.YearMod, car.CCMod}

		normalizedPred := network.Forward(input)[0]
		actualPrice := denormalize(car.PriceMod, minPrice, maxPrice)
		predictedPrice := denormalize(normalizedPred, minPrice, maxPrice)

		fmt.Printf("%s %s %d %dCC: Actual=$%.0f, Predicted=$%.0f\n",
			car.Brand, car.Model, car.Year, car.CC,
			actualPrice, predictedPrice)

		errorPct := math.Abs(actualPrice-predictedPrice) / actualPrice * 100
		if errorPct < 20 {
			allCorrect++
		}
	}
	fmt.Printf("Accuracy (within 20%%): %d/%d\n", allCorrect, min(5, len(testData)))
	fmt.Println()

	// Step 9: Save the model
	fmt.Println("--- Step 9: Saving Trained Model ---")
	err = network.Save("car_price_model.bin")
	if err != nil {
		log.Fatal("Error saving model:", err)
	}
	fmt.Println("Model saved to car_price_model.bin")
	fmt.Println()

	// Step 10: Load and verify model
	fmt.Println("--- Step 10: Verifying Saved Model ---")
	loadedNetwork, _, err := net.Load("car_price_model.bin")
	if err != nil {
		log.Fatal("Error loading model:", err)
	}
	fmt.Println("Model loaded successfully!")

	fmt.Println("Verifying predictions match between original and loaded model:")
	for i := 0; i < 3; i++ {
		car := testData[i]
		input := []float64{car.YearMod, car.CCMod}

		originalPred := network.Forward(input)[0]
		loadedPred := loadedNetwork.Forward(input)[0]

		match := "MATCH"
		if math.Abs(originalPred-loadedPred) > 1e-6 {
			match = "MISMATCH"
		}
		fmt.Printf("  Sample %d: Original=%.4f, Loaded=%.4f [%s]\n", i+1, originalPred, loadedPred, match)
	}

	fmt.Println()
	fmt.Println("=============================================================")
	fmt.Println("  Car Price Prediction Complete!")
	fmt.Println("=============================================================")
	fmt.Println()
	fmt.Println("Summary:")
	fmt.Printf("  - Training samples: %d\n", len(trainData))
	fmt.Printf("  - Testing samples: %d\n", len(testData))
	fmt.Printf("  - Training MSE: %.6f\n", trainMSE)
	fmt.Printf("  - Testing MSE: %.6f\n", testMSE)
	fmt.Println("  - Model saved: car_price_model.bin")
	fmt.Println()
	fmt.Println("To use the trained model in your application:")
	fmt.Println("  1. Load with: network, _, err := net.Load(\"car_price_model.bin\")")
	fmt.Println("  2. Prepare input: []float64{yearNorm, ccNorm}")
	fmt.Println("  3. Predict: output := network.Forward(input)")
	fmt.Println("  4. Denormalize: price = denormalize(output[0], minPrice, maxPrice)")
}

func createNetwork() *net.Network {
	l1 := layer.NewDense(2, 8, activations.Tanh{})
	l2 := layer.NewDense(8, 6, activations.Tanh{})
	l3 := layer.NewDense(6, 1, activations.ReLU{})

	network := net.New(
		[]layer.Layer{l1, l2, l3},
		loss.MSE{},
		opt.NewAdam(0.01),
	)

	return network
}

func testNetwork(network *net.Network, data []Car) float64 {
	totalError := 0.0
	for _, car := range data {
		input := []float64{car.YearMod, car.CCMod}
		pred := network.Forward(input)[0]
		error := pred - car.PriceMod
		totalError += error * error
	}
	return totalError / float64(len(data))
}

func generateCarData(count int) []Car {
	rand.Seed(time.Now().UnixNano())

	brands := []string{"Toyota", "Ford", "Honda", "BMW", "Mercedes", "Audi", "Tesla", "Hyundai"}
	models := []string{"Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Truck", "Van"}

	cars := make([]Car, count)

	for i := 0; i < count; i++ {
		brand := brands[rand.Intn(len(brands))]
		model := models[rand.Intn(len(models))]
		year := 2000 + rand.Intn(25)
		cc := 1000 + rand.Intn(3500)

		// Price calculation based on year, CC, and brand
		yearFactor := float64(year-2000) / 25.0 // 0 to 1
		ccFactor := float64(cc-1000) / 3500.0   // 0 to 1

		// Base prices by brand
		var basePrice float64
		switch brand {
		case "BMW", "Mercedes", "Audi":
			basePrice = 45000
		case "Tesla":
			basePrice = 55000
		case "Ford", "Toyota", "Hyundai":
			basePrice = 25000
		default:
			basePrice = 30000
		}

		// Simple price formula with noise
		price := basePrice + (yearFactor*0.3+0.7)*basePrice + ccFactor*0.2*basePrice
		price *= 1.0 + (rand.Float64()-0.5)*0.2 // 20% noise

		cars[i] = Car{
			Brand:   brand,
			Model:   model,
			Year:    year,
			CC:      cc,
			Price:   price,
		}
	}

	return cars
}

func saveToCSV(cars []Car, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Error creating file:", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"Brand", "Model", "Year", "CC", "Price"})

	for _, car := range cars {
		writer.Write([]string{
			car.Brand,
			car.Model,
			strconv.Itoa(car.Year),
			strconv.Itoa(car.CC),
			strconv.FormatFloat(car.Price, 'f', 2, 64),
		})
	}
}

func loadFromCSV(filename string) ([]Car, error) {
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

	var cars []Car
	for i, record := range records {
		if i == 0 {
			continue
		}

		year, err := strconv.Atoi(record[2])
		if err != nil {
			return nil, fmt.Errorf("invalid year at row %d: %v", i, err)
		}

		cc, err := strconv.Atoi(record[3])
		if err != nil {
			return nil, fmt.Errorf("invalid CC at row %d: %v", i, err)
		}

		price, err := strconv.ParseFloat(record[4], 64)
		if err != nil {
			return nil, fmt.Errorf("invalid price at row %d: %v", i, err)
		}

		cars = append(cars, Car{
			Brand: record[0],
			Model: record[1],
			Year:  year,
			CC:    cc,
			Price: price,
		})
	}

	return cars, nil
}

func findRange[T int | float64](cars []Car, getVal func(Car) T) (min T, max T) {
	if len(cars) == 0 {
		return 0, 0
	}
	min = getVal(cars[0])
	max = getVal(cars[0])
	for _, car := range cars {
		val := getVal(car)
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	return min, max
}

func normalizeData(cars []Car, minYear, maxYear, minCC, maxCC, minPrice, maxPrice float64) {
	if maxYear == minYear {
		maxYear = minYear + 1
	}
	if maxCC == minCC {
		maxCC = minCC + 1
	}
	if maxPrice == minPrice {
		maxPrice = minPrice + 1
	}

	for i := range cars {
		cars[i].YearMod = (float64(cars[i].Year) - minYear) / (maxYear - minYear)
		cars[i].CCMod = (float64(cars[i].CC) - minCC) / (maxCC - minCC)
		cars[i].PriceMod = (cars[i].Price - minPrice) / (maxPrice - minPrice)
	}
}

func denormalize(normVal, minVal, maxVal float64) float64 {
	return normVal*(maxVal-minVal) + minVal
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
