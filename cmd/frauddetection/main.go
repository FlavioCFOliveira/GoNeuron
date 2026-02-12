// Package main - Fraud Detection using Feedforward Neural Network
// Demonstrates training a neural network for bank fraud detection from CSV data.
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

// Transaction represents a transaction for fraud detection
type Transaction struct {
	Amount         float64 // Transaction amount
	TimeOfDay      float64 // Hour of day (0-23)
	DaysSince      int     // Days since card issued
	TransactionFreq float64 // Frequency of transactions
	IsInternational bool   // International transaction
	IsHighRiskCat  bool    // High-risk category
	HasRecentFraud  bool   // Recent fraud on account
	FraudLabel     int     // 1 = fraud, 0 = legitimate
}

// NormalizedTransaction represents normalized features for training
type NormalizedTransaction struct {
	Features []float64
	Target   float64
}

func main() {
	fmt.Println("=============================================================")
	fmt.Println("  GoNeuron - Bank Fraud Detection using FNN")
	fmt.Println("=============================================================")
	fmt.Println()

	// Step 1: Generate synthetic fraud detection data
	fmt.Println("--- Step 1: Generating Fraud Detection Data ---")
	transactions := generateFraudData(50000)
	saveToCSV(transactions, "fraud_data.csv")
	fmt.Printf("Generated %d transaction records saved to fraud_data.csv\n", len(transactions))
	fmt.Println()

	// Step 2: Load and prepare data
	fmt.Println("--- Step 2: Loading and Preparing Data ---")
	transactions, err := loadFromCSV("fraud_data.csv")
	if err != nil {
		log.Fatal("Error loading CSV:", err)
	}
	fmt.Printf("Loaded %d transaction records\n", len(transactions))

	fraudCount := 0
	for _, t := range transactions {
		if t.FraudLabel == 1 {
			fraudCount++
		}
	}
	fmt.Printf("Fraudulent transactions: %d (%.2f%%)\n", fraudCount, float64(fraudCount)/float64(len(transactions))*100)
	fmt.Printf("Legitimate transactions: %d (%.2f%%)\n", len(transactions)-fraudCount, float64(len(transactions)-fraudCount)/float64(len(transactions))*100)
	fmt.Println()

	// Step 3: Normalize and create training data
	fmt.Println("--- Step 3: Normalizing Data ---")
	normalizedData := normalizeData(transactions)

	// Split data
	trainSize := int(float64(len(normalizedData)) * 0.8)
	trainData := normalizedData[:trainSize]
	testData := normalizedData[trainSize:]
	fmt.Printf("Training samples: %d\n", len(trainData))
	fmt.Printf("Testing samples: %d\n", len(testData))
	fmt.Println()

		// Step 4: Create and train FNN
	fmt.Println("--- Step 4: Training Feedforward Neural Network ---")
	fmt.Println("Architecture: 7 -> 64 -> 48 -> 32 -> 16 -> 8 -> 1")
	fmt.Println("  Inputs: 7 features (normalized transaction data)")
	fmt.Println("  Hidden: 64 -> 48 -> 32 -> 16 -> 8 neurons (Tanh)")
	fmt.Println("  Output: 1 neuron (Tanh for better gradient flow)")
	fmt.Println("Loss: MSE")
	fmt.Println("Optimizer: Adam (lr=0.005)")
	fmt.Println()

	network := createFNN()

	epochs := 100
	batchSize := 32
	fmt.Println("Training progress:")
	lastLoss := 1.0

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		batchCount := 0

		shuffleData(trainData)

		for i := 0; i < len(trainData); i += batchSize {
			end := i + batchSize
			if end > len(trainData) {
				end = len(trainData)
			}

			batchX := make([][]float64, end-i)
			batchY := make([][]float64, end-i)

			for j := i; j < end; j++ {
				batchX[j-i] = trainData[j].Features
				batchY[j-i] = []float64{trainData[j].Target}
			}

			loss := network.TrainBatch(batchX, batchY)
			totalLoss += loss
			batchCount++
		}

		avgLoss := totalLoss / float64(batchCount)

		if epoch%20 == 0 {
			fmt.Printf("  Epoch %3d: Loss = %.6f\n", epoch, avgLoss)
		}
		if avgLoss < lastLoss-0.0001 {
			lastLoss = avgLoss
		}
	}

	fmt.Println("Training complete!")
	fmt.Printf("Final training loss: %.6f\n", lastLoss)
	fmt.Println()

	// Step 5: Evaluate model
	fmt.Println("--- Step 5: Evaluating Model ---")
	trainAcc, trainPrec, trainRec, trainF1 := evaluateFNN(network, trainData)
	fmt.Printf("Training Accuracy: %.2f%%\n", trainAcc*100)
	fmt.Printf("Training Precision: %.2f%%\n", trainPrec*100)
	fmt.Printf("Training Recall: %.2f%%\n", trainRec*100)
	fmt.Printf("Training F1 Score: %.2f%%\n", trainF1*100)

	testAcc, testPrec, testRec, testF1 := evaluateFNN(network, testData)
	fmt.Printf("Testing Accuracy: %.2f%%\n", testAcc*100)
	fmt.Printf("Testing Precision: %.2f%%\n", testPrec*100)
	fmt.Printf("Testing Recall: %.2f%%\n", testRec*100)
	fmt.Printf("Testing F1 Score: %.2f%%\n", testF1*100)
	fmt.Println()

	// Step 6: Sample predictions
	fmt.Println("--- Step 6: Sample Predictions ---")
	fmt.Println("Actual vs Predicted (Fraud Probability):")
	fmt.Println("----------------------------------------")

	truePos, falsePos, trueNeg, falseNeg := 0, 0, 0, 0
	for i := 0; i < min(15, len(testData)); i++ {
		tx := testData[i]
		prob := network.Forward(tx.Features)[0]

		predClass := 0
		if prob > 0.5 {
			predClass = 1
		}
		actualClass := int(tx.Target)

		if predClass == 1 && actualClass == 1 {
			truePos++
		} else if predClass == 1 && actualClass == 0 {
			falsePos++
		} else if predClass == 0 && actualClass == 0 {
			trueNeg++
		} else {
			falseNeg++
		}

		status := ""
		if predClass == actualClass {
			status = " [CORRECT]"
		} else {
			status = " [WRONG]"
		}

		fraudType := "LEGIT"
		if actualClass == 1 {
			fraudType = "FRAUD"
		}

		fmt.Printf("Sample %2d: %s, Prob=%.4f (%.1f%%)%s\n",
			i+1, fraudType, prob, prob*100, status)
	}

	fmt.Printf("\nConfusion Matrix (First 15 samples):\n")
	fmt.Printf("  True Positive:  %d\n", truePos)
	fmt.Printf("  False Positive: %d\n", falsePos)
	fmt.Printf("  True Negative:  %d\n", trueNeg)
	fmt.Printf("  False Negative: %d\n", falseNeg)
	fmt.Println()

	// Full evaluation
	truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
	for _, tx := range testData {
		prob := network.Forward(tx.Features)[0]

		predClass := 0
		if prob > 0.5 {
			predClass = 1
		}
		actualClass := int(tx.Target)

		if predClass == 1 && actualClass == 1 {
			truePos++
		} else if predClass == 1 && actualClass == 0 {
			falsePos++
		} else if predClass == 0 && actualClass == 0 {
			trueNeg++
		} else {
			falseNeg++
		}
	}

	overallAcc := float64(truePos+trueNeg) / float64(len(testData))
	fmt.Printf("\nFull Evaluation on Test Set (%d samples):\n", len(testData))
	fmt.Printf("  Accuracy: %.2f%%\n", overallAcc*100)
	if truePos+falsePos > 0 {
		precision := float64(truePos) / float64(truePos+falsePos)
		fmt.Printf("  Precision: %.2f%%\n", precision*100)
	}
	if truePos+falseNeg > 0 {
		recall := float64(truePos) / float64(truePos+falseNeg)
		fmt.Printf("  Recall: %.2f%%\n", recall*100)
	}
	fmt.Println()

	// Step 7: Save trained model
	fmt.Println("--- Step 7: Saving Trained Model ---")
	err = network.Save("fraud_detection_model.bin")
	if err != nil {
		log.Fatal("Error saving model:", err)
	}
	fmt.Println("Model saved to fraud_detection_model.bin")
	fmt.Println()

	// Step 8: Verify saved model
	fmt.Println("--- Step 8: Verifying Saved Model ---")
	loadedNetwork, _, err := net.Load("fraud_detection_model.bin")
	if err != nil {
		log.Fatal("Error loading model:", err)
	}
	fmt.Println("Model loaded successfully!")

	allMatch := true
	for i := 0; i < 5; i++ {
		tx := testData[i]
		originalPred := network.Forward(tx.Features)[0]
		loadedPred := loadedNetwork.Forward(tx.Features)[0]
		match := "MATCH"
		if math.Abs(originalPred-loadedPred) > 1e-6 {
			match = "MISMATCH"
			allMatch = false
		}
		fmt.Printf("  Sample %d: Original=%.6f, Loaded=%.6f [%s]\n", i+1, originalPred, loadedPred, match)
	}
	fmt.Println()

	// Step 9: Demonstrate inference on new transactions
	fmt.Println("--- Step 9: Inference on New Transactions ---")
	fmt.Println("Testing inference on newly generated transactions...")

	// Generate some test transactions
	newTx := []NormalizedTransaction{
		{Features: normalizeFeatures(15000, 3, 30, 0.1, true, true, true), Target: 1}, // Likely fraud
		{Features: normalizeFeatures(50, 14, 730, 0.5, false, false, false), Target: 0}, // Likely legitimate
		{Features: normalizeFeatures(8000, 2, 100, 0.2, true, true, false), Target: 1}, // Likely fraud
	}

	for i, tx := range newTx {
		prob := network.Forward(tx.Features)[0]
		predClass := 0
		if prob > 0.5 {
			predClass = 1
		}

		expected := "FRAUD"
		if tx.Target == 0 {
			expected = "LEGIT"
		}

		predicted := "FRAUD"
		if predClass == 0 {
			predicted = "LEGIT"
		}

		match := ""
		if predClass == int(tx.Target) {
			match = " [CORRECT]"
		} else {
			match = " [WRONG]"
		}

		fmt.Printf("Transaction %d (Expected %s):\n", i+1, expected)
		fmt.Printf("  Features: Amount=$%.0f, Time=%d:00, Days=%d\n", tx.Features[0]*20000, int(tx.Features[1]*24), int(tx.Features[2]))
		fmt.Printf("  Predicted: %s, Prob=%.4f (%.1f%%)%s\n", predicted, prob, prob*100, match)
	}
	fmt.Println()

	fmt.Println("=============================================================")
	if allMatch {
		fmt.Println("  SUCCESS: Model verification passed!")
	}
	fmt.Println("  Bank Fraud Detection Complete!")
	fmt.Println("=============================================================")
	fmt.Println()
	fmt.Printf("Summary:\n")
	fmt.Printf("  - Training samples: %d\n", len(trainData))
	fmt.Printf("  - Testing samples: %d\n", len(testData))
	fmt.Printf("  - Testing Accuracy: %.2f%%\n", testAcc*100)
	fmt.Printf("  - Testing Precision: %.2f%%\n", testPrec*100)
	fmt.Printf("  - Testing Recall: %.2f%%\n", testRec*100)
	fmt.Printf("  - Testing F1 Score: %.2f%%\n", testF1*100)
	fmt.Printf("  - Model saved: fraud_detection_model.bin\n")
}

// generateFraudData generates synthetic fraud detection data with clear patterns
func generateFraudData(count int) []Transaction {
	rand.Seed(42)

	transactions := make([]Transaction, count)

	// Generate data with clear patterns:
	// Fraud tends to have: high amounts, unusual hours, international, high-risk categories
	// Legitimate tends to have: normal amounts, business hours, domestic, common categories

	for i := 0; i < count; i++ {
		// 30% fraud rate for a challenging but learnable problem
		isFraud := rand.Float64() < 0.3

		var amount float64
		var timeOfDay float64
		var daysSince int
		var transactionFreq float64
		var isInternational bool
		var isHighRiskCat bool
		var hasRecentFraud bool

		if isFraud {
			// Fraud patterns - clear distinguishing features
			amount = 5000 + rand.Float64()*15000 // High amounts ($5k-$20k)
			timeOfDay = rand.Float64() * 6       // Early morning (0-6 AM) - 25% of transactions
			daysSince = 10 + rand.Intn(60)       // New card (10-70 days)
			transactionFreq = 0.05 + rand.Float64()*0.15 // Low frequency
			isInternational = rand.Float64() < 0.7 // 70% international
			isHighRiskCat = rand.Float64() < 0.6   // 60% high-risk category
			hasRecentFraud = rand.Float64() < 0.25 // 25% have recent fraud
		} else {
			// Legitimate patterns
			amount = 10 + rand.Float64()*400     // Normal amounts ($10-$410)
			timeOfDay = 7 + rand.Float64()*14    // Business hours (7-21)
			daysSince = 365 + rand.Intn(2*365)   // Card held 1-3 years
			transactionFreq = 0.3 + rand.Float64()*0.4 // Normal frequency
			isInternational = rand.Float64() < 0.15 // 15% international
			isHighRiskCat = rand.Float64() < 0.1   // 10% high-risk category
			hasRecentFraud = rand.Float64() < 0.02 // 2% have recent fraud
		}

		transactions[i] = Transaction{
			Amount:         amount,
			TimeOfDay:      timeOfDay,
			DaysSince:      daysSince,
			TransactionFreq: transactionFreq,
			IsInternational: isInternational,
			IsHighRiskCat:  isHighRiskCat,
			HasRecentFraud: hasRecentFraud,
			FraudLabel:     boolToInt(isFraud),
		}
	}

	return transactions
}

// saveToCSV saves transactions to CSV
func saveToCSV(transactions []Transaction, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Error creating file:", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{
		"Amount", "TimeOfDay", "DaysSince", "TransactionFreq",
		"IsInternational", "IsHighRiskCat", "HasRecentFraud", "FraudLabel",
	}
	writer.Write(header)

	// Write data
	for _, tx := range transactions {
		row := []string{
			strconv.FormatFloat(tx.Amount, 'f', 2, 64),
			strconv.FormatFloat(tx.TimeOfDay, 'f', 2, 64),
			strconv.Itoa(tx.DaysSince),
			strconv.FormatFloat(tx.TransactionFreq, 'f', 6, 64),
			strconv.FormatBool(tx.IsInternational),
			strconv.FormatBool(tx.IsHighRiskCat),
			strconv.FormatBool(tx.HasRecentFraud),
			strconv.Itoa(tx.FraudLabel),
		}
		writer.Write(row)
	}
}

// loadFromCSV loads transactions from CSV
func loadFromCSV(filename string) ([]Transaction, error) {
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

	var transactions []Transaction
	for i, record := range records {
		if i == 0 {
			continue // Skip header
		}

		amount, _ := strconv.ParseFloat(record[0], 64)
		timeOfDay, _ := strconv.ParseFloat(record[1], 64)
		daysSince, _ := strconv.Atoi(record[2])
		transactionFreq, _ := strconv.ParseFloat(record[3], 64)
		isInternational, _ := strconv.ParseBool(record[4])
		isHighRiskCat, _ := strconv.ParseBool(record[5])
		hasRecentFraud, _ := strconv.ParseBool(record[6])
		fraudLabel, _ := strconv.Atoi(record[7])

		transactions = append(transactions, Transaction{
			Amount:         amount,
			TimeOfDay:      timeOfDay,
			DaysSince:      daysSince,
			TransactionFreq: transactionFreq,
			IsInternational: isInternational,
			IsHighRiskCat:  isHighRiskCat,
			HasRecentFraud: hasRecentFraud,
			FraudLabel:     fraudLabel,
		})
	}

	return transactions, nil
}

// normalizeData normalizes transaction data for training
func normalizeData(transactions []Transaction) []NormalizedTransaction {
	// Find ranges for normalization
	var minAmount, maxAmount float64 = math.MaxFloat64, 0
	var minTime, maxTime float64 = math.MaxFloat64, 0
	var minDays, maxDays int = math.MaxInt32, 0
	var minFreq, maxFreq float64 = math.MaxFloat64, 0

	for _, tx := range transactions {
		if tx.Amount < minAmount {
			minAmount = tx.Amount
		}
		if tx.Amount > maxAmount {
			maxAmount = tx.Amount
		}
		if tx.TimeOfDay < minTime {
			minTime = tx.TimeOfDay
		}
		if tx.TimeOfDay > maxTime {
			maxTime = tx.TimeOfDay
		}
		if tx.DaysSince < minDays {
			minDays = tx.DaysSince
		}
		if tx.DaysSince > maxDays {
			maxDays = tx.DaysSince
		}
		if tx.TransactionFreq < minFreq {
			minFreq = tx.TransactionFreq
		}
		if tx.TransactionFreq > maxFreq {
			maxFreq = tx.TransactionFreq
		}
	}

	// Create normalized data
	normalized := make([]NormalizedTransaction, len(transactions))
	for i, tx := range transactions {
		features := make([]float64, 7)

		// Normalize amount (0-20000 range)
		features[0] = normalize(tx.Amount, minAmount, maxAmount)

		// Normalize time of day (0-24 range)
		features[1] = normalize(tx.TimeOfDay, minTime, maxTime)

		// Normalize days since (0-1500 range)
		features[2] = normalize(float64(tx.DaysSince), float64(minDays), float64(maxDays))

		// Normalize frequency (0-1 range)
		features[3] = normalize(tx.TransactionFreq, minFreq, maxFreq)

		// One-hot encode international (0 or 1)
		features[4] = boolToFloat(tx.IsInternational)

		// One-hot encode high-risk category (0 or 1)
		features[5] = boolToFloat(tx.IsHighRiskCat)

		// One-hot encode recent fraud (0 or 1)
		features[6] = boolToFloat(tx.HasRecentFraud)

		normalized[i] = NormalizedTransaction{
			Features: features,
			Target:   float64(tx.FraudLabel),
		}
	}

	return normalized
}

// createFNN creates a feedforward neural network for fraud detection
func createFNN() *net.Network {
	// 7 inputs -> 64 -> 48 -> 32 -> 16 -> 8 -> 1 output
	// Using Tanh for all layers to avoid saturation issues with Sigmoid
	l1 := layer.NewDense(7, 64, activations.Tanh{})
	l2 := layer.NewDense(64, 48, activations.Tanh{})
	l3 := layer.NewDense(48, 32, activations.Tanh{})
	l4 := layer.NewDense(32, 16, activations.Tanh{})
	l5 := layer.NewDense(16, 8, activations.Tanh{})
	l6 := layer.NewDense(8, 1, activations.Tanh{}) // Tanh output for better gradient flow

	network := net.New(
		[]layer.Layer{l1, l2, l3, l4, l5, l6},
		loss.MSE{}, // MSE loss works well with Tanh output
		opt.NewAdam(0.005),   // Lower learning rate for stable training
	)

	return network
}

// evaluateFNN evaluates the FNN model
func evaluateFNN(network *net.Network, data []NormalizedTransaction) (acc, prec, rec, f1 float64) {
	truePos, falsePos, trueNeg, falseNeg := 0, 0, 0, 0

	for _, tx := range data {
		prob := network.Forward(tx.Features)[0]

		predClass := 0
		if prob > 0.5 {
			predClass = 1
		}
		actualClass := int(tx.Target)

		if predClass == 1 && actualClass == 1 {
			truePos++
		} else if predClass == 1 && actualClass == 0 {
			falsePos++
		} else if predClass == 0 && actualClass == 0 {
			trueNeg++
		} else {
			falseNeg++
		}
	}

	total := truePos + falsePos + trueNeg + falseNeg
	acc = float64(truePos+trueNeg) / float64(total)

	if truePos+falsePos > 0 {
		prec = float64(truePos) / float64(truePos+falsePos)
	}
	if truePos+falseNeg > 0 {
		rec = float64(truePos) / float64(truePos+falseNeg)
	}
	if prec+rec > 0 {
		f1 = 2 * prec * rec / (prec + rec)
	}

	return acc, prec, rec, f1
}

// shuffleData shuffles a slice of NormalizedTransaction
func shuffleData(data []NormalizedTransaction) {
	for i := len(data) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
}

// normalize normalizes a value to [0,1] range
func normalize(val, min, max float64) float64 {
	if max == min {
		return 0.5
	}
	return (val - min) / (max - min)
}

// boolToFloat converts bool to float64
func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

// boolToInt converts bool to int
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// normalizeFeatures is a helper for new transaction inference
func normalizeFeatures(amount float64, timeOfDay float64, daysSince int, freq float64,
	isIntl, isHighRisk, hasRecentFraud bool) []float64 {
	features := make([]float64, 7)
	features[0] = normalize(amount, 0, 20000)
	features[1] = normalize(timeOfDay, 0, 24)
	features[2] = normalize(float64(daysSince), 0, 1500)
	features[3] = normalize(freq, 0, 1)
	features[4] = boolToFloat(isIntl)
	features[5] = boolToFloat(isHighRisk)
	features[6] = boolToFloat(hasRecentFraud)
	return features
}

// min returns the smaller of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
