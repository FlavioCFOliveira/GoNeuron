package net

import (
	"encoding/csv"
	"os"
	"reflect"
	"testing"
)

func TestCSVLoader(t *testing.T) {
	// Create a dummy CSV file
	filename := "test_loader.csv"
	file, err := os.Create(filename)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}
	defer os.Remove(filename)

	writer := csv.NewWriter(file)
	writer.Write([]string{"f1", "f2", "l1", "f3", "l2"})
	writer.Write([]string{"1.0", "2.0", "0.0", "3.0", "1.0"})
	writer.Write([]string{"4.0", "5.0", "1.0", "6.0", "0.0"})
	writer.Flush()
	file.Close()

	// Load the CSV
	labelCols := []int{2, 4}
	dataset, err := LoadCSV(filename, labelCols, true)
	if err != nil {
		t.Fatalf("LoadCSV failed: %v", err)
	}

	if len(dataset.Samples) != 2 {
		t.Errorf("expected 2 samples, got %d", len(dataset.Samples))
	}

	expectedSamples := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}
	if !reflect.DeepEqual(dataset.Samples, expectedSamples) {
		t.Errorf("expected samples %v, got %v", expectedSamples, dataset.Samples)
	}

	expectedLabels := [][]float32{
		{0.0, 1.0},
		{1.0, 0.0},
	}
	if !reflect.DeepEqual(dataset.Labels, expectedLabels) {
		t.Errorf("expected labels %v, got %v", expectedLabels, dataset.Labels)
	}
}

func TestCSVLogger(t *testing.T) {
	filename := "test_logger.csv"
	defer os.Remove(filename)

	logger := NewCSVLogger(filename, false)
	n := &Network{}

	logger.OnTrainBegin(n)
	logger.OnEpochEnd(0, 0.5, n)
	logger.OnEpochEnd(1, 0.4, n)
	logger.OnTrainEnd(n)

	// Read back the CSV
	file, err := os.Open(filename)
	if err != nil {
		t.Fatalf("failed to open logger file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("failed to read csv: %v", err)
	}

	if len(records) != 3 { // Header + 2 epochs
		t.Errorf("expected 3 records, got %d", len(records))
	}

	if records[1][0] != "0" || records[1][1] != "0.500000" {
		t.Errorf("unexpected record at epoch 0: %v", records[1])
	}
}

func TestDatasetNormalization(t *testing.T) {
	dataset := &Dataset{
		Samples: [][]float32{
			{10, 0},
			{20, 5},
			{30, 10},
		},
	}

	dataset.Normalize()

	expected := [][]float32{
		{0.0, 0.0},
		{0.5, 0.5},
		{1.0, 1.0},
	}

	for i := range expected {
		for j := range expected[i] {
			if dataset.Samples[i][j] != expected[i][j] {
				t.Errorf("at [%d][%d] expected %f, got %f", i, j, expected[i][j], dataset.Samples[i][j])
			}
		}
	}
}
