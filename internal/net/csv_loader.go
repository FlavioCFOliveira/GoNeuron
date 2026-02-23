package net

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

// Dataset represents a collection of samples and labels.
type Dataset struct {
	Samples [][]float32
	Labels  [][]float32
}

// LoadCSV loads data from a CSV file.
// labelCols specifies the indices of columns to be used as labels.
// All other columns are used as features.
// hasHeader skips the first line if true.
func LoadCSV(filename string, labelCols []int, hasHeader bool) (*Dataset, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read csv: %w", err)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("csv file is empty")
	}

	startRow := 0
	if hasHeader {
		startRow = 1
	}

	if len(records) <= startRow {
		return nil, fmt.Errorf("csv file has no data rows")
	}

	numCols := len(records[0])
	isLabelCol := make(map[int]bool)
	for _, col := range labelCols {
		isLabelCol[col] = true
	}

	numSamples := len(records) - startRow
	samples := make([][]float32, numSamples)
	labels := make([][]float32, numSamples)

	for i := startRow; i < len(records); i++ {
		record := records[i]
		if len(record) != numCols {
			return nil, fmt.Errorf("inconsistent number of columns at row %d", i)
		}

		sampleRow := make([]float32, 0, numCols-len(labelCols))
		labelRow := make([]float32, 0, len(labelCols))

		// We need to maintain the order of labels as specified in labelCols
		// but for features we just take everything else in order.

		// Map for quick access to label values by their index in the original record
		labelValues := make(map[int]float32)

		for j, valStr := range record {
			val, err := strconv.ParseFloat(valStr, 32)
			if err != nil {
				return nil, fmt.Errorf("failed to parse value at row %d, col %d: %w", i, j, err)
			}

			if isLabelCol[j] {
				labelValues[j] = float32(val)
			} else {
				sampleRow = append(sampleRow, float32(val))
			}
		}

		// Reconstruct label row in the order specified by labelCols
		for _, col := range labelCols {
			labelRow = append(labelRow, labelValues[col])
		}

		samples[i-startRow] = sampleRow
		labels[i-startRow] = labelRow
	}

	return &Dataset{
		Samples: samples,
		Labels:  labels,
	}, nil
}

// Normalize performs min-max normalization on the samples.
func (d *Dataset) Normalize() {
	if len(d.Samples) == 0 {
		return
	}

	numFeatures := len(d.Samples[0])
	min := make([]float32, numFeatures)
	max := make([]float32, numFeatures)

	for i := range min {
		min[i] = d.Samples[0][i]
		max[i] = d.Samples[0][i]
	}

	for _, sample := range d.Samples {
		for i, val := range sample {
			if val < min[i] {
				min[i] = val
			}
			if val > max[i] {
				max[i] = val
			}
		}
	}

	for _, sample := range d.Samples {
		for i := range sample {
			diff := max[i] - min[i]
			if diff != 0 {
				sample[i] = (sample[i] - min[i]) / diff
			} else {
				sample[i] = 0
			}
		}
	}
}

// Split splits the dataset into two based on the given ratio (0.0 to 1.0).
// Returns two new Datasets (train, test).
func (d *Dataset) Split(ratio float32) (*Dataset, *Dataset) {
	if ratio <= 0 {
		return &Dataset{}, d
	}
	if ratio >= 1 {
		return d, &Dataset{}
	}

	splitIdx := int(float32(len(d.Samples)) * ratio)

	train := &Dataset{
		Samples: d.Samples[:splitIdx],
		Labels:  d.Labels[:splitIdx],
	}

	test := &Dataset{
		Samples: d.Samples[splitIdx:],
		Labels:  d.Labels[splitIdx:],
	}

	return train, test
}
