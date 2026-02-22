package utils

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"time"
)

// Iris represents the Iris dataset
type Iris struct {
	Inputs  [][]float32
	Targets [][]float32
}

// LoadIris loads the Iris dataset. It downloads it if not present.
func LoadIris() (*Iris, error) {
	const url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	const filename = "iris.csv"

	if _, err := os.Stat(filename); os.IsNotExist(err) {
		fmt.Println("Downloading Iris dataset...")
		resp, err := http.Get(url)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		f, err := os.Create(filename)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		_, err = f.ReadFrom(resp.Body)
		if err != nil {
			return nil, err
		}
	}

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

	inputs := make([][]float32, 0)
	targets := make([][]float32, 0)

	classMap := map[string]int{
		"Iris-setosa":     0,
		"Iris-versicolor": 1,
		"Iris-virginica":  2,
	}

	for _, record := range records {
		if len(record) < 5 {
			continue
		}
		input := make([]float32, 4)
		for i := 0; i < 4; i++ {
			val, _ := strconv.ParseFloat(record[i], 32)
			input[i] = float32(val)
		}
		classStr := record[4]
		classIdx, ok := classMap[classStr]
		if !ok {
			continue
		}

		target := make([]float32, 3)
		target[classIdx] = 1.0

		inputs = append(inputs, input)
		targets = append(targets, target)
	}

	return &Iris{Inputs: inputs, Targets: targets}, nil
}

// GenerateSineWave generates synthetic sine wave data for regression
func GenerateSineWave(n int) ([][]float32, [][]float32) {
	inputs := make([][]float32, n)
	targets := make([][]float32, n)
	for i := 0; i < n; i++ {
		x := rand.Float32() * 2 * float32(math.Pi)
		inputs[i] = []float32{x}
		targets[i] = []float32{float32(math.Sin(float64(x)))}
	}
	return inputs, targets
}

// NormalizeIris normalizes Iris inputs to [0, 1]
func (i *Iris) Normalize() {
	if len(i.Inputs) == 0 {
		return
	}
	min := make([]float32, 4)
	max := make([]float32, 4)
	for j := 0; j < 4; j++ {
		min[j] = i.Inputs[0][j]
		max[j] = i.Inputs[0][j]
	}

	for _, input := range i.Inputs {
		for j := 0; j < 4; j++ {
			if input[j] < min[j] {
				min[j] = input[j]
			}
			if input[j] > max[j] {
				max[j] = input[j]
			}
		}
	}

	for _, input := range i.Inputs {
		for j := 0; j < 4; j++ {
			if max[j]-min[j] > 0 {
				input[j] = (input[j] - min[j]) / (max[j] - min[j])
			}
		}
	}
}

// ShuffleIris shuffles the Iris dataset
func (i *Iris) Shuffle() {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for j := len(i.Inputs) - 1; j > 0; j-- {
		k := r.Intn(j + 1)
		i.Inputs[j], i.Inputs[k] = i.Inputs[k], i.Inputs[j]
		i.Targets[j], i.Targets[k] = i.Targets[k], i.Targets[j]
	}
}
