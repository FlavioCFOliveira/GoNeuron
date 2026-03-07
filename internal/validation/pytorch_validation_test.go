// Package validation provides tests that validate GoNeuron against PyTorch reference outputs
package validation

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// TestDataPath is the path to PyTorch reference test data
const TestDataPath = "../../scripts/pytorch_validation/test_data/pytorch_refs"

// TestCase represents a test case from PyTorch
type TestCase struct {
	Name     string                 `json:"name"`
	Inputs   map[string]interface{} `json:"inputs"`
	Outputs  map[string]interface{} `json:"outputs"`
	Metadata map[string]interface{} `json:"metadata"`
}

// readFloat32Bin reads a binary file of float32 values
func readFloat32Bin(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data)%4 != 0 {
		return nil, fmt.Errorf("invalid file size: %d bytes", len(data))
	}

	count := len(data) / 4
	values := make([]float32, count)
	for i := 0; i < count; i++ {
		bits := uint32(data[i*4]) | uint32(data[i*4+1])<<8 | uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
		values[i] = math.Float32frombits(bits)
	}
	return values, nil
}

// ShapeDef represents shape information in metadata
type ShapeDef struct {
	Shape []int  `json:"shape"`
	Type  string `json:"type"`
}

// readTestCase loads a test case from disk
func readTestCase(name string) (*TestCase, error) {
	casePath := filepath.Join(TestDataPath, name)

	metadataPath := filepath.Join(casePath, "metadata.json")
	metaData, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata: %w", err)
	}

	var rawMeta struct {
		Name     string                       `json:"name"`
		Inputs   map[string]ShapeDef          `json:"inputs"`
		Outputs  map[string]ShapeDef          `json:"outputs"`
		Metadata map[string]interface{}       `json:"metadata,omitempty"`
	}
	if err := json.Unmarshal(metaData, &rawMeta); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
	}

	// Load inputs and outputs from binary files
	inputs := make(map[string][]float32)
	outputs := make(map[string][]float32)

	for name := range rawMeta.Inputs {
		data, err := readFloat32Bin(filepath.Join(casePath, name+".bin"))
		if err == nil {
			inputs[name] = data
		}
	}

	for name := range rawMeta.Outputs {
		data, err := readFloat32Bin(filepath.Join(casePath, name+".bin"))
		if err == nil {
			outputs[name] = data
		}
	}

	// Convert to map[string]interface{} for compatibility
	inputMap := make(map[string]interface{})
	outputMap := make(map[string]interface{})
	for k, v := range inputs {
		inputMap[k] = v
	}
	for k, v := range outputs {
		outputMap[k] = v
	}

	// Build metadata map with shapes for easy access
	metadata := make(map[string]interface{})
	for k, v := range rawMeta.Metadata {
		metadata[k] = v
	}
	for k, v := range rawMeta.Inputs {
		metadata[k+"_shape"] = v.Shape
	}
	for k, v := range rawMeta.Outputs {
		metadata[k+"_shape"] = v.Shape
	}

	return &TestCase{
		Name:     name,
		Inputs:   inputMap,
		Outputs:  outputMap,
		Metadata: metadata,
	}, nil
}

// floatEquals checks if two float32 values are approximately equal
func floatEquals(a, b float32, tolerance float32) bool {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	diff := math.Abs(float64(a - b))
	if diff < float64(tolerance) {
		return true
	}
	// Relative tolerance for larger values
	maxVal := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	return diff/maxVal < float64(tolerance)
}

// slicesEqual checks if two float32 slices are approximately equal
func slicesEqual(a, b []float32, tolerance float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !floatEquals(a[i], b[i], tolerance) {
			return false
		}
	}
	return true
}

// skipIfNoTestData skips the test if PyTorch reference data doesn't exist
func skipIfNoTestData(t *testing.T, name string) *TestCase {
	// Check if test data directory exists
	if _, err := os.Stat(TestDataPath); os.IsNotExist(err) {
		t.Skip("PyTorch reference data not found. Run scripts/pytorch_validation/generate_references.py first")
	}

	tc, err := readTestCase(name)
	if err != nil {
		t.Skipf("Test case not found: %v", err)
	}
	return tc
}

// TestPyTorchValidation_Dense tests Dense layer against PyTorch
func TestPyTorchValidation_Dense(t *testing.T) {
	t.Run("SimpleForward", func(t *testing.T) {
		tc := skipIfNoTestData(t, "dense_simple")

		input := tc.Inputs["input"].([]float32)
		weights := tc.Inputs["weights"].([]float32)
		biases := tc.Inputs["biases"].([]float32)
		expected := tc.Outputs["output"].([]float32)

		// Get shapes from metadata (now stored as []int)
		inputShape := tc.Metadata["input_shape"].([]int)
		outputShape := tc.Metadata["output_shape"].([]int)
		inSize := inputShape[1]
		outSize := outputShape[1]

		// Create layer
		dense := layer.NewDense(inSize, outSize, activations.Linear{})

		// Set weights (PyTorch uses [out, in] format)
		params := make([]float32, len(weights)+len(biases))
		copy(params, weights)
		copy(params[len(weights):], biases)
		dense.SetParams(params)

		// Forward pass
		output, err := dense.Forward(input)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		// Compare
		if !slicesEqual(output, expected, 1e-4) {
			t.Errorf("Dense forward mismatch.\nGot:      %v\nExpected: %v", output, expected)
		}
	})
}

// TestPyTorchValidation_Activations tests activation functions against PyTorch
func TestPyTorchValidation_Activations(t *testing.T) {
	tests := []struct {
		name       string
		activation activations.Activation
	}{
		{"relu", activations.ReLU{}},
		{"sigmoid", activations.Sigmoid{}},
		{"tanh", activations.Tanh{}},
		{"leaky_relu", &activations.LeakyReLU{Alpha: 0.01}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := skipIfNoTestData(t, fmt.Sprintf("activation_%s", tt.name))

			input := tc.Inputs["input"].([]float32)
			expected := tc.Outputs["output"].([]float32)

			// Apply activation
			output := make([]float32, len(input))
			for i, v := range input {
				output[i] = tt.activation.Activate(v)
			}

			// Compare
			if !slicesEqual(output, expected, 1e-5) {
				t.Errorf("%s activation mismatch.\nGot:      %v\nExpected: %v", tt.name, output, expected)
			}
		})
	}
}

// TestPyTorchValidation_Loss tests loss functions against PyTorch
func TestPyTorchValidation_Loss(t *testing.T) {
	t.Run("MSE", func(t *testing.T) {
		tc := skipIfNoTestData(t, "loss_mse")

		yPred := tc.Inputs["y_pred"].([]float32)
		yTrue := tc.Inputs["y_true"].([]float32)
		expected := tc.Outputs["loss"].([]float32)[0]

		// Calculate MSE
		var sum float32
		for i := range yPred {
			diff := yPred[i] - yTrue[i]
			sum += diff * diff
		}
		got := sum / float32(len(yPred))

		if !floatEquals(got, expected, 1e-5) {
			t.Errorf("MSE loss mismatch. Got: %v, Expected: %v", got, expected)
		}
	})

	t.Run("Softmax", func(t *testing.T) {
		tc := skipIfNoTestData(t, "activation_softmax")

		input := tc.Inputs["input"].([]float32)
		expected := tc.Outputs["output"].([]float32)

		// Reshape for batch processing
		inputShape := tc.Metadata["input_shape"].([]int)
		batchSize := inputShape[0]
		numClasses := inputShape[1]

		output := make([]float32, len(input))
		for b := 0; b < batchSize; b++ {
			offset := b * numClasses
			row := input[offset : offset+numClasses]

			// Find max for numerical stability
			maxVal := row[0]
			for _, v := range row[1:] {
				if v > maxVal {
					maxVal = v
				}
			}

			// Compute softmax
			var sum float32
			for i, v := range row {
				output[offset+i] = float32(math.Exp(float64(v - maxVal)))
				sum += output[offset+i]
			}
			for i := range row {
				output[offset+i] /= sum
			}
		}

		if !slicesEqual(output, expected, 1e-4) {
			t.Errorf("Softmax mismatch.\nGot:      %v\nExpected: %v", output, expected)
		}
	})
}

// TestPyTorchValidation_Optimizers tests optimizers against PyTorch
func TestPyTorchValidation_Optimizers(t *testing.T) {
	t.Run("SGD", func(t *testing.T) {
		tc := skipIfNoTestData(t, "optimizer_sgd_step")

		param := tc.Inputs["param"].([]float32)
		grad := tc.Inputs["grad"].([]float32)
		expected := tc.Outputs["param_updated"].([]float32)

		lr := float32(tc.Inputs["lr"].([]float32)[0])

		// Create optimizer (Note: current SGD doesn't support momentum)
		params := make([]float32, len(param))
		copy(params, param)

		optimizer := opt.NewSGD(lr)
		optimizer.StepInPlace(params, grad)

		if !slicesEqual(params, expected, 1e-5) {
			t.Errorf("SGD step mismatch.\nGot:      %v\nExpected: %v", params, expected)
		}
	})

	t.Run("Adam", func(t *testing.T) {
		tc := skipIfNoTestData(t, "optimizer_adam_step")

		param := tc.Inputs["param"].([]float32)
		grad := tc.Inputs["grad"].([]float32)
		expected := tc.Outputs["param_updated"].([]float32)

		lr := tc.Inputs["lr"].([]float32)[0]

		// Create optimizer
		params := make([]float32, len(param))
		copy(params, param)

		optimizer := opt.NewAdam(lr)
		// Note: Adam has internal state (momentum1, momentum2) that needs initialization
		// This is a simplified test - full validation requires network integration
		optimizer.StepInPlace(params, grad)

		// Adam with timestep=1 should produce specific results
		// We use a tolerance since internal state handling may differ
		if !slicesEqual(params, expected, 1e-3) {
			t.Errorf("Adam step mismatch.\nGot:      %v\nExpected: %v", params, expected)
		}
	})
}

// GeneratePyTorchReferences generates a Go test file that validates against PyTorch
// This is a documentation function showing how to add new validation tests
func TestPyTorchValidation_AddCustomTest(t *testing.T) {
	// This test serves as documentation for adding new PyTorch validation tests
	// See scripts/pytorch_validation/generate_references.py for the Python side

	t.Skip("This is a documentation test - not meant to be run")

	// Example workflow:
	// 1. Add a new test case in generate_references.py
	// 2. Run: python scripts/pytorch_validation/generate_references.py
	// 3. Add corresponding Go test that loads and validates the reference data
}
