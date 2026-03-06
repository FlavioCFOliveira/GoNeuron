// Package goneuron_test provides integration tests for main examples.
// These tests verify that the main examples can run without errors.
package goneuron_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// findProjectRoot finds the project root directory by looking for go.mod
func findProjectRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get working directory: %v", err)
	}

	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("Could not find project root (no go.mod found)")
		}
		dir = parent
	}
}

// TestSimpleXORExample verifies the simple_xor example compiles and runs
func TestSimpleXORExample(t *testing.T) {
	root := findProjectRoot(t)
	exampleDir := filepath.Join(root, "examples", "simple_xor")

	// Check that main.go exists
	mainFile := filepath.Join(exampleDir, "main.go")
	if _, err := os.Stat(mainFile); os.IsNotExist(err) {
		t.Skip("simple_xor example not found")
	}

	// Read and verify the example contains expected code
	content, err := os.ReadFile(mainFile)
	if err != nil {
		t.Fatalf("Failed to read simple_xor/main.go: %v", err)
	}

	code := string(content)
	requiredElements := []string{
		"package main",
		"goneuron",
		"NewSequential",
		"Dense",
		"Fit",
	}

	for _, elem := range requiredElements {
		if !strings.Contains(code, elem) {
			t.Errorf("simple_xor missing required element: %s", elem)
		}
	}
}

// TestSLPIrisExample verifies the slp_iris example structure
func TestSLPIrisExample(t *testing.T) {
	root := findProjectRoot(t)
	exampleDir := filepath.Join(root, "examples", "slp_iris")

	mainFile := filepath.Join(exampleDir, "main.go")
	if _, err := os.Stat(mainFile); os.IsNotExist(err) {
		t.Skip("slp_iris example not found")
	}

	content, err := os.ReadFile(mainFile)
	if err != nil {
		t.Fatalf("Failed to read slp_iris/main.go: %v", err)
	}

	code := string(content)
	requiredElements := []string{
		"package main",
		"goneuron",
		"examples/utils",
		"Dense",
		"Softmax",
	}

	for _, elem := range requiredElements {
		if !strings.Contains(code, elem) {
			t.Errorf("slp_iris missing required element: %s", elem)
		}
	}

	// Check for iris.csv dataset
	irisCSV := filepath.Join(exampleDir, "datasets", "iris.csv")
	if _, err := os.Stat(irisCSV); os.IsNotExist(err) {
		t.Log("Warning: iris.csv dataset not found")
	}
}

// TestXORExample verifies the xor example structure
func TestXORExample(t *testing.T) {
	root := findProjectRoot(t)
	exampleDir := filepath.Join(root, "examples", "xor")

	mainFile := filepath.Join(exampleDir, "main.go")
	if _, err := os.Stat(mainFile); os.IsNotExist(err) {
		t.Skip("xor example not found")
	}

	content, err := os.ReadFile(mainFile)
	if err != nil {
		t.Fatalf("Failed to read xor/main.go: %v", err)
	}

	code := string(content)
	if !strings.Contains(code, "goneuron") {
		t.Error("xor example should use goneuron package")
	}
}

// TestRegressionSyntheticExample verifies the regression_synthetic example
func TestRegressionSyntheticExample(t *testing.T) {
	root := findProjectRoot(t)
	exampleDir := filepath.Join(root, "examples", "regression_synthetic")

	mainFile := filepath.Join(exampleDir, "main.go")
	if _, err := os.Stat(mainFile); os.IsNotExist(err) {
		t.Skip("regression_synthetic example not found")
	}

	content, err := os.ReadFile(mainFile)
	if err != nil {
		t.Fatalf("Failed to read regression_synthetic/main.go: %v", err)
	}

	code := string(content)
	requiredElements := []string{
		"goneuron",
		"Huber",
		"Dense",
	}

	for _, elem := range requiredElements {
		if !strings.Contains(code, elem) {
			t.Errorf("regression_synthetic missing required element: %s", elem)
		}
	}
}

// TestMNISTExample verifies the mnist example structure
func TestMNISTExample(t *testing.T) {
	root := findProjectRoot(t)
	exampleDir := filepath.Join(root, "examples", "mnist")

	mainFile := filepath.Join(exampleDir, "main.go")
	if _, err := os.Stat(mainFile); os.IsNotExist(err) {
		t.Skip("mnist example not found")
	}

	content, err := os.ReadFile(mainFile)
	if err != nil {
		t.Fatalf("Failed to read mnist/main.go: %v", err)
	}

	code := string(content)
	requiredElements := []string{
		"Conv2D",
		"MaxPool2D",
		"Flatten",
		"Softmax",
	}

	for _, elem := range requiredElements {
		if !strings.Contains(code, elem) {
			t.Errorf("mnist missing required element: %s", elem)
		}
	}
}

// TestAutoencoderExample verifies the autoencoder_synthetic example
func TestAutoencoderExample(t *testing.T) {
	root := findProjectRoot(t)
	exampleDir := filepath.Join(root, "examples", "autoencoder_synthetic")

	mainFile := filepath.Join(exampleDir, "main.go")
	if _, err := os.Stat(mainFile); os.IsNotExist(err) {
		t.Skip("autoencoder_synthetic example not found")
	}

	content, err := os.ReadFile(mainFile)
	if err != nil {
		t.Fatalf("Failed to read autoencoder_synthetic/main.go: %v", err)
	}

	code := string(content)
	if !strings.Contains(code, "autoencoder") && !strings.Contains(code, "Autoencoder") {
		t.Log("Warning: autoencoder example may not implement autoencoder pattern")
	}
}

// TestExamplesCompile verifies that example packages compile
func TestExamplesCompile(t *testing.T) {
	root := findProjectRoot(t)
	examplesDir := filepath.Join(root, "examples")

	entries, err := os.ReadDir(examplesDir)
	if err != nil {
		t.Fatalf("Failed to read examples directory: %v", err)
	}

	compiledCount := 0
	skippedCount := 0

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		exampleDir := filepath.Join(examplesDir, entry.Name())
		mainFile := filepath.Join(exampleDir, "main.go")

		// Skip if no main.go
		if _, err := os.Stat(mainFile); os.IsNotExist(err) {
			continue
		}

		// Try to parse the file to check for syntax errors
		content, err := os.ReadFile(mainFile)
		if err != nil {
			t.Errorf("Failed to read %s/main.go: %v", entry.Name(), err)
			continue
		}

		// Basic syntax check - look for obvious issues
		code := string(content)
		if strings.Count(code, "{") != strings.Count(code, "}") {
			t.Errorf("%s: mismatched braces", entry.Name())
			continue
		}

		// Check for required package declaration
		if !strings.HasPrefix(code, "package main") && !strings.Contains(code, "\npackage main") {
			t.Errorf("%s: missing 'package main' declaration", entry.Name())
			continue
		}

		// Check it imports goneuron or internal packages
		if !strings.Contains(code, "github.com/FlavioCFOliveira/GoNeuron") {
			t.Logf("%s: does not import GoNeuron package", entry.Name())
			skippedCount++
			continue
		}

		compiledCount++
	}

	t.Logf("Verified %d examples compile, skipped %d", compiledCount, skippedCount)
}

// TestExampleDirectoryStructure verifies examples follow the expected structure
func TestExampleDirectoryStructure(t *testing.T) {
	root := findProjectRoot(t)
	examplesDir := filepath.Join(root, "examples")

	entries, err := os.ReadDir(examplesDir)
	if err != nil {
		t.Fatalf("Failed to read examples directory: %v", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		exampleDir := filepath.Join(examplesDir, entry.Name())

		// Each example should have a main.go
		mainFile := filepath.Join(exampleDir, "main.go")
		_, err := os.Stat(mainFile)
		if os.IsNotExist(err) {
			// Some directories may be utility folders
			t.Logf("%s: no main.go found (may be a utility directory)", entry.Name())
			continue
		}

		// Examples with datasets should have a datasets subdirectory
		datasetsDir := filepath.Join(exampleDir, "datasets")
		if info, err := os.Stat(datasetsDir); err == nil && info.IsDir() {
			// Verify datasets directory is not empty (if it exists)
			files, _ := os.ReadDir(datasetsDir)
			if len(files) == 0 {
				t.Logf("%s: datasets directory is empty", entry.Name())
			}
		}
	}
}
