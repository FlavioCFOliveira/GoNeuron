// Package config provides configuration structures for the GoNeuron CLI
package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// TrainingConfig represents a complete training configuration
type TrainingConfig struct {
	// Model architecture
	Model ModelConfig `yaml:"model"`

	// Training parameters
	Training TrainingParams `yaml:"training"`

	// Dataset configuration
	Dataset DatasetConfig `yaml:"dataset"`

	// Output configuration
	Output OutputConfig `yaml:"output"`

	// Hardware configuration
	Hardware HardwareConfig `yaml:"hardware,omitempty"`
}

// ModelConfig defines the model architecture
type ModelConfig struct {
	Name   string        `yaml:"name"`
	Layers []LayerConfig `yaml:"layers"`
}

// LayerConfig defines a single layer
type LayerConfig struct {
	Type       string                 `yaml:"type"`
	Params     map[string]interface{} `yaml:"params"`
	Activation string                 `yaml:"activation,omitempty"`
}

// TrainingParams defines training hyperparameters
type TrainingParams struct {
	Epochs       int     `yaml:"epochs"`
	BatchSize    int     `yaml:"batch_size"`
	LearningRate float32 `yaml:"learning_rate"`
	Optimizer    string  `yaml:"optimizer"`
	Loss         string  `yaml:"loss"`
	ValidationSplit float32 `yaml:"validation_split,omitempty"`
	Shuffle      bool    `yaml:"shuffle,omitempty"`
	Seed         int     `yaml:"seed,omitempty"`
}

// DatasetConfig defines dataset parameters
type DatasetConfig struct {
	Type       string            `yaml:"type"` // csv, mnist, cifar10, etc.
	Path       string            `yaml:"path"`
	Preprocess map[string]string `yaml:"preprocess,omitempty"`
	Augment    AugmentConfig     `yaml:"augment,omitempty"`
}

// AugmentConfig defines data augmentation parameters
type AugmentConfig struct {
	Enabled   bool    `yaml:"enabled"`
	Rotation  float32 `yaml:"rotation,omitempty"`
	FlipH     bool    `yaml:"flip_horizontal,omitempty"`
	FlipV     bool    `yaml:"flip_vertical,omitempty"`
	Zoom      float32 `yaml:"zoom,omitempty"`
	Shift     float32 `yaml:"shift,omitempty"`
}

// OutputConfig defines output settings
type OutputConfig struct {
	ModelPath    string `yaml:"model_path"`
	CheckpointDir string `yaml:"checkpoint_dir,omitempty"`
	SaveEvery    int    `yaml:"save_every_epochs,omitempty"`
	LogDir       string `yaml:"log_dir,omitempty"`
	ExportONNX   bool   `yaml:"export_onnx,omitempty"`
}

// HardwareConfig defines hardware settings
type HardwareConfig struct {
	Device    string `yaml:"device,omitempty"` // cpu, gpu, auto
	Threads   int    `yaml:"threads,omitempty"`
	MixedPrecision bool `yaml:"mixed_precision,omitempty"`
}

// LoadConfig loads a configuration from a YAML file
func LoadConfig(path string) (*TrainingConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config TrainingConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Set defaults
	config.setDefaults()

	// Validate
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return &config, nil
}

// setDefaults sets default values for optional fields
func (c *TrainingConfig) setDefaults() {
	if c.Training.Epochs == 0 {
		c.Training.Epochs = 10
	}
	if c.Training.BatchSize == 0 {
		c.Training.BatchSize = 32
	}
	if c.Training.LearningRate == 0 {
		c.Training.LearningRate = 0.001
	}
	if c.Training.Optimizer == "" {
		c.Training.Optimizer = "adam"
	}
	if c.Training.Loss == "" {
		c.Training.Loss = "crossentropy"
	}
	if c.Training.ValidationSplit == 0 {
		c.Training.ValidationSplit = 0.2
	}
	if c.Hardware.Device == "" {
		c.Hardware.Device = "auto"
	}
}

// Validate checks if the configuration is valid
func (c *TrainingConfig) Validate() error {
	if c.Model.Name == "" {
		return fmt.Errorf("model name is required")
	}
	if len(c.Model.Layers) == 0 {
		return fmt.Errorf("at least one layer is required")
	}
	if c.Dataset.Type == "" {
		return fmt.Errorf("dataset type is required")
	}
	if c.Output.ModelPath == "" {
		return fmt.Errorf("output model path is required")
	}
	return nil
}

// Save saves the configuration to a YAML file
func (c *TrainingConfig) Save(path string) error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}
