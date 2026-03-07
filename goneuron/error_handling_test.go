// Package goneuron provides tests for error handling in public APIs
package goneuron

import (
	"strings"
	"testing"
)

// TestDenseE_ValidArgs tests DenseE with valid arguments
func TestDenseE_ValidArgs(t *testing.T) {
	tests := []struct {
		name string
		args []interface{}
	}{
		{"2 args lazy", []interface{}{10, ReLU}},
		{"3 args in-out-act", []interface{}{64, 10, ReLU}},
		{"3 args out-act-in", []interface{}{10, ReLU, 64}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, err := DenseE(tt.args...)
			if err != nil {
				t.Errorf("DenseE() returned unexpected error: %v", err)
			}
			if layer == nil {
				t.Error("DenseE() returned nil layer")
			}
		})
	}
}

// TestDenseE_InvalidArgs tests DenseE with invalid arguments
func TestDenseE_InvalidArgs(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		wantErr string
	}{
		{"1 arg", []interface{}{10}, "expects 2 or 3 arguments"},
		{"4 args", []interface{}{64, 10, ReLU, 64}, "expects 2 or 3 arguments"},
		{"wrong type first", []interface{}{"invalid", ReLU}, "must be int"},
		{"wrong type second", []interface{}{10, "invalid"}, "must be Activation"},
		{"wrong type third", []interface{}{10, ReLU, "invalid"}, "must be int"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, err := DenseE(tt.args...)
			if err == nil {
				t.Error("DenseE() expected error, got nil")
			}
			if layer != nil {
				t.Error("DenseE() expected nil layer on error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("DenseE() error = %v, want containing %v", err, tt.wantErr)
			}
		})
	}
}

// TestConv2DE_ValidArgs tests Conv2DE with valid arguments
func TestConv2DE_ValidArgs(t *testing.T) {
	tests := []struct {
		name string
		args []interface{}
	}{
		{"5 args lazy", []interface{}{32, 3, 1, 1, ReLU}},
		{"6 args in-out", []interface{}{3, 32, 3, 1, 1, ReLU}},
		{"6 args out-in", []interface{}{32, 3, 1, 1, ReLU, 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, err := Conv2DE(tt.args...)
			if err != nil {
				t.Errorf("Conv2DE() returned unexpected error: %v", err)
			}
			if layer == nil {
				t.Error("Conv2DE() returned nil layer")
			}
		})
	}
}

// TestConv2DE_InvalidArgs tests Conv2DE with invalid arguments
func TestConv2DE_InvalidArgs(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		wantErr string
	}{
		{"4 args", []interface{}{32, 3, 1, 1}, "expects 5 or 6 arguments"},
		{"7 args", []interface{}{3, 32, 3, 1, 1, ReLU, 3}, "expects 5 or 6 arguments"},
		{"wrong type activation", []interface{}{32, 3, 1, 1, "invalid"}, "must be Activation"},
		{"wrong type int", []interface{}{"invalid", 3, 1, 1, ReLU}, "must be int"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, err := Conv2DE(tt.args...)
			if err == nil {
				t.Error("Conv2DE() expected error, got nil")
			}
			if layer != nil {
				t.Error("Conv2DE() expected nil layer on error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("Conv2DE() error = %v, want containing %v", err, tt.wantErr)
			}
		})
	}
}

// TestTransformerBlockE_ValidArgs tests TransformerBlockE with valid arguments
func TestTransformerBlockE_ValidArgs(t *testing.T) {
	tests := []struct {
		name string
		args []interface{}
	}{
		{"4 args lazy", []interface{}{8, 10, 64, true}},
		{"5 args explicit", []interface{}{128, 8, 10, 64, false}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, err := TransformerBlockE(tt.args...)
			if err != nil {
				t.Errorf("TransformerBlockE() returned unexpected error: %v", err)
			}
			if layer == nil {
				t.Error("TransformerBlockE() returned nil layer")
			}
		})
	}
}

// TestTransformerBlockE_InvalidArgs tests TransformerBlockE with invalid arguments
func TestTransformerBlockE_InvalidArgs(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		wantErr string
	}{
		{"3 args", []interface{}{8, 10, 64}, "expects 4 or 5 arguments"},
		{"6 args", []interface{}{128, 8, 10, 64, true, "extra"}, "expects 4 or 5 arguments"},
		{"wrong type int", []interface{}{"invalid", 8, 10, 64}, "must be int"},
		{"wrong type bool", []interface{}{8, 10, 64, "invalid"}, "must be bool"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, err := TransformerBlockE(tt.args...)
			if err == nil {
				t.Error("TransformerBlockE() expected error, got nil")
			}
			if layer != nil {
				t.Error("TransformerBlockE() expected nil layer on error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("TransformerBlockE() error = %v, want containing %v", err, tt.wantErr)
			}
		})
	}
}

// TestDense_BackwardCompatibility verifies that the original Dense function still works
func TestDense_BackwardCompatibility(t *testing.T) {
	// This should not panic with valid args
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Dense() panicked with valid args: %v", r)
		}
	}()

	_ = Dense(10, ReLU)
}

// TestDense_ReturnsNilOnInvalid verifies that Dense returns nil on invalid args
func TestDense_ReturnsNilOnInvalid(t *testing.T) {
	l := Dense(10) // 1 arg - should return nil
	if l != nil {
		t.Error("Dense() should return nil with invalid args")
	}
}

// TestConv2D_BackwardCompatibility verifies that the original Conv2D function still works
func TestConv2D_BackwardCompatibility(t *testing.T) {
	l := Conv2D(32, 3, 1, 1, ReLU)
	if l == nil {
		t.Error("Conv2D() returned nil with valid args")
	}
}

// TestTransformerBlock_BackwardCompatibility verifies that the original TransformerBlock function still works
func TestTransformerBlock_BackwardCompatibility(t *testing.T) {
	l := TransformerBlock(8, 10, 64, true)
	if l == nil {
		t.Error("TransformerBlock() returned nil with valid args")
	}
}
