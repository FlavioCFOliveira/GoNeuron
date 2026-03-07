package logger

import (
	"bytes"
	"strings"
	"sync"
	"testing"
)

func TestLevelString(t *testing.T) {
	tests := []struct {
		level    Level
		expected string
	}{
		{DebugLevel, "DEBUG"},
		{InfoLevel, "INFO"},
		{WarnLevel, "WARN"},
		{ErrorLevel, "ERROR"},
		{Level(99), "UNKNOWN"},
	}

	for _, tc := range tests {
		got := tc.level.String()
		if got != tc.expected {
			t.Errorf("Level(%d).String() = %q, want %q", tc.level, got, tc.expected)
		}
	}
}

func TestParseLevel(t *testing.T) {
	tests := []struct {
		input    string
		expected Level
	}{
		{"DEBUG", DebugLevel},
		{"INFO", InfoLevel},
		{"WARN", WarnLevel},
		{"ERROR", ErrorLevel},
		{"SILENT", SilentLevel},
		{"UNKNOWN", InfoLevel}, // default
	}

	for _, tc := range tests {
		got := ParseLevel(tc.input)
		if got != tc.expected {
			t.Errorf("ParseLevel(%q) = %d, want %d", tc.input, got, tc.expected)
		}
	}
}

func TestLoggerLevelFiltering(t *testing.T) {
	var buf bytes.Buffer
	l := New(InfoLevel, &buf)

	l.Debug("debug message")
	l.Info("info message")
	l.Warn("warn message")
	l.Error("error message")

	output := buf.String()

	if strings.Contains(output, "debug message") {
		t.Error("Debug message should be filtered out")
	}
	if !strings.Contains(output, "info message") {
		t.Error("Info message should be present")
	}
	if !strings.Contains(output, "warn message") {
		t.Error("Warn message should be present")
	}
	if !strings.Contains(output, "error message") {
		t.Error("Error message should be present")
	}
}

func TestLoggerFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(DebugLevel, &buf)

	l.Info("test message", F("key1", "value1"), F("key2", 42))

	output := buf.String()

	if !strings.Contains(output, "key1=value1") {
		t.Error("Field key1 should be present")
	}
	if !strings.Contains(output, "key2=42") {
		t.Error("Field key2 should be present")
	}
}

func TestFormattedLogging(t *testing.T) {
	var buf bytes.Buffer
	l := New(DebugLevel, &buf)

	l.Infof("formatted %s %d", "string", 42)

	output := buf.String()

	if !strings.Contains(output, "formatted string 42") {
		t.Errorf("Formatted message incorrect: %s", output)
	}
}

func TestSilentLevel(t *testing.T) {
	var buf bytes.Buffer
	l := New(SilentLevel, &buf)

	l.Error("error message")

	if buf.Len() > 0 {
		t.Error("Silent level should not output anything")
	}
}

func TestDefaultLogger(t *testing.T) {
	// Reset default logger for testing
	defaultLogger = nil
	once = sync.Once{}

	l := Default()
	if l == nil {
		t.Error("Default() should return a non-nil logger")
	}

	if l.Level() != InfoLevel {
		t.Errorf("Default level should be InfoLevel, got %v", l.Level())
	}
}

func TestSetLevel(t *testing.T) {
	var buf bytes.Buffer
	l := New(InfoLevel, &buf)

	l.SetLevel(DebugLevel)
	l.Debug("debug after level change")

	output := buf.String()
	if !strings.Contains(output, "debug after level change") {
		t.Error("Debug message should appear after level change")
	}
}

func TestPackageLevelFunctions(t *testing.T) {
	var buf bytes.Buffer
	SetOutput(&buf)
	SetLevel(DebugLevel)

	Debug("debug")
	Info("info")
	Warn("warn")
	Error("error")

	output := buf.String()

	if !strings.Contains(output, "debug") {
		t.Error("Debug should be logged")
	}
	if !strings.Contains(output, "info") {
		t.Error("Info should be logged")
	}
	if !strings.Contains(output, "warn") {
		t.Error("Warn should be logged")
	}
	if !strings.Contains(output, "error") {
		t.Error("Error should be logged")
	}
}
