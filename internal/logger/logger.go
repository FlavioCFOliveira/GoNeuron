// Package logger provides structured logging for the GoNeuron library.
// It supports multiple log levels and can be configured to output to
// various destinations (stdout, file, etc.).
package logger

import (
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// Level represents the severity of a log message.
type Level int

const (
	// DebugLevel is the most verbose level, useful for development.
	DebugLevel Level = iota
	// InfoLevel is for general information about operation.
	InfoLevel
	// WarnLevel is for warning messages that don't prevent operation.
	WarnLevel
	// ErrorLevel is for error messages that indicate a problem.
	ErrorLevel
	// SilentLevel disables all logging.
	SilentLevel
)

// String returns the string representation of a log level.
func (l Level) String() string {
	switch l {
	case DebugLevel:
		return "DEBUG"
	case InfoLevel:
		return "INFO"
	case WarnLevel:
		return "WARN"
	case ErrorLevel:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// ParseLevel converts a string to a Level.
func ParseLevel(s string) Level {
	switch s {
	case "DEBUG":
		return DebugLevel
	case "INFO":
		return InfoLevel
	case "WARN":
		return WarnLevel
	case "ERROR":
		return ErrorLevel
	case "SILENT":
		return SilentLevel
	default:
		return InfoLevel
	}
}

// Field represents a key-value pair for structured logging.
type Field struct {
	Key   string
	Value interface{}
}

// F creates a new Field with the given key and value.
func F(key string, value interface{}) Field {
	return Field{Key: key, Value: value}
}

// Logger is a structured logger with configurable output and level.
type Logger struct {
	level  Level
	output io.Writer
	mu     sync.Mutex
	// Formatter is the function used to format log messages
	Formatter func(time time.Time, level Level, msg string, fields []Field) string
}

var (
	// Default logger instance
	defaultLogger *Logger
	once          sync.Once
)

// Default returns the default logger instance.
func Default() *Logger {
	once.Do(func() {
		defaultLogger = New(InfoLevel, os.Stdout)
	})
	return defaultLogger
}

// SetDefault sets the default logger instance.
func SetDefault(l *Logger) {
	defaultLogger = l
}

// New creates a new Logger with the specified level and output.
func New(level Level, output io.Writer) *Logger {
	return &Logger{
		level:     level,
		output:    output,
		Formatter: DefaultFormatter,
	}
}

// DefaultFormatter is the default log message formatter.
func DefaultFormatter(t time.Time, level Level, msg string, fields []Field) string {
	// Build field string
	fieldStr := ""
	for _, f := range fields {
		fieldStr += fmt.Sprintf(" %s=%v", f.Key, f.Value)
	}

	return fmt.Sprintf("[%s] %s: %s%s\n",
		t.Format("2006-01-02 15:04:05"),
		level.String(),
		msg,
		fieldStr,
	)
}

// SetLevel sets the minimum log level.
func (l *Logger) SetLevel(level Level) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.level = level
}

// Level returns the current log level.
func (l *Logger) Level() Level {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.level
}

// SetOutput sets the output writer.
func (l *Logger) SetOutput(w io.Writer) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.output = w
}

// log writes a log message at the specified level.
func (l *Logger) log(level Level, msg string, fields []Field) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if level < l.level || l.level == SilentLevel {
		return
	}

	formatted := l.Formatter(time.Now(), level, msg, fields)
	fmt.Fprint(l.output, formatted)
}

// Debug logs a debug message.
func (l *Logger) Debug(msg string, fields ...Field) {
	l.log(DebugLevel, msg, fields)
}

// Info logs an info message.
func (l *Logger) Info(msg string, fields ...Field) {
	l.log(InfoLevel, msg, fields)
}

// Warn logs a warning message.
func (l *Logger) Warn(msg string, fields ...Field) {
	l.log(WarnLevel, msg, fields)
}

// Error logs an error message.
func (l *Logger) Error(msg string, fields ...Field) {
	l.log(ErrorLevel, msg, fields)
}

// Debugf logs a formatted debug message.
func (l *Logger) Debugf(format string, args ...interface{}) {
	l.log(DebugLevel, fmt.Sprintf(format, args...), nil)
}

// Infof logs a formatted info message.
func (l *Logger) Infof(format string, args ...interface{}) {
	l.log(InfoLevel, fmt.Sprintf(format, args...), nil)
}

// Warnf logs a formatted warning message.
func (l *Logger) Warnf(format string, args ...interface{}) {
	l.log(WarnLevel, fmt.Sprintf(format, args...), nil)
}

// Errorf logs a formatted error message.
func (l *Logger) Errorf(format string, args ...interface{}) {
	l.log(ErrorLevel, fmt.Sprintf(format, args...), nil)
}

// WithFields returns a new Logger with the given fields added to all messages.
func (l *Logger) WithFields(fields ...Field) *Logger {
	return &Logger{
		level:     l.level,
		output:    l.output,
		Formatter: l.Formatter,
	}
}

// Package-level convenience functions using the default logger.

// Debug logs a debug message using the default logger.
func Debug(msg string, fields ...Field) {
	Default().Debug(msg, fields...)
}

// Info logs an info message using the default logger.
func Info(msg string, fields ...Field) {
	Default().Info(msg, fields...)
}

// Warn logs a warning message using the default logger.
func Warn(msg string, fields ...Field) {
	Default().Warn(msg, fields...)
}

// Error logs an error message using the default logger.
func Error(msg string, fields ...Field) {
	Default().Error(msg, fields...)
}

// Debugf logs a formatted debug message using the default logger.
func Debugf(format string, args ...interface{}) {
	Default().Debugf(format, args...)
}

// Infof logs a formatted info message using the default logger.
func Infof(format string, args ...interface{}) {
	Default().Infof(format, args...)
}

// Warnf logs a formatted warning message using the default logger.
func Warnf(format string, args ...interface{}) {
	Default().Warnf(format, args...)
}

// Errorf logs a formatted error message using the default logger.
func Errorf(format string, args ...interface{}) {
	Default().Errorf(format, args...)
}

// SetLevel sets the log level on the default logger.
func SetLevel(level Level) {
	Default().SetLevel(level)
}

// GetLevel returns the log level of the default logger.
func GetLevel() Level {
	return Default().Level()
}

// SetOutput sets the output writer on the default logger.
func SetOutput(w io.Writer) {
	Default().SetOutput(w)
}
