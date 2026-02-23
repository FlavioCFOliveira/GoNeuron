package net

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"time"
)

// CSVLogger logs training progress to a CSV file.
type CSVLogger struct {
	BaseCallback
	Filename string
	Append   bool

	file   *os.File
	writer *csv.Writer
	start  time.Time
}

// NewCSVLogger creates a new CSVLogger.
func NewCSVLogger(filename string, append bool) *CSVLogger {
	return &CSVLogger{
		Filename: filename,
		Append:   append,
	}
}

func (c *CSVLogger) OnTrainBegin(n *Network) {
	mode := os.O_CREATE | os.O_WRONLY
	if c.Append {
		mode |= os.O_APPEND
	} else {
		mode |= os.O_TRUNC
	}

	file, err := os.OpenFile(c.Filename, mode, 0644)
	if err != nil {
		fmt.Printf("CSVLogger: failed to open file %s: %v\n", c.Filename, err)
		return
	}
	c.file = file
	c.writer = csv.NewWriter(file)
	c.start = time.Now()

	// Write header if not appending or if file is empty
	info, err := file.Stat()
	if err == nil && (info.Size() == 0 || !c.Append) {
		c.writer.Write([]string{"epoch", "loss", "time_seconds"})
		c.writer.Flush()
	}
}

func (c *CSVLogger) OnEpochEnd(epoch int, loss float32, n *Network) {
	if c.writer == nil {
		return
	}

	elapsed := time.Since(c.start).Seconds()
	record := []string{
		strconv.Itoa(epoch),
		fmt.Sprintf("%.6f", loss),
		fmt.Sprintf("%.2f", elapsed),
	}

	if err := c.writer.Write(record); err != nil {
		fmt.Printf("CSVLogger: failed to write record: %v\n", err)
	}
	c.writer.Flush()
}

func (c *CSVLogger) OnTrainEnd(n *Network) {
	if c.file != nil {
		c.writer.Flush()
		c.file.Close()
		c.file = nil
		c.writer = nil
	}
}
