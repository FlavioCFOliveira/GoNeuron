# CSV Training Pipeline

A practical example of building an end-to-end machine learning pipeline that consumes data directly from **CSV files**. It demonstrates robust data loading, parsing, and normalization for tabular datasets.

## Technical Objectives

1.  **Tabular Data Handling**: Parsing multi-column CSV files into model-ready tensors.
2.  **Feature Normalization**: Implementing Z-score or Min-Max scaling on CSV columns to facilitate training.
3.  **Pipeline Automation**: From raw file input to trained model evaluation.

## Key Implementation Details

### 1. CSV Parsing Utility
The example highlights the use of Go's `encoding/csv` combined with type conversion and error handling to create a resilient data loader that handles header rows and varying numeric formats.

### 2. Batching from Disk
Demonstrates how to efficiently buffer CSV data to feed the model's `Fit` method without exhausting system memory on large datasets.

## Execution

```bash
go run main.go
```
