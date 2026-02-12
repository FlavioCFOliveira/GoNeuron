# Neural Network Research

Experimental neural network implementations in Go.

## Project Structure

- `internal/net/` - Core neural network types and operations
- `internal/layer/` - Layer implementations (dense, convolutional, etc.)
- `internal/activations/` - Activation functions
- `internal/loss/` - Loss/cost functions
- `internal/opt/` - Optimization algorithms
- `cmd/` - Example applications and tests

## Getting Started

```bash
go mod download
go test ./...
```

## Dependencies

- `gonum.org/v1/gonum` - Numerical linear algebra operations
