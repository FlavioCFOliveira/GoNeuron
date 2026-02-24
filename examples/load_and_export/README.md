# Model Restoration and GGUF Conversion

This tutorial illustrates the process of restoring a pre-trained model from the library's native `.gob` format and converting it to the industry-standard **GGUF v3** format. This workflow is essential for interoperability between Go-based training environments and C++/Python-based inference engines.

## Technical Objectives

1.  **Native Deserialization**: Restoring model topology and parameter states using the `net.Load` protocol.
2.  **Format Interoperability**: Bridging the gap between GoNeuron's internal state and the GGUF ecosystem.
3.  **Metadata Preservation**: Ensuring that architecture identifiers and layer counts are correctly mapped during conversion.

## Key Implementation Details

### 1. Dynamic Restoration
```go
model, lossFn, err := net.Load("checkpoint_modelo.gob")
```
*   **Mechanism**: The `net.Load` function utilizes Go's `encoding/gob` registry to dynamically reconstruct layer types (e.g., `Dense`, `Conv2D`) and re-bind them to contiguous memory buffers. This ensures that the restored object is fully functional for further training or inference.

### 2. Cross-Format Conversion
```go
err := model.SaveGGUF("modelo_final_interoperavel.gguf")
```
*   **Mechanism**: Once the `Network` object is resident in memory, the GGUF writer traverses the layer hierarchy, maps internal parameter names (e.g., `weights`, `biases`) to standard GGUF keys (e.g., `blk.N.weight`, `blk.N.bias`), and writes the aligned binary data.

## Execution

To execute the conversion workflow:

```bash
go run main.go
```

The tutorial will generate a comparative report showing the efficiency gains and structural differences between the native Go binary and the GGUF representation.
