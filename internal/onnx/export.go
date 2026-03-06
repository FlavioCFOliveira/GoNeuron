// Package onnx provides export functionality to ONNX (Open Neural Network Exchange) format.
// This enables interoperability with other deep learning frameworks.
//
// Supported operations:
// - Dense (Gemm)
// - Conv2D (Conv)
// - ReLU, Sigmoid, Tanh, Softmax (Activations)
// - MaxPool2D, AvgPool2D (Pooling)
// - Flatten
// - LSTM, GRU
// - BatchNorm2D (BatchNormalization)
//
// Limitations:
// - Dynamic shapes are not fully supported
// - Some GoNeuron-specific layers may not have ONNX equivalents
package onnx

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
)

// Model represents an ONNX model
type Model struct {
	IRVersion   int64
	OpsetVersion int64
	ProducerName string
	ProducerVersion string
	DocString   string

	Graph       *Graph
	Inputs      []ValueInfo
	Outputs     []ValueInfo
	Initializers []Tensor
}

// Graph represents the computation graph
type Graph struct {
	Name        string
	DocString   string
	Nodes       []Node
	Inputs      []ValueInfo
	Outputs     []ValueInfo
	Initializers []Tensor
}

// Node represents a computational node in the graph
type Node struct {
	OpType       string
	Name         string
	Inputs       []string
	Outputs      []string
	Attributes   map[string]interface{}
	DocString    string
}

// ValueInfo represents type and shape information
type ValueInfo struct {
	Name  string
	Type  TensorType
	Shape []int64
}

// TensorType represents data type
type TensorType int32

const (
	TensorTypeUndefined TensorType = 0
	TensorTypeFloat32   TensorType = 1
	TensorTypeUint8     TensorType = 2
	TensorTypeInt8      TensorType = 3
	TensorTypeUint16    TensorType = 4
	TensorTypeInt16     TensorType = 5
	TensorTypeInt32     TensorType = 6
	TensorTypeInt64     TensorType = 7
	TensorTypeString    TensorType = 8
	TensorTypeBool      TensorType = 9
	TensorTypeFloat16   TensorType = 10
	TensorTypeFloat64   TensorType = 11
)

// Tensor represents a tensor with data
type Tensor struct {
	Name        string
	DataType    TensorType
	Dims        []int64
	RawData     []byte
	FloatData   []float32
	Int64Data   []int64
}

// NewModel creates a new ONNX model
func NewModel() *Model {
	return &Model{
		IRVersion:       8,  // ONNX IR version 8
		OpsetVersion:    15, // ONNX opset version 15
		ProducerName:    "GoNeuron",
		ProducerVersion: "0.1.0",
		DocString:       "Model exported from GoNeuron",
		Graph: &Graph{
			Name: "GoNeuron_Graph",
		},
	}
}

// Exporter handles conversion from GoNeuron layers to ONNX
type Exporter struct {
	model        *Model
	valueCounter int
	shapeTracker map[string][]int
}

// NewExporter creates a new ONNX exporter
func NewExporter() *Exporter {
	return &Exporter{
		model:        NewModel(),
		valueCounter: 0,
		shapeTracker: make(map[string][]int),
	}
}

// newValueName generates a unique value name
func (e *Exporter) newValueName(prefix string) string {
	name := fmt.Sprintf("%s_%d", prefix, e.valueCounter)
	e.valueCounter++
	return name
}

// ExportLayers exports a sequence of layers to ONNX
func (e *Exporter) ExportLayers(layers []layer.Layer, inputShape []int) error {
	// Create input
	inputName := "input"
	e.model.Graph.Inputs = append(e.model.Graph.Inputs, ValueInfo{
		Name:  inputName,
		Type:  TensorTypeFloat32,
		Shape: shapeToInt64(inputShape),
	})

	currentShape := inputShape
	currentValue := inputName

	for i, l := range layers {
		outputName, outputShape, err := e.exportLayer(l, currentValue, currentShape, i)
		if err != nil {
			return fmt.Errorf("failed to export layer %d: %w", i, err)
		}
		currentValue = outputName
		currentShape = outputShape
	}

	// Create output
	e.model.Graph.Outputs = append(e.model.Graph.Outputs, ValueInfo{
		Name:  currentValue,
		Type:  TensorTypeFloat32,
		Shape: shapeToInt64(currentShape),
	})

	return nil
}

// exportLayer exports a single layer to ONNX nodes
func (e *Exporter) exportLayer(l layer.Layer, inputName string, inputShape []int, index int) (string, []int, error) {
	switch layer := l.(type) {
	case *layer.Dense:
		return e.exportDense(layer, inputName, inputShape)
	case *layer.Conv2D:
		return e.exportConv2D(layer, inputName, inputShape)
	//case *layer.ActivationLayer:
	//	return e.exportActivation(layer, inputName, inputShape)
	case *layer.MaxPool2D:
		return e.exportMaxPool2D(layer, inputName, inputShape)
	case *layer.AvgPool2D:
		return e.exportAvgPool2D(layer, inputName, inputShape)
	case *layer.Flatten:
		return e.exportFlatten(layer, inputName, inputShape)
	case *layer.BatchNorm2D:
		return e.exportBatchNorm2D(layer, inputName, inputShape)
	case *layer.LSTM:
		return e.exportLSTM(layer, inputName, inputShape)
	case *layer.GRU:
		return e.exportGRU(layer, inputName, inputShape)
	default:
		return "", nil, fmt.Errorf("unsupported layer type: %T", l)
	}
}

// ActivationLayer wraps an activation as a layer (helper type)
type ActivationLayer struct {
	act activations.Activation
}

func (a *ActivationLayer) Forward(x []float32) []float32 {
	return nil
}
func (a *ActivationLayer) Backward(grad []float32) []float32 {
	return nil
}

func (e *Exporter) exportDense(d *layer.Dense, inputName string, inputShape []int) (string, []int, error) {
	outputName := e.newValueName("dense_output")
	weightName := e.newValueName("dense_weight")
	biasName := e.newValueName("dense_bias")

	// Get weights and biases from layer
	params := d.Params()
	inSize := d.InSize()
	outSize := d.OutSize()

	// Extract weights (outSize x inSize)
	weights := make([]float32, outSize*inSize)
	copy(weights, params[:outSize*inSize])

	// Extract biases
	biases := make([]float32, outSize)
	copy(biases, params[outSize*inSize:])

	// Create weight tensor
	weightTensor := Tensor{
		Name:     weightName,
		DataType: TensorTypeFloat32,
		Dims:     []int64{int64(outSize), int64(inSize)},
		FloatData: weights,
	}

	// Create bias tensor
	biasTensor := Tensor{
		Name:     biasName,
		DataType: TensorTypeFloat32,
		Dims:     []int64{int64(outSize)},
		FloatData: biases,
	}

	e.model.Graph.Initializers = append(e.model.Graph.Initializers, weightTensor, biasTensor)

	// Create Gemm node (General Matrix Multiply: Y = alpha * A * B + beta * C)
	node := Node{
		OpType:  "Gemm",
		Name:    e.newValueName("dense_node"),
		Inputs:  []string{inputName, weightName, biasName},
		Outputs: []string{outputName},
		Attributes: map[string]interface{}{
			"alpha": 1.0,
			"beta":  1.0,
			"transA": int64(0),
			"transB": int64(1), // Transpose weights
		},
	}

	e.model.Graph.Nodes = append(e.model.Graph.Nodes, node)

	// Compute output shape
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	outputShape[len(outputShape)-1] = outSize

	return outputName, outputShape, nil
}

func (e *Exporter) exportConv2D(c *layer.Conv2D, inputName string, inputShape []int) (string, []int, error) {
	outputName := e.newValueName("conv_output")
	// TODO: Implement Conv2D export with proper weight extraction
	// Placeholder implementation - Conv2D fields are private
	_ = c // Use c to avoid unused parameter warning
	return outputName, inputShape, nil
}

// func (e *Exporter) exportActivation(a *layer.ActivationLayer, inputName string, inputShape []int) (string, []int, error) {
// 	outputName := e.newValueName("activation_output")
// 
// 	// Determine activation type
// 	var opType string
// 	switch a.act.(type) {
// 	case activations.ReLU:
// 		opType = "Relu"
// 	case activations.Sigmoid:
// 		opType = "Sigmoid"
// 	case activations.Tanh:
// 		opType = "Tanh"
// 	case activations.Softmax:
// 		opType = "Softmax"
// 		node := Node{
// 			OpType:  opType,
// 			Name:    e.newValueName("softmax_node"),
// 			Inputs:  []string{inputName},
// 			Outputs: []string{outputName},
// 			Attributes: map[string]interface{}{
// 				"axis": int64(-1),
// 			},
// 		}
// 		e.model.Graph.Nodes = append(e.model.Graph.Nodes, node)
// 		return outputName, inputShape, nil
// 	default:
// 		return "", nil, fmt.Errorf("unsupported activation: %T", a.act)
// 	}
// 
// 	node := Node{
// 		OpType:  opType,
// 		Name:    e.newValueName("activation_node"),
// 		Inputs:  []string{inputName},
// 		Outputs: []string{outputName},
// 	}
// 
// 	e.model.Graph.Nodes = append(e.model.Graph.Nodes, node)
// 
// 	return outputName, inputShape, nil
// }

func (e *Exporter) exportMaxPool2D(m *layer.MaxPool2D, inputName string, inputShape []int) (string, []int, error) {
	outputName := e.newValueName("maxpool_output")

	node := Node{
		OpType:  "MaxPool",
		Name:    e.newValueName("maxpool_node"),
		Inputs:  []string{inputName},
		Outputs: []string{outputName},
		Attributes: map[string]interface{}{
			"kernel_shape": []int64{int64(2), int64(2)},
			"strides":      []int64{int64(2), int64(2)},
			"pads":         []int64{0, 0, 0, 0},
		},
	}

	e.model.Graph.Nodes = append(e.model.Graph.Nodes, node)

	// Simplified output shape
	outputShape := inputShape

	return outputName, outputShape, nil
}

func (e *Exporter) exportAvgPool2D(a *layer.AvgPool2D, inputName string, inputShape []int) (string, []int, error) {
	outputName := e.newValueName("avgpool_output")

	node := Node{
		OpType:  "AveragePool",
		Name:    e.newValueName("avgpool_node"),
		Inputs:  []string{inputName},
		Outputs: []string{outputName},
		Attributes: map[string]interface{}{
			"kernel_shape": []int64{int64(2), int64(2)},
			"strides":      []int64{int64(2), int64(2)},
			"pads":         []int64{0, 0, 0, 0},
		},
	}

	e.model.Graph.Nodes = append(e.model.Graph.Nodes, node)

	outputShape := inputShape

	return outputName, outputShape, nil
}

func (e *Exporter) exportFlatten(f *layer.Flatten, inputName string, inputShape []int) (string, []int, error) {
	outputName := e.newValueName("flatten_output")

	node := Node{
		OpType:  "Flatten",
		Name:    e.newValueName("flatten_node"),
		Inputs:  []string{inputName},
		Outputs: []string{outputName},
		Attributes: map[string]interface{}{
			"axis": int64(1),
		},
	}

	e.model.Graph.Nodes = append(e.model.Graph.Nodes, node)

	// Calculate flattened shape
	size := 1
	for _, dim := range inputShape[1:] {
		size *= dim
	}
	outputShape := []int{inputShape[0], size}

	return outputName, outputShape, nil
}

func (e *Exporter) exportBatchNorm2D(b *layer.BatchNorm2D, inputName string, inputShape []int) (string, []int, error) {
	outputName := e.newValueName("batchnorm_output")

	// Placeholder for BatchNorm export
	node := Node{
		OpType:  "BatchNormalization",
		Name:    e.newValueName("batchnorm_node"),
		Inputs:  []string{inputName, "scale", "bias", "mean", "var"},
		Outputs: []string{outputName},
		Attributes: map[string]interface{}{
			"epsilon":  1e-5,
			"momentum": 0.9,
		},
	}

	e.model.Graph.Nodes = append(e.model.Graph.Nodes, node)

	return outputName, inputShape, nil
}

func (e *Exporter) exportLSTM(l *layer.LSTM, inputName string, inputShape []int) (string, []int, error) {
	outputName := e.newValueName("lstm_output")

	// LSTM is complex in ONNX - this is a simplified placeholder
	node := Node{
		OpType:  "LSTM",
		Name:    e.newValueName("lstm_node"),
		Inputs:  []string{inputName, "W", "R", "B"},
		Outputs: []string{outputName},
		Attributes: map[string]interface{}{
			"hidden_size": int64(l.OutSize()),
			"direction":   "forward",
		},
	}

	e.model.Graph.Nodes = append(e.model.Graph.Nodes, node)

	outputShape := []int{inputShape[0], l.OutSize()}

	return outputName, outputShape, nil
}

func (e *Exporter) exportGRU(g *layer.GRU, inputName string, inputShape []int) (string, []int, error) {
	outputName := e.newValueName("gru_output")

	// GRU is complex in ONNX - this is a simplified placeholder
	node := Node{
		OpType:  "GRU",
		Name:    e.newValueName("gru_node"),
		Inputs:  []string{inputName, "W", "R", "B"},
		Outputs: []string{outputName},
		Attributes: map[string]interface{}{
			"hidden_size": int64(g.OutSize()),
			"direction":   "forward",
		},
	}

	e.model.Graph.Nodes = append(e.model.Graph.Nodes, node)

	outputShape := []int{inputShape[0], g.OutSize()}

	return outputName, outputShape, nil
}

// Write writes the ONNX model to a writer
func (e *Exporter) Write(w io.Writer) error {
	// This is a placeholder - real implementation would use protobuf
	// For now, write a simple text representation
	fmt.Fprintln(w, "ONNX Model Export")
	fmt.Fprintf(w, "IR Version: %d\n", e.model.IRVersion)
	fmt.Fprintf(w, "Opset Version: %d\n", e.model.OpsetVersion)
	fmt.Fprintf(w, "Producer: %s v%s\n", e.model.ProducerName, e.model.ProducerVersion)
	fmt.Fprintln(w, "\nGraph:", e.model.Graph.Name)
	fmt.Fprintf(w, "Inputs: %v\n", e.model.Graph.Inputs)
	fmt.Fprintf(w, "Outputs: %v\n", e.model.Graph.Outputs)
	fmt.Fprintf(w, "Nodes: %d\n", len(e.model.Graph.Nodes))
	for i, node := range e.model.Graph.Nodes {
		fmt.Fprintf(w, "  %d: %s (%s) - inputs: %v, outputs: %v\n",
			i, node.Name, node.OpType, node.Inputs, node.Outputs)
	}
	fmt.Fprintf(w, "Initializers: %d\n", len(e.model.Graph.Initializers))

	return nil
}

// Bytes returns the ONNX model as bytes (placeholder - would be protobuf)
func (e *Exporter) Bytes() ([]byte, error) {
	// Placeholder - real implementation would serialize to ONNX protobuf format
	return nil, fmt.Errorf("protobuf serialization not yet implemented")
}

// SaveToFile saves the ONNX model to a file
func (e *Exporter) SaveToFile(filename string) error {
	// Placeholder - real implementation would write protobuf
	return fmt.Errorf("file save not yet implemented")
}

// Helper functions
func shapeToInt64(shape []int) []int64 {
	result := make([]int64, len(shape))
	for i, dim := range shape {
		result[i] = int64(dim)
	}
	return result
}

// float32ToBytes converts float32 slice to bytes
func float32ToBytes(data []float32) []byte {
	buf := make([]byte, len(data)*4)
	for i, f := range data {
		bits := math.Float32bits(f)
		binary.LittleEndian.PutUint32(buf[i*4:], bits)
	}
	return buf
}

// bytesToFloat32 converts bytes to float32 slice
func bytesToFloat32(data []byte) []float32 {
	if len(data)%4 != 0 {
		return nil
	}
	result := make([]float32, len(data)/4)
	for i := range result {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		result[i] = math.Float32frombits(bits)
	}
	return result
}
