package goneuron

import (
	"github.com/FlavioCFOliveira/GoNeuron/internal/activations"
	"github.com/FlavioCFOliveira/GoNeuron/internal/layer"
	"github.com/FlavioCFOliveira/GoNeuron/internal/loss"
	"github.com/FlavioCFOliveira/GoNeuron/internal/net"
	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// Re-export common types and functions for easier access
type (
	Model     = net.Sequential
	Layer     = layer.Layer
	Optimizer = opt.Optimizer
	Loss      = loss.Loss
	Dataset   = net.Dataset
)

// Model creation
func NewSequential(layers ...Layer) *Model {
	return net.NewSequential(layers...)
}

// Activations
var (
	ReLU       = activations.ReLU{}
	Sigmoid    = activations.Sigmoid{}
	Tanh       = activations.Tanh{}
	Softmax    = activations.Softmax{}
	LogSoftmax = activations.LogSoftmax{}
	Linear     = activations.Linear{}
	GELU       = activations.GELU{}
	SiLU       = activations.SiLU{}
	SELU       = activations.SELU{}
)

func LeakyReLU(alpha float32) activations.Activation {
	return activations.NewLeakyReLU(alpha)
}

func ELU(alpha float32) activations.Activation {
	return activations.NewELU(alpha)
}

// Layers

// Dense creates a fully connected layer.
// Supports:
// - Dense(out, act) -> Lazy inference of input size
// - Dense(in, out, act) -> Explicit input size
// - Dense(out, act, in) -> Explicit input size (alternative order used in some examples)
func Dense(args ...interface{}) Layer {
	if len(args) == 2 {
		out := args[0].(int)
		act := args[1].(activations.Activation)
		return layer.NewDense(-1, out, act)
	}
	if len(args) == 3 {
		// Check if 2nd arg is int (in, out, act) or activation (out, act, in)
		if out, ok := args[1].(int); ok {
			in := args[0].(int)
			act := args[2].(activations.Activation)
			return layer.NewDense(in, out, act)
		} else {
			out := args[0].(int)
			act := args[1].(activations.Activation)
			in := args[2].(int)
			return layer.NewDense(in, out, act)
		}
	}
	panic("Dense expects (out, act) or (in, out, act)")
}

// Conv2D creates a 2D convolutional layer.
// Supports:
// - Conv2D(outC, k, s, p, act) -> Lazy inference of inChannels
// - Conv2D(inC, outC, k, s, p, act) -> Explicit inChannels
// - Conv2D(outC, k, s, p, act, inC) -> Explicit inChannels (alternative order)
func Conv2D(args ...interface{}) Layer {
	if len(args) == 5 {
		outC := args[0].(int)
		k := args[1].(int)
		s := args[2].(int)
		p := args[3].(int)
		act := args[4].(activations.Activation)
		return layer.NewConv2D(-1, outC, k, s, p, act)
	}
	if len(args) == 6 {
		// Check if 6th arg is activation (inC, outC, k, s, p, act) or int (outC, k, s, p, act, inC)
		if act, ok := args[5].(activations.Activation); ok {
			inC := args[0].(int)
			outC := args[1].(int)
			k := args[2].(int)
			s := args[3].(int)
			p := args[4].(int)
			return layer.NewConv2D(inC, outC, k, s, p, act)
		} else {
			outC := args[0].(int)
			k := args[1].(int)
			s := args[2].(int)
			p := args[3].(int)
			act := args[4].(activations.Activation)
			inC := args[5].(int)
			return layer.NewConv2D(inC, outC, k, s, p, act)
		}
	}
	panic("Conv2D expects (outC, k, s, p, act) or (inC, outC, k, s, p, act)")
}

func LSTM(out int, in ...int) Layer {
	inSize := -1
	if len(in) > 0 {
		inSize = in[0]
	}
	return layer.NewLSTM(inSize, out)
}

func GRU(out int, in ...int) Layer {
	inSize := -1
	if len(in) > 0 {
		inSize = in[0]
	}
	return layer.NewGRU(inSize, out)
}

func Dropout(prob float32, in ...int) Layer {
	inSize := -1
	if len(in) > 0 {
		inSize = in[0]
	}
	return layer.NewDropout(prob, inSize)
}

func BatchNorm2D(inChannels ...int) Layer {
	in := -1
	if len(inChannels) > 0 {
		in = inChannels[0]
	}
	return layer.NewBatchNorm2D(in, 1e-5, 0.1, true)
}

func LayerNorm(normalizedShape ...int) Layer {
	in := -1
	if len(normalizedShape) > 0 {
		in = normalizedShape[0]
	}
	return layer.NewLayerNorm(in, 1e-5, true)
}

func RMSNorm(normalizedShape ...int) Layer {
	in := -1
	if len(normalizedShape) > 0 {
		in = normalizedShape[0]
	}
	return layer.NewRMSNorm(in, 1e-5)
}

func SwiGLU(out int, in ...int) Layer {
	inSize := -1
	if len(in) > 0 {
		inSize = in[0]
	}
	return layer.NewSwiGLU(inSize, out)
}

func MaxPool2D(kernelSize, stride, padding int, inChannels ...int) Layer {
	in := -1
	if len(inChannels) > 0 {
		in = inChannels[0]
	}
	return layer.NewMaxPool2D(in, kernelSize, stride, padding)
}

func AvgPool2D(kernelSize, stride, padding int, inChannels ...int) Layer {
	in := -1
	if len(inChannels) > 0 {
		in = inChannels[0]
	}
	return layer.NewAvgPool2D(in, kernelSize, stride, padding)
}

func Flatten(inSize ...int) Layer {
	f := layer.NewFlatten()
	if len(inSize) > 0 {
		f.Build(inSize[0])
	}
	return f
}

func Embedding(numEmbeddings, embeddingDim int) Layer {
	return layer.NewEmbedding(numEmbeddings, embeddingDim)
}

func SequenceUnroller(l Layer, sequenceLength int, returnSequences bool) Layer {
	return layer.NewSequenceUnroller(l, sequenceLength, returnSequences)
}

func RBF(in, numCenters, out int, gamma float32) Layer {
	return layer.NewRBF(in, numCenters, out, gamma)
}

// TransformerBlock creates a Transformer block.
// Supports:
// - TransformerBlock(numHeads, seqLen, ffDim, causal) -> Lazy inference of dim
// - TransformerBlock(dim, numHeads, seqLen, ffDim, causal) -> Explicit dim
func TransformerBlock(args ...interface{}) Layer {
	if len(args) == 4 {
		numHeads := args[0].(int)
		seqLen := args[1].(int)
		ffDim := args[2].(int)
		causal := args[3].(bool)
		return layer.NewTransformerBlock(-1, numHeads, seqLen, ffDim, causal)
	}
	if len(args) == 5 {
		dim := args[0].(int)
		numHeads := args[1].(int)
		seqLen := args[2].(int)
		ffDim := args[3].(int)
		causal := args[4].(bool)
		return layer.NewTransformerBlock(dim, numHeads, seqLen, ffDim, causal)
	}
	panic("TransformerBlock expects 4 or 5 arguments")
}

func TransformerBlockExt(numHeads, seqLen, ffDim int, causal bool, actType layer.ActivationType, normType layer.NormType, useRoPE bool, dim ...int) Layer {
	inDim := -1
	if len(dim) > 0 {
		inDim = dim[0]
	}
	return layer.NewTransformerBlockExt(inDim, numHeads, seqLen, ffDim, causal, actType, normType, useRoPE)
}

// Transformer Constants
const (
	ActReLU   = layer.ActReLU
	ActSwiGLU = layer.ActSwiGLU
	ActGELU   = layer.ActGELU
	NormLN    = layer.NormLN
	NormRMS   = layer.NormRMS
)

func PositionalEncoding(seqLen int, dim ...int) Layer {
	inDim := -1
	if len(dim) > 0 {
		inDim = dim[0]
	}
	return layer.NewPositionalEncoding(seqLen, inDim)
}

func MultiHeadAttention(numHeads, seqLen int, causal bool, dim ...int) Layer {
	inDim := -1
	if len(dim) > 0 {
		inDim = dim[0]
	}
	return layer.NewMultiHeadAttention(inDim, numHeads, seqLen, causal)
}

func GlobalAveragePooling1D(seqLen, dim int) Layer {
	return layer.NewGlobalAveragePooling1D(seqLen, dim)
}

func CLSPooling(seqLen, dim int) Layer {
	return layer.NewCLSPooling(seqLen, dim)
}

func GlobalAttention(inSize int) Layer {
	return layer.NewGlobalAttention(inSize)
}

func Bidirectional(l Layer) Layer {
	return layer.NewBidirectional(l)
}

func MoE(outSize, numExperts, k int, in ...int) Layer {
	inSize := -1
	if len(in) > 0 {
		inSize = in[0]
	}
	return layer.NewMoE(inSize, outSize, numExperts, k)
}

// Optimizers
func Adam(lr float32) Optimizer {
	return opt.NewAdam(lr)
}

func SGD(lr float32) Optimizer {
	return &opt.SGD{LearningRate: lr}
}

func ReduceLROnPlateau(optimizer Optimizer, factor float32, patience int, threshold, minLR float32) *opt.ReduceLROnPlateau {
	return opt.NewReduceLROnPlateau(optimizer, factor, patience, threshold, minLR)
}

// Callbacks
type Callback = net.Callback

func Logger(interval int) net.Logger {
	return net.Logger{Interval: interval}
}

func ModelCheckpoint(filename string) net.Callback {
	return net.NewModelCheckpoint(filename)
}

func EarlyStopping(patience int, minDelta float32) *net.EarlyStopping {
	return net.NewEarlyStopping(patience, minDelta)
}

func SchedulerCallback(scheduler opt.Scheduler) net.Callback {
	return net.NewSchedulerCallback(scheduler)
}

func CSVLogger(filename string, append bool) net.Callback {
	return net.NewCSVLogger(filename, append)
}

// Devices
func GetDefaultDevice() layer.Device {
	return layer.GetDefaultDevice()
}

func CPUDevice() layer.Device {
	return &layer.CPUDevice{}
}

// Losses
var (
	MSE          = loss.MSE{}
	CrossEntropy = loss.CrossEntropy{}
	BCELoss      = loss.BCELoss{}
	NLLLoss      = loss.NLLLoss{}
)

func Huber(delta float32) Loss {
	return loss.NewHuber(delta)
}

// Data Loading
func LoadCSV(filename string, labelCols []int, hasHeader bool) (*Dataset, error) {
	return net.LoadCSV(filename, labelCols, hasHeader)
}

// Model Persistence
func Load(filename string) (*Model, Loss, error) {
	n, l, err := net.Load(filename)
	if err != nil {
		return nil, nil, err
	}
	return &net.Sequential{Network: n}, l, nil
}
