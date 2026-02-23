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
func Dense(out int, act activations.Activation, in ...int) Layer {
	inSize := -1
	if len(in) > 0 {
		inSize = in[0]
	}
	return layer.NewDense(inSize, out, act)
}

func Conv2D(outChannels, kernelSize, stride, padding int, act activations.Activation, in ...int) Layer {
	inChannels := -1
	if len(in) > 0 {
		inChannels = in[0]
	}
	return layer.NewConv2D(inChannels, outChannels, kernelSize, stride, padding, act)
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

func Dropout(prob float32, in int) Layer {
	return layer.NewDropout(prob, in)
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

func TransformerBlock(numHeads, seqLen, ffDim int, causal bool, dim ...int) Layer {
	inDim := -1
	if len(dim) > 0 {
		inDim = dim[0]
	}
	return layer.NewTransformerBlock(inDim, numHeads, seqLen, ffDim, causal)
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
