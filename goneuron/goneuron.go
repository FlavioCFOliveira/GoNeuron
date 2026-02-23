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
func Dense(in, out int, act activations.Activation) Layer {
	return layer.NewDense(in, out, act)
}

func Conv2D(inChannels, outChannels, kernelSize, stride, padding int, act activations.Activation) Layer {
	return layer.NewConv2D(inChannels, outChannels, kernelSize, stride, padding, act)
}

func LSTM(in, out int) Layer {
	return layer.NewLSTM(in, out)
}

func GRU(in, out int) Layer {
	return layer.NewGRU(in, out)
}

func Dropout(prob float32, in int) Layer {
	return layer.NewDropout(prob, in)
}

func BatchNorm2D(inChannels int) Layer {
	return layer.NewBatchNorm2D(inChannels, 1e-5, 0.1, true)
}

func LayerNorm(normalizedShape int) Layer {
	return layer.NewLayerNorm(normalizedShape, 1e-5, true)
}

func MaxPool2D(inChannels, kernelSize, stride, padding int) Layer {
	return layer.NewMaxPool2D(inChannels, kernelSize, stride, padding)
}

func Flatten() Layer {
	return layer.NewFlatten()
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

func TransformerBlock(dim, numHeads, seqLen, ffDim int, causal bool) Layer {
	return layer.NewTransformerBlock(dim, numHeads, seqLen, ffDim, causal)
}

func PositionalEncoding(seqLen, dim int) Layer {
	return layer.NewPositionalEncoding(seqLen, dim)
}

func MultiHeadAttention(dim, numHeads, seqLen int, causal bool) Layer {
	return layer.NewMultiHeadAttention(dim, numHeads, seqLen, causal)
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

func MoE(inSize, outSize, numExperts, k int) Layer {
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

// Model Persistence
func Load(filename string) (*Model, Loss, error) {
	n, l, err := net.Load(filename)
	if err != nil {
		return nil, nil, err
	}
	return &net.Sequential{Network: n}, l, nil
}
