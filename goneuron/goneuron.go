package goneuron

import (
	"fmt"

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
//
// Deprecated: This function panics on invalid arguments. Use DenseE() instead for error handling.
func Dense(args ...interface{}) Layer {
	l, err := DenseE(args...)
	if err != nil {
		panic(err)
	}
	return l
}

// DenseE creates a fully connected layer and returns an error if arguments are invalid.
// Supports:
// - DenseE(out, act) -> Lazy inference of input size
// - DenseE(in, out, act) -> Explicit input size
// - DenseE(out, act, in) -> Explicit input size (alternative order)
func DenseE(args ...interface{}) (Layer, error) {
	if len(args) == 2 {
		out, ok1 := args[0].(int)
		act, ok2 := args[1].(activations.Activation)
		if !ok1 {
			return nil, fmt.Errorf("Dense: first argument must be int (output size), got %T", args[0])
		}
		if !ok2 {
			return nil, fmt.Errorf("Dense: second argument must be Activation, got %T", args[1])
		}
		return layer.NewDense(-1, out, act), nil
	}
	if len(args) == 3 {
		// Check if 2nd arg is int (in, out, act) or activation (out, act, in)
		if out, ok := args[1].(int); ok {
			in, ok1 := args[0].(int)
			act, ok3 := args[2].(activations.Activation)
			if !ok1 {
				return nil, fmt.Errorf("Dense: first argument must be int (input size), got %T", args[0])
			}
			if !ok3 {
				return nil, fmt.Errorf("Dense: third argument must be Activation, got %T", args[2])
			}
			return layer.NewDense(in, out, act), nil
		} else {
			out, ok1 := args[0].(int)
			act, ok2 := args[1].(activations.Activation)
			in, ok3 := args[2].(int)
			if !ok1 {
				return nil, fmt.Errorf("Dense: first argument must be int (output size), got %T", args[0])
			}
			if !ok2 {
				return nil, fmt.Errorf("Dense: second argument must be Activation, got %T", args[1])
			}
			if !ok3 {
				return nil, fmt.Errorf("Dense: third argument must be int (input size), got %T", args[2])
			}
			return layer.NewDense(in, out, act), nil
		}
	}
	return nil, fmt.Errorf("Dense expects 2 or 3 arguments: (out, act) or (in, out, act), got %d arguments", len(args))
}

// Conv2D creates a 2D convolutional layer.
// Supports:
// - Conv2D(outC, k, s, p, act) -> Lazy inference of inChannels
// - Conv2D(inC, outC, k, s, p, act) -> Explicit inChannels
// - Conv2D(outC, k, s, p, act, inC) -> Explicit inChannels (alternative order)
//
// Deprecated: This function panics on invalid arguments. Use Conv2DE() instead for error handling.
func Conv2D(args ...interface{}) Layer {
	l, err := Conv2DE(args...)
	if err != nil {
		panic(err)
	}
	return l
}

// Conv2DE creates a 2D convolutional layer and returns an error if arguments are invalid.
// Supports:
// - Conv2DE(outC, k, s, p, act) -> Lazy inference of inChannels
// - Conv2DE(inC, outC, k, s, p, act) -> Explicit inChannels
// - Conv2DE(outC, k, s, p, act, inC) -> Explicit inChannels (alternative order)
func Conv2DE(args ...interface{}) (Layer, error) {
	if len(args) == 5 {
		outC, ok1 := args[0].(int)
		k, ok2 := args[1].(int)
		s, ok3 := args[2].(int)
		p, ok4 := args[3].(int)
		act, ok5 := args[4].(activations.Activation)
		if !ok1 || !ok2 || !ok3 || !ok4 {
			return nil, fmt.Errorf("Conv2D: first 4 arguments must be int (outC, k, s, p)")
		}
		if !ok5 {
			return nil, fmt.Errorf("Conv2D: 5th argument must be Activation, got %T", args[4])
		}
		return layer.NewConv2D(-1, outC, k, s, p, act), nil
	}
	if len(args) == 6 {
		// Check if 6th arg is activation (inC, outC, k, s, p, act) or int (outC, k, s, p, act, inC)
		if act, ok := args[5].(activations.Activation); ok {
			inC, ok1 := args[0].(int)
			outC, ok2 := args[1].(int)
			k, ok3 := args[2].(int)
			s, ok4 := args[3].(int)
			p, ok5 := args[4].(int)
			if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 {
				return nil, fmt.Errorf("Conv2D: first 5 arguments must be int (inC, outC, k, s, p)")
			}
			return layer.NewConv2D(inC, outC, k, s, p, act), nil
		} else {
			outC, ok1 := args[0].(int)
			k, ok2 := args[1].(int)
			s, ok3 := args[2].(int)
			p, ok4 := args[3].(int)
			act, ok5 := args[4].(activations.Activation)
			inC, ok6 := args[5].(int)
			if !ok1 || !ok2 || !ok3 || !ok4 {
				return nil, fmt.Errorf("Conv2D: arguments 1-4 and 6 must be int")
			}
			if !ok5 {
				return nil, fmt.Errorf("Conv2D: 5th argument must be Activation, got %T", args[4])
			}
			if !ok6 {
				return nil, fmt.Errorf("Conv2D: 6th argument must be int (inC), got %T", args[5])
			}
			return layer.NewConv2D(inC, outC, k, s, p, act), nil
		}
	}
	return nil, fmt.Errorf("Conv2D expects 5 or 6 arguments: (outC, k, s, p, act) or (inC, outC, k, s, p, act), got %d arguments", len(args))
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
//
// Deprecated: This function panics on invalid arguments. Use TransformerBlockE() instead for error handling.
func TransformerBlock(args ...interface{}) Layer {
	l, err := TransformerBlockE(args...)
	if err != nil {
		panic(err)
	}
	return l
}

// TransformerBlockE creates a Transformer block and returns an error if arguments are invalid.
// Supports:
// - TransformerBlockE(numHeads, seqLen, ffDim, causal) -> Lazy inference of dim
// - TransformerBlockE(dim, numHeads, seqLen, ffDim, causal) -> Explicit dim
func TransformerBlockE(args ...interface{}) (Layer, error) {
	if len(args) == 4 {
		numHeads, ok1 := args[0].(int)
		seqLen, ok2 := args[1].(int)
		ffDim, ok3 := args[2].(int)
		causal, ok4 := args[3].(bool)
		if !ok1 || !ok2 || !ok3 {
			return nil, fmt.Errorf("TransformerBlock: first 3 arguments must be int (numHeads, seqLen, ffDim)")
		}
		if !ok4 {
			return nil, fmt.Errorf("TransformerBlock: 4th argument must be bool (causal), got %T", args[3])
		}
		return layer.NewTransformerBlock(-1, numHeads, seqLen, ffDim, causal), nil
	}
	if len(args) == 5 {
		dim, ok1 := args[0].(int)
		numHeads, ok2 := args[1].(int)
		seqLen, ok3 := args[2].(int)
		ffDim, ok4 := args[3].(int)
		causal, ok5 := args[4].(bool)
		if !ok1 || !ok2 || !ok3 || !ok4 {
			return nil, fmt.Errorf("TransformerBlock: first 4 arguments must be int (dim, numHeads, seqLen, ffDim)")
		}
		if !ok5 {
			return nil, fmt.Errorf("TransformerBlock: 5th argument must be bool (causal), got %T", args[4])
		}
		return layer.NewTransformerBlock(dim, numHeads, seqLen, ffDim, causal), nil
	}
	return nil, fmt.Errorf("TransformerBlock expects 4 or 5 arguments: (numHeads, seqLen, ffDim, causal) or (dim, numHeads, seqLen, ffDim, causal), got %d arguments", len(args))
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

// AdamW creates an AdamW optimizer with decoupled weight decay.
// weightDecay defaults to 0.01 if set to 0.
func AdamW(lr, weightDecay float32) *opt.AdamW {
	aw := opt.NewAdamW(lr)
	if weightDecay != 0 {
		aw.WeightDecay = weightDecay
	}
	return aw
}

func SGD(lr float32) Optimizer {
	return opt.NewSGD(lr)
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
	BCELoss           = loss.BCELoss{}
	BCEWithLogitsLoss = loss.BCEWithLogitsLoss{}
	NLLLoss           = loss.NLLLoss{}
	L1Loss            = loss.L1Loss{}
	KLDivLoss         = loss.KLDivLoss{}
)

func Huber(delta float32) Loss {
	return loss.NewHuber(delta)
}

func TripletMarginLoss(margin float32, p int) Loss {
	return loss.NewTripletMarginLoss(margin, p)
}

func MarginRankingLoss(margin float32) Loss {
	return loss.NewMarginRankingLoss(margin)
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
