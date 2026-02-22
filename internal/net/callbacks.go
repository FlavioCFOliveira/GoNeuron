package net

import (
	"fmt"
	"math"

	"github.com/FlavioCFOliveira/GoNeuron/internal/opt"
)

// Callback defines the interface for training callbacks.
type Callback interface {
	OnTrainBegin(n *Network)
	OnTrainEnd(n *Network)
	OnEpochBegin(epoch int, n *Network)
	OnEpochEnd(epoch int, loss float32, n *Network)
	OnBatchBegin(batch int, n *Network)
	OnBatchEnd(batch int, loss float32, n *Network)
}

// SchedulerCallback is a callback that wraps a learning rate scheduler.
type SchedulerCallback struct {
	BaseCallback
	scheduler opt.Scheduler
}

func NewSchedulerCallback(scheduler opt.Scheduler) *SchedulerCallback {
	return &SchedulerCallback{scheduler: scheduler}
}

func (c *SchedulerCallback) OnEpochEnd(epoch int, loss float32, n *Network) {
	c.scheduler.Step()
	c.scheduler.StepWithLoss(loss)
}

// BaseCallback provides default empty implementations for Callback.
type BaseCallback struct{}

func (c BaseCallback) OnTrainBegin(n *Network)                {}
func (c BaseCallback) OnTrainEnd(n *Network)                  {}
func (c BaseCallback) OnEpochBegin(epoch int, n *Network)     {}
func (c BaseCallback) OnEpochEnd(epoch int, loss float32, n *Network) {}
func (c BaseCallback) OnBatchBegin(batch int, n *Network)     {}
func (c BaseCallback) OnBatchEnd(batch int, loss float32, n *Network)   {}

// EarlyStopping stops training when a monitored metric has stopped improving.
type EarlyStopping struct {
	BaseCallback
	Patience  int
	Threshold float32
	Monitor   string // "loss" (default)

	bestLoss     float32
	numBadEpochs int
	Stopped      bool
}

func NewEarlyStopping(patience int, threshold float32) *EarlyStopping {
	return &EarlyStopping{
		Patience:  patience,
		Threshold: threshold,
		bestLoss:  math.MaxFloat32,
	}
}

func (c *EarlyStopping) OnEpochEnd(epoch int, loss float32, n *Network) {
	if loss < c.bestLoss-c.Threshold {
		c.bestLoss = loss
		c.numBadEpochs = 0
	} else {
		c.numBadEpochs++
	}

	if c.numBadEpochs >= c.Patience {
		fmt.Printf("\nEarly stopping at epoch %d: loss %.6f did not improve for %d epochs\n", epoch, loss, c.Patience)
		c.Stopped = true
	}
}

// ModelCheckpoint saves the model after every epoch if it's the best so far.
type ModelCheckpoint struct {
	BaseCallback
	Filename string
	Monitor  string // "loss" (default)

	bestLoss float32
}

func NewModelCheckpoint(filename string) *ModelCheckpoint {
	return &ModelCheckpoint{
		Filename: filename,
		bestLoss: math.MaxFloat32,
	}
}

func (c *ModelCheckpoint) OnEpochEnd(epoch int, loss float32, n *Network) {
	if loss < c.bestLoss {
		c.bestLoss = loss
		err := n.Save(c.Filename)
		if err != nil {
			fmt.Printf("Error saving checkpoint: %v\n", err)
		} else {
			fmt.Printf("Checkpoint saved: loss %.6f is new best\n", loss)
		}
	}
}

// Logger logs training progress to console.
type Logger struct {
	BaseCallback
	Interval int
}

func (c Logger) OnEpochEnd(epoch int, loss float32, n *Network) {
	if c.Interval > 0 && epoch%c.Interval == 0 {
		fmt.Printf("Epoch %d: loss = %.6f\n", epoch, loss)
	}
}
