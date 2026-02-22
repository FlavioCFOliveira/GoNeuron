package opt

import "math"

// Scheduler defines the interface for learning rate schedulers.
type Scheduler interface {
	Step()
	StepWithLoss(loss float64)
	GetLR() float64
}

// BaseScheduler provides default implementations for Scheduler.
type BaseScheduler struct{}

func (s BaseScheduler) Step()                {}
func (s BaseScheduler) StepWithLoss(loss float64) {}

// StepLR decays the learning rate of each parameter group by gamma every step_size epochs.
type StepLR struct {
	BaseScheduler
	optimizer Optimizer
	stepSize  int
	gamma     float64
	lastEpoch int
	initialLR float64
}

func NewStepLR(optimizer Optimizer, stepSize int, gamma float64, initialLR float64) *StepLR {
	return &StepLR{
		optimizer: optimizer,
		stepSize:  stepSize,
		gamma:     gamma,
		lastEpoch: 0,
		initialLR: initialLR,
	}
}

func (s *StepLR) Step() {
	s.lastEpoch++
	if s.lastEpoch%s.stepSize == 0 {
		state := s.optimizer.State()
		if lr, ok := state["LearningRate"].(float64); ok {
			state["LearningRate"] = lr * s.gamma
			s.optimizer.SetState(state)
		}
	}
}

func (s *StepLR) GetLR() float64 {
	state := s.optimizer.State()
	if lr, ok := state["LearningRate"].(float64); ok {
		return lr
	}
	return 0
}

// ExponentialLR decays the learning rate of each parameter group by gamma every epoch.
type ExponentialLR struct {
	BaseScheduler
	optimizer Optimizer
	gamma     float64
	initialLR float64
}

func NewExponentialLR(optimizer Optimizer, gamma float64, initialLR float64) *ExponentialLR {
	return &ExponentialLR{
		optimizer: optimizer,
		gamma:     gamma,
		initialLR: initialLR,
	}
}

func (s *ExponentialLR) Step() {
	state := s.optimizer.State()
	if lr, ok := state["LearningRate"].(float64); ok {
		state["LearningRate"] = lr * s.gamma
		s.optimizer.SetState(state)
	}
}

func (s *ExponentialLR) GetLR() float64 {
	state := s.optimizer.State()
	if lr, ok := state["LearningRate"].(float64); ok {
		return lr
	}
	return 0
}

// ReduceLROnPlateau reduces learning rate when a metric has stopped improving.
type ReduceLROnPlateau struct {
	BaseScheduler
	optimizer Optimizer
	factor    float64
	patience  int
	threshold float64
	cooldown  int
	minLR     float64

	bestLoss        float64
	numBadEpochs    int
	cooldownCounter int
}

func NewReduceLROnPlateau(optimizer Optimizer, factor float64, patience int, threshold float64, minLR float64) *ReduceLROnPlateau {
	return &ReduceLROnPlateau{
		optimizer: optimizer,
		factor:    factor,
		patience:  patience,
		threshold: threshold,
		minLR:     minLR,
		bestLoss:  math.MaxFloat64,
	}
}

func (s *ReduceLROnPlateau) StepWithLoss(currentLoss float64) {
	if s.cooldownCounter > 0 {
		s.cooldownCounter--
		return
	}

	if currentLoss < s.bestLoss-s.threshold {
		s.bestLoss = currentLoss
		s.numBadEpochs = 0
	} else {
		s.numBadEpochs++
	}

	if s.numBadEpochs >= s.patience {
		state := s.optimizer.State()
		if lr, ok := state["LearningRate"].(float64); ok {
			newLR := lr * s.factor
			if newLR < s.minLR {
				newLR = s.minLR
			}
			state["LearningRate"] = newLR
			s.optimizer.SetState(state)
			s.numBadEpochs = 0
			s.cooldownCounter = s.cooldown
		}
	}
}

func (s *ReduceLROnPlateau) GetLR() float64 {
	state := s.optimizer.State()
	if lr, ok := state["LearningRate"].(float64); ok {
		return lr
	}
	return 0
}
