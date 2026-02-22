package opt

import "math"

// Scheduler defines the interface for learning rate schedulers.
type Scheduler interface {
	Step()
	StepWithLoss(loss float32)
	GetLR() float32
}

// BaseScheduler provides default implementations for Scheduler.
type BaseScheduler struct{}

func (s BaseScheduler) Step()                {}
func (s BaseScheduler) StepWithLoss(loss float32) {}

// StepLR decays the learning rate of each parameter group by gamma every step_size epochs.
type StepLR struct {
	BaseScheduler
	optimizer Optimizer
	stepSize  int
	gamma     float32
	lastEpoch int
	initialLR float32
}

func NewStepLR(optimizer Optimizer, stepSize int, gamma float32, initialLR float32) *StepLR {
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
		if lr, ok := state["LearningRate"].(float32); ok {
			state["LearningRate"] = lr * s.gamma
			s.optimizer.SetState(state)
		}
	}
}

func (s *StepLR) GetLR() float32 {
	state := s.optimizer.State()
	if lr, ok := state["LearningRate"].(float32); ok {
		return lr
	}
	return 0
}

// ExponentialLR decays the learning rate of each parameter group by gamma every epoch.
type ExponentialLR struct {
	BaseScheduler
	optimizer Optimizer
	gamma     float32
	initialLR float32
}

func NewExponentialLR(optimizer Optimizer, gamma float32, initialLR float32) *ExponentialLR {
	return &ExponentialLR{
		optimizer: optimizer,
		gamma:     gamma,
		initialLR: initialLR,
	}
}

func (s *ExponentialLR) Step() {
	state := s.optimizer.State()
	if lr, ok := state["LearningRate"].(float32); ok {
		state["LearningRate"] = lr * s.gamma
		s.optimizer.SetState(state)
	}
}

func (s *ExponentialLR) GetLR() float32 {
	state := s.optimizer.State()
	if lr, ok := state["LearningRate"].(float32); ok {
		return lr
	}
	return 0
}

// ReduceLROnPlateau reduces learning rate when a metric has stopped improving.
type ReduceLROnPlateau struct {
	BaseScheduler
	optimizer Optimizer
	factor    float32
	patience  int
	threshold float32
	cooldown  int
	minLR     float32

	bestLoss        float32
	numBadEpochs    int
	cooldownCounter int
}

func NewReduceLROnPlateau(optimizer Optimizer, factor float32, patience int, threshold float32, minLR float32) *ReduceLROnPlateau {
	return &ReduceLROnPlateau{
		optimizer: optimizer,
		factor:    factor,
		patience:  patience,
		threshold: threshold,
		minLR:     minLR,
		bestLoss:  math.MaxFloat32,
	}
}

func (s *ReduceLROnPlateau) StepWithLoss(currentLoss float32) {
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
		if lr, ok := state["LearningRate"].(float32); ok {
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

func (s *ReduceLROnPlateau) GetLR() float32 {
	state := s.optimizer.State()
	if lr, ok := state["LearningRate"].(float32); ok {
		return lr
	}
	return 0
}
