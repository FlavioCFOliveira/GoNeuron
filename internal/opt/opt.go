// Package opt provides optimization algorithms.
package opt

import "math"

// Optimizer updates network parameters based on gradients.
type Optimizer interface {

	// Step computes updated parameters: params - lr * gradients
	// Returns a new slice with updated values
	Step(params, gradients []float32) []float32

	// StepInPlace updates params in-place: params = params - lr * gradients
	// This avoids allocations for better performance
	StepInPlace(params, gradients []float32)

	// NewStep signals the beginning of a new optimization step (e.g., for Adam timestep).
	NewStep()

	// State returns the optimizer state for serialization.
	State() map[string]any

	// SetState sets the optimizer state from a serialized map.
	SetState(state map[string]any)

	// GobState returns the optimizer state in a gob-encodable format.
	// This is used for checkpointing with gob encoding.
	GobState() any

	// SetGobState sets the optimizer state from a gob-encodable format.
	SetGobState(state any)
}

// SGD (Stochastic Gradient Descent) optimizer.
type SGD struct {
	LearningRate float32
}

// Step computes updated parameters: params - lr * gradients
func (s *SGD) Step(params, gradients []float32) []float32 {
	result := make([]float32, len(params))
	for i := range params {
		result[i] = params[i] - s.LearningRate*gradients[i]
	}
	return result
}

// StepInPlace updates params in-place: params = params - lr * gradients
// This avoids allocations for better performance
func (s *SGD) StepInPlace(params, gradients []float32) {
	for i := range params {
		params[i] -= s.LearningRate * gradients[i]
	}
}

// NewStep signals the beginning of a new optimization step (no-op for SGD).
func (s *SGD) NewStep() {}

// State returns SGD state.
func (s *SGD) State() map[string]any {
	return map[string]any{
		"LearningRate": s.LearningRate,
	}
}

// SetState sets SGD state.
func (s *SGD) SetState(state map[string]any) {
	if lr, ok := state["LearningRate"].(float32); ok {
		s.LearningRate = lr
	}
}

// GobState returns SGD state in a gob-encodable format.
func (s *SGD) GobState() any {
	return *s
}

// SetGobState sets SGD state from a gob-decoded value.
func (s *SGD) SetGobState(state any) {
	if st, ok := state.(SGD); ok {
		s.LearningRate = st.LearningRate
	}
}

// Adam optimizer for faster convergence.
// Adam maintains per-parameter state (first and second moment estimates).
// Uses flat slices for momentum buffers (one per layer in the network).
type Adam struct {
	LearningRate float32
	Beta1        float32 // Exponential decay rate for first moment
	Beta2        float32 // Exponential decay rate for second moment
	Epsilon      float32 // Small constant for numerical stability

	// Per-layer state - indexed by layer
	// momentum1[i] and momentum2[i] are for layer i
	momentum1 [][]float32 // First moment (m) for each layer
	momentum2 [][]float32 // Second moment (v) for each layer

	// Internal counter to track layer index during training
	// This is set by the network before calling Step
	currentLayer int

	// Global timestep for bias correction
	timestep int
}

// NewAdam creates a new Adam optimizer with default values.
func NewAdam(learningRate float32) *Adam {
	return &Adam{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		currentLayer: 0,
		timestep:     0,
	}
}

// initMoments initializes the moment buffers for given parameter shape.
func (a *Adam) initMoments(params []float32) {
	// Ensure we have enough layer buffers
	for len(a.momentum1) <= a.currentLayer {
		a.momentum1 = append(a.momentum1, nil)
		a.momentum2 = append(a.momentum2, nil)
	}

	// Initialize or resize the buffers for current layer
	if a.momentum1[a.currentLayer] == nil || len(a.momentum1[a.currentLayer]) != len(params) {
		a.momentum1[a.currentLayer] = make([]float32, len(params))
		a.momentum2[a.currentLayer] = make([]float32, len(params))
	}
}

// Step computes updated parameters using Adam
func (a *Adam) Step(params, gradients []float32) []float32 {
	a.initMoments(params)

	currentLayer := a.currentLayer
	a.currentLayer++

	lr := a.LearningRate
	// Use default values if not set
	beta1 := a.Beta1
	if beta1 == 0 {
		beta1 = 0.9
	}
	beta2 := a.Beta2
	if beta2 == 0 {
		beta2 = 0.999
	}
	eps := a.Epsilon
	if eps == 0 {
		eps = 1e-8
	}

	m := a.momentum1[currentLayer]
	v := a.momentum2[currentLayer]

	// Bias correction: timestep is the number of steps already taken
	// For first step (timestep=0), use t=1 to avoid division by zero
	ts := float64(a.timestep + 1)
	bias1 := float32(1 - math.Pow(float64(beta1), ts))
	bias2 := float32(1 - math.Pow(float64(beta2), ts))

	// Effective learning rate with bias correction
	// lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
	effLR := lr * float32(math.Sqrt(float64(bias2))) / bias1

	result := make([]float32, len(params))
	for i := range params {
		// Update biased first moment estimate
		m[i] = beta1*m[i] + (1-beta1)*gradients[i]
		// Update biased second moment estimate
		v[i] = beta2*v[i] + (1-beta2)*gradients[i]*gradients[i]
		// Update parameters
		result[i] = params[i] - effLR*m[i]/(float32(math.Sqrt(float64(v[i])))+eps)
	}
	return result
}

// StepInPlace updates params in-place using Adam
func (a *Adam) StepInPlace(params, gradients []float32) {
	a.initMoments(params)

	currentLayer := a.currentLayer
	a.currentLayer++

	lr := a.LearningRate
	// Use default values if not set
	beta1 := a.Beta1
	if beta1 == 0 {
		beta1 = 0.9
	}
	beta2 := a.Beta2
	if beta2 == 0 {
		beta2 = 0.999
	}
	eps := a.Epsilon
	if eps == 0 {
		eps = 1e-8
	}

	m := a.momentum1[currentLayer]
	v := a.momentum2[currentLayer]

	// Bias correction: timestep is the number of steps already taken
	// For first step (timestep=0), use t=1 to avoid division by zero
	ts := float64(a.timestep + 1)
	bias1 := float32(1 - math.Pow(float64(beta1), ts))
	bias2 := float32(1 - math.Pow(float64(beta2), ts))

	// Effective learning rate with bias correction
	effLR := lr * float32(math.Sqrt(float64(bias2))) / bias1

	for i := range params {
		// Update biased first moment estimate
		m[i] = beta1*m[i] + (1-beta1)*gradients[i]
		// Update biased second moment estimate
		v[i] = beta2*v[i] + (1-beta2)*gradients[i]*gradients[i]
		// Update parameters in-place
		params[i] -= effLR * m[i] / (float32(math.Sqrt(float64(v[i]))) + eps)
	}
}

// NewStep signals the beginning of a new optimization step for Adam.
func (a *Adam) NewStep() {
	a.timestep++
	a.currentLayer = 0
}

// adamGobState is a gob-encodable state for Adam optimizer.
// This struct uses only gob-compatible types (slice of slice of float32).
type adamGobState struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	Momentum1    [][]float32
	Momentum2    [][]float32
	Timestep     int
}

// State returns Adam state.
func (a *Adam) State() map[string]any {
	return map[string]any{
		"LearningRate": a.LearningRate,
		"Beta1":        a.Beta1,
		"Beta2":        a.Beta2,
		"Epsilon":      a.Epsilon,
		"momentum1":    a.momentum1,
		"momentum2":    a.momentum2,
		"timestep":     a.timestep,
	}
}

// SetState sets Adam state from a map.
// This is used for JSON encoding and other non-gob serialization.
func (a *Adam) SetState(state map[string]any) {
	if lr, ok := state["LearningRate"].(float32); ok {
		a.LearningRate = lr
	}
	if b1, ok := state["Beta1"].(float32); ok {
		a.Beta1 = b1
	}
	if b2, ok := state["Beta2"].(float32); ok {
		a.Beta2 = b2
	}
	if eps, ok := state["Epsilon"].(float32); ok {
		a.Epsilon = eps
	}
	if m1, ok := state["momentum1"].([][]float32); ok {
		a.momentum1 = m1
	}
	if m2, ok := state["momentum2"].([][]float32); ok {
		a.momentum2 = m2
	}
	if ts, ok := state["timestep"].(int); ok {
		a.timestep = ts
	}
}

// GobState returns Adam state in a gob-encodable format.
func (a *Adam) GobState() any {
	return adamGobState{
		LearningRate: a.LearningRate,
		Beta1:        a.Beta1,
		Beta2:        a.Beta2,
		Epsilon:      a.Epsilon,
		Momentum1:    a.momentum1,
		Momentum2:    a.momentum2,
		Timestep:     a.timestep,
	}
}

// SetGobState sets Adam state from a gob-decoded value.
func (a *Adam) SetGobState(state any) {
	if st, ok := state.(adamGobState); ok {
		a.LearningRate = st.LearningRate
		a.Beta1 = st.Beta1
		a.Beta2 = st.Beta2
		a.Epsilon = st.Epsilon
		a.momentum1 = st.Momentum1
		a.momentum2 = st.Momentum2
		a.timestep = st.Timestep
	}
}
