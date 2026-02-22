package layer

import (
	"runtime"
)

// DeviceType represents the hardware device used for computation.
type DeviceType int

const (
	CPU DeviceType = iota
	GPU
)

// Device manages the hardware resources for neural network operations.
type Device interface {
	Type() DeviceType
	IsAvailable() bool
}

// CPUDevice handles computations on the host CPU.
type CPUDevice struct{}

func (d *CPUDevice) Type() DeviceType { return CPU }
func (d *CPUDevice) IsAvailable() bool { return true }

// GetDefaultDevice returns the best available device for the current platform.
func GetDefaultDevice() Device {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		// Metal is likely available on Apple Silicon
		gpu := NewMetalDevice()
		if gpu.IsAvailable() {
			return gpu
		}
	}
	return &CPUDevice{}
}
