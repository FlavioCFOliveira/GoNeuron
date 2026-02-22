package metal

import (
	"runtime"
	"testing"
)

func TestMetalAvailability(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal is only available on macOS")
	}

	available := IsAvailable()
	t.Logf("Metal available: %v", available)
}

func TestDeviceAndBuffer(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal is only available on macOS")
	}

	if !IsAvailable() {
		t.Skip("Metal not available on this device")
	}

	dev := NewDevice()
	if dev == nil {
		t.Fatal("Failed to initialize Metal device")
	}

	// Create a buffer for 100 float32s
	size := 100 * 4
	buf := dev.NewBuffer(size)
	if buf == nil {
		t.Fatal("Failed to create shared buffer")
	}

	data := buf.Float32Data()
	if len(data) != 100 {
		t.Errorf("Expected buffer length 100, got %d", len(data))
	}

	// Test sync from float64
	src := make([]float64, 100)
	for i := range src {
		src[i] = float64(i)
	}
	buf.SyncFromFloat64(src)

	// Verify data in buffer
	for i, v := range data {
		if v != float32(i) {
			t.Errorf("Index %d: expected %f, got %f", i, float32(i), v)
		}
	}

	// Test sync to float64
	dst := make([]float64, 100)
	buf.SyncToFloat64(dst)
	for i, v := range dst {
		if v != float64(i) {
			t.Errorf("Index %d: expected %f, got %f", i, float64(i), v)
		}
	}
}

func TestMatMul(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal is only available on macOS")
	}

	if !IsAvailable() {
		t.Skip("Metal not available on this device")
	}

	dev := NewDevice()
	if dev == nil {
		t.Fatal("Failed to initialize Metal device")
	}

	// 2x3 * 3x2 = 2x2
	m, n, k := 2, 2, 3

	// A = [1 2 3; 4 5 6]
	// B = [7 8; 9 10; 11 12]
	// C = A * B

	bufA := dev.NewBuffer(m * k * 4)
	bufB := dev.NewBuffer(k * n * 4)
	bufC := dev.NewBuffer(m * n * 4)

	bufA.SyncFromFloat64([]float64{1, 2, 3, 4, 5, 6})
	bufB.SyncFromFloat64([]float64{7, 8, 9, 10, 11, 12})

	dev.MatMul(bufA, bufB, bufC, m, n, k)

	res := make([]float64, m*n)
	bufC.SyncToFloat64(res)

	// Expected:
	// [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [7+18+33, 8+20+36] = [58, 64]
	// [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [28+45+66, 32+50+72] = [139, 154]

	expected := []float64{58, 64, 139, 154}
	for i, v := range res {
		if v != expected[i] {
			t.Errorf("Index %d: expected %f, got %f", i, expected[i], v)
		}
	}
}
