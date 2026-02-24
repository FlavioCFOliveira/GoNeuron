package net

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// GGUF Constants
const (
	GGUFMagic   = 0x46554747 // "GGUF" in little-endian
	GGUFVersion = 3
)

// GGUF Value Types
type GGUFType uint32

const (
	GGUFTypeUint8   GGUFType = 0
	GGUFTypeInt8    GGUFType = 1
	GGUFTypeUint16  GGUFType = 2
	GGUFTypeInt16   GGUFType = 3
	GGUFTypeUint32  GGUFType = 4
	GGUFTypeInt32   GGUFType = 5
	GGUFTypeFloat32 GGUFType = 6
	GGUFTypeBool    GGUFType = 7
	GGUFTypeString  GGUFType = 8
	GGUFTypeArray   GGUFType = 9
	GGUFTypeUint64  GGUFType = 10
	GGUFTypeInt64   GGUFType = 11
	GGUFTypeFloat64 GGUFType = 12
)

// GGML Tensor Types
type GGMLType uint32

const (
	GGMLTypeF32  GGMLType = 0
	GGMLTypeF16  GGMLType = 1
	GGMLTypeQ4_0 GGMLType = 2
	GGMLTypeQ4_1 GGMLType = 3
	GGMLTypeQ5_0 GGMLType = 6
	GGMLTypeQ5_1 GGMLType = 7
	GGMLTypeQ8_0 GGMLType = 8
	GGMLTypeQ8_1 GGMLType = 9
)

// GGUFWriter helps writing GGUF files
type GGUFWriter struct {
	w         io.Writer
	alignment uint64
}

func NewGGUFWriter(w io.Writer) *GGUFWriter {
	return &GGUFWriter{
		w:         w,
		alignment: 32, // Default alignment
	}
}

func (gw *GGUFWriter) WriteHeader(kvCount, tensorCount uint64) error {
	if err := binary.Write(gw.w, binary.LittleEndian, uint32(GGUFMagic)); err != nil {
		return err
	}
	if err := binary.Write(gw.w, binary.LittleEndian, uint32(GGUFVersion)); err != nil {
		return err
	}
	if err := binary.Write(gw.w, binary.LittleEndian, tensorCount); err != nil {
		return err
	}
	if err := binary.Write(gw.w, binary.LittleEndian, kvCount); err != nil {
		return err
	}
	return nil
}

func (gw *GGUFWriter) WriteString(s string) error {
	n := uint64(len(s))
	if err := binary.Write(gw.w, binary.LittleEndian, n); err != nil {
		return err
	}
	_, err := gw.w.Write([]byte(s))
	return err
}

func (gw *GGUFWriter) WriteKV(key string, valType GGUFType, value interface{}) error {
	if err := gw.WriteString(key); err != nil {
		return err
	}
	if err := binary.Write(gw.w, binary.LittleEndian, uint32(valType)); err != nil {
		return err
	}

	switch valType {
	case GGUFTypeUint8:
		return binary.Write(gw.w, binary.LittleEndian, value.(uint8))
	case GGUFTypeInt8:
		return binary.Write(gw.w, binary.LittleEndian, value.(int8))
	case GGUFTypeUint16:
		return binary.Write(gw.w, binary.LittleEndian, value.(uint16))
	case GGUFTypeInt16:
		return binary.Write(gw.w, binary.LittleEndian, value.(int16))
	case GGUFTypeUint32:
		return binary.Write(gw.w, binary.LittleEndian, value.(uint32))
	case GGUFTypeInt32:
		return binary.Write(gw.w, binary.LittleEndian, value.(int32))
	case GGUFTypeFloat32:
		return binary.Write(gw.w, binary.LittleEndian, value.(float32))
	case GGUFTypeUint64:
		return binary.Write(gw.w, binary.LittleEndian, value.(uint64))
	case GGUFTypeInt64:
		return binary.Write(gw.w, binary.LittleEndian, value.(int64))
	case GGUFTypeFloat64:
		return binary.Write(gw.w, binary.LittleEndian, value.(float64))
	case GGUFTypeBool:
		var b uint8
		if value.(bool) {
			b = 1
		}
		return binary.Write(gw.w, binary.LittleEndian, b)
	case GGUFTypeString:
		return gw.WriteString(value.(string))
	case GGUFTypeArray:
		// value should be a struct { Type GGUFType; Data interface{} }
		// or handle it separately
		return fmt.Errorf("GGUFTypeArray not fully implemented in helper")
	default:
		return fmt.Errorf("unsupported GGUF type: %v", valType)
	}
}

func (gw *GGUFWriter) WriteTensorInfo(name string, shape []uint64, ggmlType GGMLType, offset uint64) error {
	if err := gw.WriteString(name); err != nil {
		return err
	}
	rank := uint32(len(shape))
	if err := binary.Write(gw.w, binary.LittleEndian, rank); err != nil {
		return err
	}
	// GGUF dimensions are in reverse order (last dimension first)
	for i := 0; i < int(rank); i++ {
		if err := binary.Write(gw.w, binary.LittleEndian, shape[rank-1-uint32(i)]); err != nil {
			return err
		}
	}
	if err := binary.Write(gw.w, binary.LittleEndian, uint32(ggmlType)); err != nil {
		return err
	}
	if err := binary.Write(gw.w, binary.LittleEndian, offset); err != nil {
		return err
	}
	return nil
}

// Float32ToFloat16 converts a float32 to float16 (represented as uint16)
func Float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	s := uint16((bits >> 16) & 0x8000)
	e := int16((bits >> 23) & 0xFF)
	m := bits & 0x7FFFFF

	if e == 0 {
		// Zero or denormal
		return s
	} else if e == 0xFF {
		// Inf or NaN
		if m == 0 {
			return s | 0x7C00
		}
		return s | 0x7C00 | uint16(m>>13) | 1
	}

	e -= 127 - 15
	if e >= 31 {
		// Overflow to Inf
		return s | 0x7C00
	} else if e <= 0 {
		// Underflow to denormal or zero
		if e < -10 {
			return s
		}
		m |= 0x800000
		m >>= uint32(1 - e)
		return s | uint16(m>>13)
	}

	return s | uint16(e<<10) | uint16(m>>13)
}
