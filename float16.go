package rknnlite

import "github.com/x448/float16"

var f16LookupTable [65536]float32

func init() {
	// precompute float16 lookup table for faster conversion to float32
	for i := range f16LookupTable {
		f16 := float16.Frombits(uint16(i))
		f16LookupTable[i] = f16.Float32()
	}
}
