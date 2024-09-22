package rknnlite

/*
#cgo CFLAGS: -march=native -mtune=native -Ofast -flto
#cgo LDFLAGS: -march=native -mtune=native -Ofast

#include <stdint.h>

void float16_to_float32_buffer(const uint16_t* input, float* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        _Float16 tmp = *(_Float16*)&input[i];
        output[i] = (float)tmp;
    }
}

*/
import "C"
import (
	"unsafe"
)

// float16toFloat32Buffer takes a float16 and 32 buffer and converts it using
// optimisation via C
func float16ToFloat32Buffer(float16Buf []uint16, float32Buf []float32) {
	C.float16_to_float32_buffer(
		(*C.uint16_t)(unsafe.Pointer(&float16Buf[0])), // Pointer to the input buffer
		(*C.float)(unsafe.Pointer(&float32Buf[0])),    // Pointer to the output buffer
		C.size_t(len(float16Buf)),                     // Number of elements to convert
	)
}
