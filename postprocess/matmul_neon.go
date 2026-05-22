//go:build arm64 && cgo

package postprocess

/*
#cgo CFLAGS: -O3 -march=armv8-a+simd
#include <stdint.h>
#include <stddef.h>

// matmul_mask_f32_neon performs the segmentation prototype matrix
// multiplication using ARM NEON SIMD acceleration.
//
// The function computes:
//
//   output_mask = sum(proto[channel] * coeff[channel])
//
// for one or more detected objects.
//
// Output pixels are thresholded:
//
//   > 0.0f -> 4
//   <= 0.0f -> 0
//
// The output value 4 intentionally matches the original Go
// implementation so later OpenCV resize/interpolation produces
// visually identical mask edges.
void matmul_mask_f32_neon(
	const float *proto,
	const float *coeffs,
	uint8_t *out,
	int boxes,
	int channels,
	int area
);
*/
import "C"

import (
	"runtime"
	"sync"
	"unsafe"
)

// matmulUint8 generates binary segmentation masks from YOLO prototype
// masks and mask coefficients using a NEON-accelerated C implementation.
//
// The output mask format matches the original Go implementation:
//
//	object pixels     = 4
//	background pixels = 0
//
// Parameters:
//
//	data      - segmentation prototype and coefficient data
//	boxesNum  - number of detected objects
//	protoC    - number of prototype channels
//	protoH/W  - prototype mask dimensions
//	out       - output binary masks buffer
func matmulUint8(
	data *strideData,
	boxesNum int,
	protoC int,
	protoH int,
	protoW int,
	out []uint8,
) {
	// total pixels per prototype mask
	area := protoH * protoW

	// validate dimensions before calling into C
	if boxesNum <= 0 || protoC <= 0 || area <= 0 {
		return
	}

	// run NEON-accelerated prototype matrix multiplication
	C.matmul_mask_f32_neon(
		(*C.float)(unsafe.Pointer(&data.proto[0])),
		(*C.float)(unsafe.Pointer(&data.filterSegmentsByNMS[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
		C.int(boxesNum),
		C.int(protoC),
		C.int(area),
	)
}

func matmulUint8Parallel(
	data *strideData,
	boxesNum int,
	protoC int,
	protoH int,
	protoW int,
	out []uint8,
) {
	area := protoH * protoW
	if boxesNum <= 0 || protoC <= 0 || area <= 0 {
		return
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > boxesNum {
		workers = boxesNum
	}
	if workers < 1 {
		workers = 1
	}

	var wg sync.WaitGroup
	wg.Add(workers)

	for worker := 0; worker < workers; worker++ {
		startBox := boxesNum * worker / workers
		endBox := boxesNum * (worker + 1) / workers
		count := endBox - startBox

		go func(startBox int, count int) {
			defer wg.Done()

			coeffOffset := startBox * protoC
			outOffset := startBox * area

			C.matmul_mask_f32_neon(
				(*C.float)(unsafe.Pointer(&data.proto[0])),
				(*C.float)(unsafe.Pointer(&data.filterSegmentsByNMS[coeffOffset])),
				(*C.uint8_t)(unsafe.Pointer(&out[outOffset])),
				C.int(count),
				C.int(protoC),
				C.int(area),
			)
		}(startBox, count)
	}

	wg.Wait()
}
