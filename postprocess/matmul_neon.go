//go:build arm64 && cgo

package postprocess

/*
#cgo CFLAGS: -O3 -march=armv8-a+simd
#include <stdint.h>
#include <stddef.h>

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

func matmulUint8(
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


