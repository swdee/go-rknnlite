package preprocess

/*
#cgo CFLAGS:   -I/usr/include/rga
#cgo CXXFLAGS: -std=c++11 -I/usr/include/rga
#cgo LDFLAGS: -lrga -lstdc++
#include "rga_resize.h"
#include <rga.h>
*/
import "C"

import (
	"fmt"
	"unsafe"

	"gocv.io/x/gocv"
)

// ResizeRGA resizes src to dst using the RK3588 RGA.
// Both src and dst must be CV_8UC4 (BGRA) Mats, continuous in memory.
func ResizeRGA(src, dst gocv.Mat) error {

	if !src.IsContinuous() || !dst.IsContinuous() {
		return fmt.Errorf("mats must be continuous")
	}

	// Get a Go slice backed by the Mat’s own buffer:
	srcBuf, err := src.DataPtrUint8()

	if err != nil {
		return fmt.Errorf("src.DataPtrUint8: %v", err)
	}

	dstBuf, err := dst.DataPtrUint8()

	if err != nil {
		return fmt.Errorf("dst.DataPtrUint8: %v", err)
	}

	// Take the address of the first element in the real cv::Mat.data pointer
	srcPtr := unsafe.Pointer(&srcBuf[0])
	dstPtr := unsafe.Pointer(&dstBuf[0])

	status := C.resize_rga(
		srcPtr,
		C.int(src.Cols()), C.int(src.Rows()), C.int(C.RK_FORMAT_BGRA_8888),
		dstPtr,
		C.int(dst.Cols()), C.int(dst.Rows()), C.int(C.RK_FORMAT_BGRA_8888),
	)

	if status != 0 {
		return fmt.Errorf("RGA resize failed: %d", int(status))
	}

	return nil
}

var (
	initDone bool
)

// InitRGA pins, imports, wraps & checks your two Mats.
// Must be called once before any ResizeRGAFrame calls.
func InitRGA(src, dst gocv.Mat) error {

	if initDone {
		return nil
	}

	if !src.IsContinuous() || !dst.IsContinuous() {
		return fmt.Errorf("mats must be continuous")
	}

	// get backing []uint8 slices
	srcBuf, err := src.DataPtrUint8()

	if err != nil {
		return fmt.Errorf("src.DataPtrUint8: %v", err)
	}

	dstBuf, err := dst.DataPtrUint8()

	if err != nil {
		return fmt.Errorf("dst.DataPtrUint8: %v", err)
	}

	// call C init
	ret := C.resize_rga_init(
		unsafe.Pointer(&srcBuf[0]),
		C.int(src.Cols()), C.int(src.Rows()), C.int(C.RK_FORMAT_BGRA_8888),
		unsafe.Pointer(&dstBuf[0]),
		C.int(dst.Cols()), C.int(dst.Rows()), C.int(C.RK_FORMAT_BGRA_8888),
	)

	if ret != 0 {
		return fmt.Errorf("resize_rga_init failed: %d", int(ret))
	}

	initDone = true
	return nil
}

// ResizeRGAFrame runs one hardware rescale on the buffers initialized above.
// Must call InitRGA first.
func ResizeRGAFrame() error {

	if !initDone {
		return fmt.Errorf("ResizeRGAFrame called before InitRGA")
	}

	ret := C.resize_rga_frame()

	if ret != 0 {
		return fmt.Errorf("resize_rga_frame failed: %d", int(ret))
	}

	return nil
}

// CloseRGA tears down the RGA handles.
// Call this once when you’re done (e.g. at program exit).
func CloseRGA() {

	if !initDone {
		return
	}

	C.resize_rga_deinit()
	initDone = false
}
