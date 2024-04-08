package rknnlite

/*
#include "rknn_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// QueryModelIONumber queries the number of Input and Output tensors of the model
func (r *Runtime) QueryModelIONumber() (ioNum IONumber, err error) {

	// prepare the structure to receive the Input/Output number
	var cIONum C.rknn_input_output_num

	// call the C function
	ret := C.rknn_query(r.ctx, C.RKNN_QUERY_IN_OUT_NUM, unsafe.Pointer(&cIONum), C.uint(C.sizeof_rknn_input_output_num))

	if ret != C.RKNN_SUCC {
		return IONumber{}, fmt.Errorf("rknn_query failed with return code %d", int(ret))
	}

	ioNum = IONumber{
		NumberInput:  uint32(cIONum.n_input),
		NumberOutput: uint32(cIONum.n_output),
	}

	return ioNum, nil
}

// IONumber represents the C.rknn_input_output_num struct
type IONumber struct {
	NumberInput  uint32
	NumberOutput uint32
}
