package rknnlite

/*
#include "rknn_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"strings"
	"unsafe"
)

// TensorFormat wraps C.rknn_tensor_format
type TensorFormat int

const (
	TensorNCHW      TensorFormat = C.RKNN_TENSOR_NCHW
	TensorNHWC      TensorFormat = C.RKNN_TENSOR_NHWC
	TensorNC1HWC2   TensorFormat = C.RKNN_TENSOR_NC1HWC2
	TensorUndefined TensorFormat = C.RKNN_TENSOR_UNDEFINED
)

// TensorType wraps C.rknn_tensor_type
type TensorType int

const (
	TensorFloat32 TensorType = C.RKNN_TENSOR_FLOAT32
	TensorFloat16 TensorType = C.RKNN_TENSOR_FLOAT16
	TensorInt8    TensorType = C.RKNN_TENSOR_INT8
	TensorUint8   TensorType = C.RKNN_TENSOR_UINT8
	TensorInt16   TensorType = C.RKNN_TENSOR_INT16
	TensorUint16  TensorType = C.RKNN_TENSOR_UINT16
	TensorInt32   TensorType = C.RKNN_TENSOR_INT32
	TensorUint32  TensorType = C.RKNN_TENSOR_UINT32
	TensorInt64   TensorType = C.RKNN_TENSOR_INT64
	TensorBool    TensorType = C.RKNN_TENSOR_BOOL
	TensorInt4    TensorType = C.RKNN_TENSOR_INT4
)

// TensorQntType wraps C.rknn_tensor_qnt_type
type TensorQntType int

const (
	TensorQntNone   TensorQntType = C.RKNN_TENSOR_QNT_NONE
	TensorQntDFP    TensorQntType = C.RKNN_TENSOR_QNT_DFP
	TensorQntAffine TensorQntType = C.RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
)

// AttrMaxDimensions are the maximum dimensions for an attribute in a tensor
type AttrMaxDimensions int

// maximum field lengths of attributes in a tensor
const (
	AttrMaxDimension  AttrMaxDimensions = C.RKNN_MAX_DIMS
	AttrMaxChannels   AttrMaxDimensions = C.RKNN_MAX_NUM_CHANNEL
	AttrMaxNameLength AttrMaxDimensions = C.RKNN_MAX_NAME_LEN
	AttrMaxDynShape   AttrMaxDimensions = C.RKNN_MAX_DYNAMIC_SHAPE_NUM
)

// TensorAttr represents the C.rknn_tensor_attr structure
type TensorAttr struct {
	Index          uint32
	NDims          uint32
	Dims           [AttrMaxDimension]uint32
	Name           string
	NElems         uint32
	Size           uint32
	Fmt            TensorFormat
	Type           TensorType
	QntType        TensorQntType
	FL             int8
	ZP             int32
	Scale          float32
	WStride        uint32
	SizeWithStride uint32
	PassThrough    bool
	HStride        uint32
}

// convertTensorAttr converts a C.rknn_tensor_attr to a Go TensorAttr
func (r *Runtime) convertTensorAttr(cAttr *C.rknn_tensor_attr) TensorAttr {

	// convert C char array to Go string for Name field
	nameBytes := C.GoBytes(unsafe.Pointer(&cAttr.name[0]), C.int(AttrMaxNameLength))
	goName := string(nameBytes)

	// find the first null byte to correctly end the string (if present)
	nullIndex := strings.IndexByte(goName, 0)

	if nullIndex != -1 {
		// Trim the string at the first null character
		goName = goName[:nullIndex]
	}

	return TensorAttr{
		Index:          uint32(cAttr.index),
		NDims:          uint32(cAttr.n_dims),
		Dims:           *(*[AttrMaxDimension]uint32)(unsafe.Pointer(&cAttr.dims)),
		Name:           goName,
		NElems:         uint32(cAttr.n_elems),
		Size:           uint32(cAttr.size),
		Fmt:            TensorFormat(cAttr.fmt),
		Type:           TensorType(cAttr._type),
		QntType:        TensorQntType(cAttr.qnt_type),
		FL:             int8(cAttr.fl),
		ZP:             int32(cAttr.zp),
		Scale:          float32(cAttr.scale),
		WStride:        uint32(cAttr.w_stride),
		SizeWithStride: uint32(cAttr.size_with_stride),
		PassThrough:    cAttr.pass_through != 0,
		HStride:        uint32(cAttr.h_stride),
	}
}

// QueryInputTensors gets the model Input Tensor attributes
func (r *Runtime) QueryInputTensors() ([]TensorAttr, error) {

	// allocate memory for input attributes in C
	cInputAttrs := make([]C.rknn_tensor_attr, r.ioNum.NumberInput)

	for i := uint32(0); i < r.ioNum.NumberInput; i++ {
		cInputAttrs[i].index = C.uint32_t(i)

		ret := C.rknn_query(r.ctx, C.RKNN_QUERY_INPUT_ATTR,
			unsafe.Pointer(&cInputAttrs[i]), C.uint(unsafe.Sizeof(cInputAttrs[i])))

		if ret != C.RKNN_SUCC {
			return nil, fmt.Errorf("C.rknn_query RKNN_QUERY_INPUT_ATTR failed with code %d, error: %s",
				int(ret), ErrorCodes(ret).String())
		}
	}

	// convert the C.rknn_tensor_attr array to a TensorAttr
	inputAttrs := make([]TensorAttr, r.ioNum.NumberInput)

	for i, cAttr := range cInputAttrs {
		inputAttrs[i] = r.convertTensorAttr(&cAttr)
	}

	return inputAttrs, nil
}

// QueryOutputTensors gets the model Output Tensor attributes
func (r *Runtime) QueryOutputTensors() ([]TensorAttr, error) {

	// allocate memory for input attributes in C
	cOutputAttrs := make([]C.rknn_tensor_attr, r.ioNum.NumberOutput)

	for i := uint32(0); i < r.ioNum.NumberOutput; i++ {
		cOutputAttrs[i].index = C.uint32_t(i)

		ret := C.rknn_query(r.ctx, C.RKNN_QUERY_OUTPUT_ATTR, unsafe.Pointer(&cOutputAttrs[i]), C.uint(unsafe.Sizeof(cOutputAttrs[i])))

		if ret != C.RKNN_SUCC {
			return nil, fmt.Errorf("rknn_query RKNN_QUERY_OUTPUT_ATTR failed with code %d, error: %s",
				int(ret), ErrorCodes(ret).String())
		}
	}

	// convert the C rknn_tensor_attr array to a Go slice of RKNNTensorAttr
	outputAttrs := make([]TensorAttr, r.ioNum.NumberOutput)

	for i, cAttr := range cOutputAttrs {
		outputAttrs[i] = r.convertTensorAttr(&cAttr)
	}

	return outputAttrs, nil

}

// String returns the TensorAttr's attributes formatted as a string
func (a TensorAttr) String() string {

	return fmt.Sprintf("index=%d, name=%s, n_dims=%d, "+
		"dims=[%d, %d, %d, %d], n_elems=%d, "+
		"size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f",
		a.Index, a.Name, a.NDims, a.Dims[0], a.Dims[1], a.Dims[2], a.Dims[3],
		a.NElems, a.Size, a.Fmt.String(), a.Type.String(), a.QntType.String(), a.ZP, a.Scale,
	)
}

// String returns a readable description of the TensorType
func (t TensorType) String() string {
	switch t {
	case TensorFloat32:
		return "FP32"
	case TensorFloat16:
		return "FP16"
	case TensorInt8:
		return "INT8"
	case TensorUint8:
		return "UINT8"
	case TensorInt16:
		return "INT16"
	case TensorUint16:
		return "UINT16"
	case TensorInt32:
		return "INT32"
	case TensorUint32:
		return "UINT32"
	case TensorInt64:
		return "INT64"
	case TensorBool:
		return "BOOL"
	case TensorInt4:
		return "INT4"
	default:
		return "UNKNOW"
	}
}

// String returns a readable description of the TensorQntType
func (t TensorQntType) String() string {
	switch t {
	case TensorQntNone:
		return "NONE"
	case TensorQntDFP:
		return "DFP"
	case TensorQntAffine:
		return "AFFINE"
	default:
		return "UNKNOW"
	}
}

// String returns a readable description of the TensorFormat
func (t TensorFormat) String() string {
	switch t {
	case TensorNCHW:
		return "NCHW"
	case TensorNHWC:
		return "NHWC"
	case TensorNC1HWC2:
		return "NC1HWC2"
	case TensorUndefined:
		return "UNDEFINED"
	default:
		return "UNKNOW"
	}
}
