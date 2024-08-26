package rknnlite

/*
#include "rknn_api.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"fmt"
	"github.com/x448/float16"
	"gocv.io/x/gocv"
	"sync"
	"unsafe"
)

// Input represents the C.rknn_input struct and defines the Input used for
// inference
type Input struct {
	// Index is the input index
	Index uint32
	// Buf is the gocv Mat input
	Buf unsafe.Pointer
	// Size is the number of bytes of Buf
	Size uint32
	// Passthrough defines the mode, if True the buf data is passed directly to
	// the input node of the rknn model without any conversion.  If False the
	// buf data is converted into an input consistent with the model according
	// to the following type and fmt
	PassThrough bool
	// Type is the data type of Buf. This is a required parameter if Passthrough
	// is False
	Type TensorType
	// Fmt is the data format of Buf.  This is a required parameter if Passthrough
	// is False
	Fmt TensorFormat
}

// Inference runs the model inference on the given inputs
func (r *Runtime) Inference(mats []gocv.Mat) (*Outputs, error) {

	// convert the cv Mat's into RKNN inputs
	inputs := make([]Input, len(mats))

	for idx, mat := range mats {

		// make mat continuous
		if !mat.IsContinuous() {
			mat = mat.Clone()
		}

		if r.inputTypeFloat32 {
			// pass data as float32 to RKNN backend
			data, err := mat.DataPtrFloat32()

			if err != nil {
				return &Outputs{}, fmt.Errorf("error getting data pointer to Mat: %w", err)
			}

			inputs[idx] = Input{
				Index: uint32(idx),
				Type:  TensorFloat32,
				// multiply by 4 for size of float32
				Size:        uint32(mat.Cols() * mat.Rows() * mat.Channels() * 4),
				Fmt:         TensorNHWC,
				Buf:         unsafe.Pointer(&data[0]),
				PassThrough: false,
			}

		} else {
			// pass data as uint8 to RKNN backend
			data, err := mat.DataPtrUint8()

			if err != nil {
				return &Outputs{}, fmt.Errorf("error getting data pointer to Mat: %w", err)
			}

			inputs[idx] = Input{
				Index:       uint32(idx),
				Type:        TensorUint8,
				Size:        uint32(mat.Cols() * mat.Rows() * mat.Channels()),
				Fmt:         TensorNHWC,
				Buf:         unsafe.Pointer(&data[0]),
				PassThrough: false,
			}
		}
	}

	// set the Inputs
	err := r.SetInputs(inputs)

	if err != nil {
		return &Outputs{}, fmt.Errorf("error setting inputs: %w", err)
	}

	// run the model
	err = r.RunModel()

	if err != nil {
		return &Outputs{}, fmt.Errorf("error running model: %w", err)
	}

	// get Outputs
	return r.GetOutputs(r.ioNum.NumberOutput, r.wantFloat)
}

// setInputs wraps C.rknn_inputs_set
func (r *Runtime) SetInputs(inputs []Input) error {

	nInputs := C.uint32_t(len(inputs))
	// make a C array of inputs
	cInputs := make([]C.rknn_input, len(inputs))

	for i, input := range inputs {
		cInputs[i].index = C.uint32_t(input.Index)
		cInputs[i].buf = input.Buf
		cInputs[i].size = C.uint32_t(input.Size)
		cInputs[i].pass_through = C.uint8_t(0)
		if input.PassThrough {
			cInputs[i].pass_through = C.uint8_t(1)
		}
		cInputs[i]._type = C.rknn_tensor_type(input.Type)
		cInputs[i].fmt = C.rknn_tensor_format(input.Fmt)
	}

	ret := C.rknn_inputs_set(r.ctx, nInputs, &cInputs[0])

	if ret != 0 {
		return fmt.Errorf("C.rknn_inputs_set failed with code %d, error: %s",
			int(ret), ErrorCodes(ret).String())
	}

	return nil
}

// RunModel wraps C.rknn_run
func (r *Runtime) RunModel() error {

	ret := C.rknn_run(r.ctx, nil)

	if ret < 0 {
		return fmt.Errorf("C.rknn_run failed with code %d, error: %s",
			int(ret), ErrorCodes(ret).String())
	}

	return nil
}

// Output wraps C.rknn_output
type Output struct {
	WantFloat  uint8  // want transfer output data to float
	IsPrealloc uint8  // whether buf is pre-allocated
	Index      uint32 // the output index
	// the output buf cast to float32, when WantFloat = 1
	// this is a slice header that points to C memory
	BufFloat []float32
	// the output buf cast to int8, when WantFloat = 0
	// this is a slice header that points to C memory
	BufInt []int8
	Size   uint32 // the size of output buf
}

// Outputs is a struct containing Go and C output data
type Outputs struct {
	Output   []Output
	cOutputs []C.rknn_output
	// freed is a flag to indicate if the cOutputs have been released from
	// memory or not
	freed bool
	// mutex to lock access to freed variable
	sync.Mutex
	// rknn runtime instance
	rt *Runtime
}

// GetOutputs returns the Output results
func (r *Runtime) GetOutputs(nOutputs uint32, wantFloat bool) (*Outputs, error) {

	outputs := &Outputs{
		Output:   make([]Output, nOutputs),
		cOutputs: make([]C.rknn_output, nOutputs),
		rt:       r,
	}

	// set want float for all outputs
	useWantFloat := uint8(1)

	if !wantFloat {
		useWantFloat = 0
	}

	for idx := range outputs.cOutputs {
		outputs.cOutputs[idx].index = C.uint32_t(idx)
		outputs.cOutputs[idx].want_float = C.uint8_t(useWantFloat)
	}

	// call C function
	ret := C.rknn_outputs_get(r.ctx, C.uint32_t(nOutputs),
		(*C.rknn_output)(unsafe.Pointer(&outputs.cOutputs[0])), nil)

	if ret < 0 {
		return &Outputs{}, fmt.Errorf("C.rknn_outputs_get failed with code %d, error: %s",
			int(ret), ErrorCodes(ret).String())
	}

	// convert C.rknn_output array back to Go Output array
	for i, cOutput := range outputs.cOutputs {
		outputs.Output[i] = Output{
			WantFloat:  uint8(cOutput.want_float),
			IsPrealloc: uint8(cOutput.is_prealloc),
			Index:      uint32(cOutput.index),
			Size:       uint32(cOutput.size),
		}

		if outputs.Output[i].WantFloat == 1 {
			// convert buffer to []float32
			outputs.Output[i].BufFloat = (*[1 << 30]float32)(outputs.cOutputs[i].buf)[:outputs.cOutputs[i].size/4]

		} else if outputs.Output[i].WantFloat == 0 {
			// yolov8-pose has output tensors of int8 and fp16, so we need to
			// handle the fp16 specially
			if r.outputAttrs[i].Type == TensorFloat16 {
				// convert float16 buffer to []float32
				float16Buf := (*[1 << 30]uint16)(outputs.cOutputs[i].buf)[:outputs.cOutputs[i].size/2]
				outputs.Output[i].BufFloat = convertFloat16BufferToFloat32(float16Buf)

			} else {
				// convert buffer to []int8
				outputs.Output[i].BufInt = (*[1 << 30]int8)(outputs.cOutputs[i].buf)[:outputs.cOutputs[i].size]
			}
		}
	}

	return outputs, nil
}

// convertFloat16BufferToFloat32 converts a float16 buffer to float32 as Go
// has not support for FP16.
func convertFloat16BufferToFloat32(float16Buf []uint16) []float32 {
	float32Buf := make([]float32, len(float16Buf))

	for i, val := range float16Buf {
		f16 := float16.Frombits(val)
		float32Buf[i] = f16.Float32()
	}

	return float32Buf
}

// Free C memory buffer holding RKNN inference outputs
func (o *Outputs) Free() error {
	o.Lock()
	defer o.Unlock()

	if o.freed {
		// C memory already released
		return nil
	}

	o.freed = true
	return o.rt.releaseOutputs(o.cOutputs)
}

// InputAttribute of trained model input tensor
type InputAttribute struct {
	Width   uint32
	Height  uint32
	Channel uint32
}

// InputAttributes queries the Model and returns Input image dimensions
func (o *Outputs) InputAttributes() InputAttribute {

	// set default vars where inputAttr is NCHW
	channel := o.rt.inputAttrs[0].Dims[1]
	height := o.rt.inputAttrs[0].Dims[2]
	width := o.rt.inputAttrs[0].Dims[3]

	if o.rt.inputAttrs[0].Fmt == TensorNHWC {
		height = o.rt.inputAttrs[0].Dims[1]
		width = o.rt.inputAttrs[0].Dims[2]
		channel = o.rt.inputAttrs[0].Dims[3]
	}

	return InputAttribute{
		Width:   width,
		Height:  height,
		Channel: channel,
	}
}

// OutputAttribute of trained model output tensor
type OutputAttribute struct {
	DimForDFL  uint32
	Scales     []float32
	ZPs        []int32
	DimHeights []uint32
	DimWidths  []uint32
	IONumber   uint32
}

// OutputAttributes returns the Model output attribute scales and zero points
func (o *Outputs) OutputAttributes() OutputAttribute {

	data := OutputAttribute{
		DimForDFL:  o.rt.outputAttrs[0].Dims[1],
		Scales:     make([]float32, 0),
		ZPs:        make([]int32, 0),
		DimHeights: make([]uint32, 0),
		DimWidths:  make([]uint32, 0),
		IONumber:   o.rt.ioNum.NumberOutput,
	}

	for i := 0; i < int(o.rt.ioNum.NumberOutput); i++ {
		data.Scales = append(data.Scales, o.rt.outputAttrs[i].Scale)
		data.ZPs = append(data.ZPs, o.rt.outputAttrs[i].ZP)
		data.DimHeights = append(data.DimHeights, o.rt.outputAttrs[i].Dims[2])
		data.DimWidths = append(data.DimWidths, o.rt.outputAttrs[i].Dims[3])
	}

	return data
}

// releaseOutputs releases the memory allocated for the outputs by the RKNN
// toolkit directly using C rknn_output structs
func (r *Runtime) releaseOutputs(cOutputs []C.rknn_output) error {

	// directly use the C array of rknn_output obtained from getOutputs or similar.
	outputsPtr := (*C.rknn_output)(unsafe.Pointer(&cOutputs[0]))

	// call C.rknn_outputs_release with the context and the outputs pointer
	ret := C.rknn_outputs_release(r.ctx, C.uint32_t(len(cOutputs)), outputsPtr)

	if ret != 0 {
		return fmt.Errorf("C.rknn_outputs_release failed with code %d, error: %s",
			ret, ErrorCodes(ret).String())
	}

	return nil
}

type Probability struct {
	LabelIndex  int32
	Probability float32
}

// GetTop5 outputs the Top5 matches in the model, with left column as label
// index and right column the match probability.  The results are returned
// in the Probability slice in descending order from top match.
func GetTop5(outputs []Output) []Probability {

	probs := make([]Probability, 5)

	for i := 0; i < len(outputs); i++ {
		var MaxClass [5]int32
		var fMaxProb [5]float32

		GetTop(outputs[i].BufFloat, fMaxProb[:], MaxClass[:], int32(len(outputs[i].BufFloat)), 5)

		for i := 0; i < 5; i++ {
			probs[i] = Probability{
				LabelIndex:  MaxClass[i],
				Probability: fMaxProb[i],
			}
		}
	}

	return probs
}

const MAX_TOP_NUM = 20

// GetTop takes outputs and produces a top list of matches by probability
func GetTop(pfProb []float32, pfMaxProb []float32, pMaxClass []int32,
	outputCount int32, topNum int32) int {

	if topNum > MAX_TOP_NUM {
		return 0
	}

	// initialize pfMaxProb with default values, ie: 0
	for j := range pfMaxProb {
		pfMaxProb[j] = 0
	}
	// initialize pMaxClass with default values, ie: -1
	for j := range pMaxClass {
		pMaxClass[j] = -1
	}

	for j := int32(0); j < topNum; j++ {
		for i := int32(0); i < outputCount; i++ {

			// skip if the current class is already in the top list
			skip := false

			for k := 0; k < len(pMaxClass); k++ {
				if i == pMaxClass[k] {
					skip = true
					break
				}
			}

			if skip {
				continue
			}

			// if the current probability is greater than the j'th max
			// probability, update pfMaxProb and pMaxClass
			if pfProb[i] > pfMaxProb[j] && pfProb[i] > 0.000001 {
				pfMaxProb[j] = pfProb[i]
				pMaxClass[j] = i
			}
		}
	}

	return 1
}
