package rknnlite

/*
#include "rknn_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"os"
	"unsafe"
)

// CoreMask wraps C.rknn_core_mask
type CoreMask int

// rknn_core_mask values used to target which cores on the NPU the model is run
// on. The rk3588 has three cores, auto will pick an idle core to run the model
// on, whilst the others specify the specific core or combined number of cores
// to run. For multi-core modes the following ops have better acceleration: Conv,
// DepthwiseConvolution, Add, Concat, Relu, Clip, Relu6, ThresholdedRelu, Prelu,
// and LeakyRelu. Other type of ops will fallback to Core0 to continue running
const (
	NPUCoreAuto    CoreMask = C.RKNN_NPU_CORE_AUTO
	NPUCore0       CoreMask = C.RKNN_NPU_CORE_0
	NPUCore1       CoreMask = C.RKNN_NPU_CORE_1
	NPUCore2       CoreMask = C.RKNN_NPU_CORE_2
	NPUCore01      CoreMask = C.RKNN_NPU_CORE_0_1
	NPUCore012     CoreMask = C.RKNN_NPU_CORE_0_1_2
	NPUSkipSetCore CoreMask = 9999
)

var (
	// A list of Rockchip models and the NPU core masks used for each.
	// These are provided for passing to NewPool() to define which NPU
	// cores to pin the Model and Runtime too.
	RK3588 = []CoreMask{NPUCore0, NPUCore1, NPUCore2}
	RK3582 = []CoreMask{NPUCore0, NPUCore1, NPUCore2}
	RK3576 = []CoreMask{NPUCore0, NPUCore1}
	RK3568 = []CoreMask{NPUCore0}
	RK3566 = []CoreMask{NPUCore0}
	RK3562 = []CoreMask{NPUCore0}
)

// ErrorCodes
type ErrorCodes int

// error code values returned by the C API
const (
	Success                ErrorCodes = C.RKNN_SUCC
	ErrFail                ErrorCodes = C.RKNN_ERR_FAIL
	ErrTimeout             ErrorCodes = C.RKNN_ERR_TIMEOUT
	ErrDeviceUnavailable   ErrorCodes = C.RKNN_ERR_DEVICE_UNAVAILABLE
	ErrMallocFail          ErrorCodes = C.RKNN_ERR_MALLOC_FAIL
	ErrParamInvalid        ErrorCodes = C.RKNN_ERR_PARAM_INVALID
	ErrModelInvalid        ErrorCodes = C.RKNN_ERR_MODEL_INVALID
	ErrCtxInvalid          ErrorCodes = C.RKNN_ERR_CTX_INVALID
	ErrInputInvalid        ErrorCodes = C.RKNN_ERR_INPUT_INVALID
	ErrOutputInvalid       ErrorCodes = C.RKNN_ERR_OUTPUT_INVALID
	ErrDeviceMismatch      ErrorCodes = C.RKNN_ERR_DEVICE_UNMATCH
	ErrPreCompiledModel    ErrorCodes = C.RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL
	ErrOptimizationVersion ErrorCodes = C.RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION
	ErrPlatformMismatch    ErrorCodes = C.RKNN_ERR_TARGET_PLATFORM_UNMATCH
)

// String returns a readable description of the error code
func (e ErrorCodes) String() string {
	switch e {
	case Success:
		return "execution successful"
	case ErrFail:
		return "execution failed"
	case ErrTimeout:
		return "execution timed out"
	case ErrDeviceUnavailable:
		return "device is unavailable"
	case ErrMallocFail:
		return "C memory allocation failed"
	case ErrParamInvalid:
		return "parameter is invalid"
	case ErrModelInvalid:
		return "model file is invalid"
	case ErrCtxInvalid:
		return "context is invalid"
	case ErrInputInvalid:
		return "input is invalid"
	case ErrOutputInvalid:
		return "output is invalid"
	case ErrDeviceMismatch:
		return "device mismatch, please update rknn sdk and npu driver/firmware"
	case ErrPreCompiledModel:
		return "the RKNN model uses pre_compile mode, but is not compatible with current driver"
	case ErrOptimizationVersion:
		return "the RKNN model optimization level is not compatible with current driver"
	case ErrPlatformMismatch:
		return "the RKNN model target platform is not compatible with the current platform"
	default:
		return fmt.Sprintf("unknown error code %d", e)
	}
}

// Runtime defines the RKNN run time instance
type Runtime struct {
	// ctx is the C runtime context
	ctx C.rknn_context
	// ioNum caches the IONumber of Model Input/Output tensors
	ioNum IONumber
	// inputAttrs caches the Input Tensor Attributes of the Model
	inputAttrs []TensorAttr
	// inputAttrs caches the Output Tensor Attributes of the Model
	outputAttrs []TensorAttr
	// wantFloat indicates if Outputs are converted to float32 or left as int8.
	// default option is True
	wantFloat bool
	// inputTypeFloat32 indicates if we pass the input gocv.Mat's data as float32
	// to the RKNN backend
	inputTypeFloat32 bool
}

// NewRuntime returns a RKNN run time instance.  Provide the full path and
// filename of the RKNN compiled model file to run.
func NewRuntime(modelFile string, core CoreMask) (*Runtime, error) {

	r := &Runtime{
		wantFloat: true,
	}

	err := r.init(modelFile)

	if err != nil {
		return nil, err
	}

	// setCoreMask is only supported on RK3588, allow skipping for other Rockchip models
	// like RK3566
	if core != NPUSkipSetCore {
		err = r.setCoreMask(core)

		if err != nil {
			return nil, err
		}
	}

	// cache IONumber
	r.ioNum, err = r.QueryModelIONumber()

	if err != nil {
		return nil, err
	}

	// query Input tensors
	r.inputAttrs, err = r.QueryInputTensors()

	if err != nil {
		return nil, err
	}

	// query Output tensors
	r.outputAttrs, err = r.QueryOutputTensors()

	if err != nil {
		return nil, err
	}

	return r, nil
}

// init wraps C.rknn_init which initializes the RKNN context with the given
// model.  The modelFile is the full path and filename of the RKNN compiled
// model file to run.
func (r *Runtime) init(modelFile string) error {

	// check file exists in Go, before passing to C
	info, err := os.Stat(modelFile)

	if err != nil {
		return fmt.Errorf("model file does not exist at %s, error: %w",
			modelFile, err)
	}

	if info.IsDir() {
		return fmt.Errorf("model file is a directory")
	}

	// convert the Go string to a C string
	cModelFile := C.CString(modelFile)
	defer C.free(unsafe.Pointer(cModelFile))

	// call the C function.
	ret := C.rknn_init(&r.ctx, unsafe.Pointer(cModelFile), 0, 0, nil)

	if ret != C.RKNN_SUCC {
		return fmt.Errorf("C.rknn_init call failed with code %d, error: %s",
			ret, ErrorCodes(ret).String())
	}

	return nil
}

// setCoreMark wraps C.rknn_set_core_mask and specifies the NPU core configuration
// to run the model on
func (r *Runtime) setCoreMask(mask CoreMask) error {

	ret := C.rknn_set_core_mask(r.ctx, C.rknn_core_mask(mask))

	if ret != C.RKNN_SUCC {
		return fmt.Errorf("C.rknn_set_core_mask failed with code %d, error: %s",
			ret, ErrorCodes(ret).String())
	}

	return nil
}

// Close wraps C.rknn_destroy which unloads the RKNN model from the runtime and
// destroys the context releasing all C resources
func (r *Runtime) Close() error {

	ret := C.rknn_destroy(r.ctx)

	if ret != C.RKNN_SUCC {
		return fmt.Errorf("C.rknn_destroy failed with code %d, error: %s",
			ret, ErrorCodes(ret).String())
	}

	return nil
}

// SetWantFloat defines if the Model load requires Output tensors to be converted
// to float32 for post processing, or left as quantitized int8
func (r *Runtime) SetWantFloat(val bool) {
	r.wantFloat = val
}

// SetInputTypeFloat32 defines if the Model requires the Inference() function to
// pass the gocv.Mat's as float32 data to RKNN backend.  Setting this overrides
// the default behaviour to pass gocv.Mat data as Uint8
func (r *Runtime) SetInputTypeFloat32(val bool) {
	r.inputTypeFloat32 = val
}

// SDKVersion represents the C.rknn_sdk_version struct
type SDKVersion struct {
	DriverVersion string
	APIVersion    string
}

// SDKVersion returns the RKNN API and Driver versions
func (r *Runtime) SDKVersion() (SDKVersion, error) {

	// prepare the structure to receive the SDK version info
	var cSdkVer C.rknn_sdk_version

	// call the C function
	ret := C.rknn_query(
		r.ctx,
		C.RKNN_QUERY_SDK_VERSION,
		unsafe.Pointer(&cSdkVer),
		C.uint(C.sizeof_rknn_sdk_version),
	)

	if ret != C.RKNN_SUCC {
		return SDKVersion{}, fmt.Errorf("rknn_query failed with return code %d", int(ret))
	}

	// convert the C rknn_sdk_version to Go rknn_sdk_version
	version := SDKVersion{
		DriverVersion: C.GoString(&(cSdkVer.drv_version[0])),
		APIVersion:    C.GoString(&(cSdkVer.api_version[0])),
	}

	return version, nil
}

// InputAttrs returns the loaded model's input tensor attributes
func (r *Runtime) InputAttrs() []TensorAttr {
	return r.inputAttrs
}

// OutputAttrs returns the loaded model's output tensor attributes
func (r *Runtime) OutputAttrs() []TensorAttr {
	return r.outputAttrs
}
