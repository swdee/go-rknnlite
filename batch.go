package rknnlite

import (
	"fmt"
	"gocv.io/x/gocv"
)

// Batch defines a struct used for concatenating a batch of gocv.Mat's
// together into a single gocv.Mat for use with image batching on
// a Model
type Batch struct {
	mat gocv.Mat
	// size of the batch
	size int
	// width is the input tensor size width
	width int
	// height is the input tensor size height
	height int
	// channels is the input tensor number of channels
	channels int
	// inputTypeFloat32 sets the runtime.inputTypeFloat32 value
	inputTypeFloat32 bool
	// matType is the Mat type images must be passed as
	matType gocv.MatType
	// matCnt is a counter for how many Mats have been added with Add()
	matCnt int
	// imgSize stores an images size made up from its elements
	imgSize int
}

// NewBatch creates a batch of concatenated Mats for the given input tensor
// and batch size
func NewBatch(batchSize, height, width, channels int, inputTypeFloat32 bool) *Batch {

	// Choose output Mat type
	var matType gocv.MatType

	if inputTypeFloat32 {
		matType = gocv.MatTypeCV32F
	} else {
		matType = gocv.MatTypeCV8U
	}

	shape := []int{batchSize, height, width, channels}

	return &Batch{
		size:             batchSize,
		height:           height,
		width:            width,
		channels:         channels,
		mat:              gocv.NewMatWithSizes(shape, matType),
		inputTypeFloat32: inputTypeFloat32,
		matType:          matType,
		matCnt:           0,
		imgSize:          height * width * channels,
	}
}

// Add a Mat to the batch
func (b *Batch) Add(img gocv.Mat) error {

	// check if batch is full
	if b.matCnt >= b.size {
		return fmt.Errorf("batch full")
	}

	res := b.addAt(b.matCnt, img)

	if res != nil {
		return res
	}

	// increment image counter
	b.matCnt++
	return nil
}

// AddAt adds a Mat to the batch at the specific index location
func (b *Batch) AddAt(idx int, img gocv.Mat) error {

	if idx < 0 || idx >= b.size {
		return fmt.Errorf("index %d out of range [0-%d)", idx, b.size)
	}

	return b.addAt(idx, img)
}

// addAt adds a Mat to the specified index location
func (b *Batch) addAt(idx int, img gocv.Mat) error {

	// validate mat dimensions
	if img.Rows() != b.height || img.Cols() != b.width ||
		img.Channels() != b.channels {
		return fmt.Errorf("image does not match batch shape")
	}

	if !img.IsContinuous() {
		img = img.Clone()
	}

	if b.inputTypeFloat32 {
		// pointer of the batch mat
		dstAll, err := b.mat.DataPtrFloat32()

		if err != nil {
			return fmt.Errorf("error accessing float32 batch memory: %w", err)
		}

		src, err := img.DataPtrFloat32()

		if err != nil {
			return fmt.Errorf("error getting float32 data from image: %w", err)
		}

		offset := idx * b.imgSize
		copy(dstAll[offset:], src)

	} else {
		// pointer of the batch mat
		dstAll, err := b.mat.DataPtrUint8()

		if err != nil {
			return fmt.Errorf("error accessing uint8 batch memory: %w", err)
		}

		src, err := img.DataPtrUint8()

		if err != nil {
			return fmt.Errorf("error getting uint8 data from image: %w", err)
		}

		offset := idx * b.imgSize
		copy(dstAll[offset:], src)
	}

	return nil
}

// GetOutputInt returns the tensor output for the specified image number
// as an int8 output. idx starts counting from 1 to (batchsize-1)
func (b *Batch) GetOutputInt(idx int, outputs Output, size int) ([]int8, error) {

	if idx < 0 || idx >= b.size {
		return nil, fmt.Errorf("index %d out of range [0-%d)", idx, b.size)
	}

	offset := idx * size

	if offset+size > int(outputs.Size) {
		return nil, fmt.Errorf("offset %d out of range [%d,%d)", offset, outputs.Size, offset+size)
	}

	return outputs.BufInt[offset : offset+size], nil
}

// GetOutputF32 returns the tensor output for the specified image number
// as an float32 output.  idx starts counting from 0 to (batchsize-1)
func (b *Batch) GetOutputF32(idx int, outputs Output, size int) ([]float32, error) {

	if idx < 0 || idx >= b.size {
		return nil, fmt.Errorf("index %d out of range [0-%d)", idx, b.size)
	}

	offset := idx * size

	if offset+size > int(outputs.Size) {
		return nil, fmt.Errorf("offset %d out of range [%d,%d)", offset, outputs.Size, offset+size)
	}

	return outputs.BufFloat[offset : offset+size], nil
}

// Mat returns the concatenated mat
func (b *Batch) Mat() gocv.Mat {
	return b.mat
}

// Clear the batch so it can be reused again
func (b *Batch) Clear() {
	// just reset the counter, we don't need to clear the underlying b.mat
	// as it will be overwritten with Add() is called with new images
	b.matCnt = 0
}

// Close the batch and free allocated memory
func (b *Batch) Close() error {
	return b.mat.Close()
}
