package postprocess

import (
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/preprocess"
	"runtime"
	"sync"
)

// YOLOv5Seg defines the struct for YOLOv5Seg model inference post processing
type YOLOv5Seg struct {
	// Params are the Model configuration parameters
	Params YOLOv5SegParams
	// nextID is a counter that increments and provides the next number
	// for each detection result ID
	idGen *idGenerator
}

// YOLOv5SegParams defines the struct containing the YOLOv5Seg parameters to use
// for post processing operations
type YOLOv5SegParams struct {
	// Strides
	Strides []YOLOStride
	// BoxThreshold is the minimum probability score required for a bounding box
	// region to be considered for processing
	BoxThreshold float32
	// NMSThreshold is the Non-Maximum Suppression threshold used for defining
	// the maximum allowed Intersection Over Union (IoU) between two
	// bounding boxes for both to be kept
	NMSThreshold float32
	// ObjectClassNum is the number of different object classes the Model has
	// been trained with
	ObjectClassNum int
	// ProbBoxSize is the length of array elements representing each bounding
	// box's attributes.  Which represents the bounding box attributes plus
	// number of objects (ObjectClassNum) the Model was trained with
	ProbBoxSize int
	// MaxObjectNumber is the maximum number of objects detected that can be
	// returned
	MaxObjectNumber int
}

// SegMask defines the segment mask data that is returned with detection results
type SegMask struct {
	// Mask is the segment mask data
	Mask []uint8
	// BoxIDs is the list of object detection Box ID's, whose position in the
	// array maps to the ID assigned to that in the segment mask
	BoxIDs []int64
}

// YOLOv5SegDefaultParams returns an instance of YOLOv5SegParams configured with
// default values for a Model trained on the COCO dataset featuring:
// - Object Classes: 80
// - Anchor Boxes for each Stride of:
//   - Stride 8: (10x13), (16x30), (33x23)
//   - Stride 16: (30x61), (62x45), (59x119)
//   - Stride 32: (116x90), (156x198), (373x326)
//
// - Box Threshold: 0.25
// - NMS Threshold: 0.45
// - Prob Box Size: 85
//   - This is 80 Object Classes plus the 5 attributes used to define a bounding
//     box being:
//   - x & y coordinates for the center of the bounding box
//   - width and height of the box relative to whole image
//   - confidence score
//
// - Maximum Object Number: 64
func YOLOv5SegCOCOParams() YOLOv5SegParams {
	return YOLOv5SegParams{
		Strides: []YOLOStride{
			{
				Size:   8,
				Anchor: []int{10, 13, 16, 30, 33, 23},
			},
			{
				Size:   16,
				Anchor: []int{30, 61, 62, 45, 59, 119},
			},
			{
				Size:   32,
				Anchor: []int{116, 90, 156, 198, 373, 326},
			},
		},
		BoxThreshold:    0.25,
		NMSThreshold:    0.45,
		ObjectClassNum:  80,
		ProbBoxSize:     85,
		MaxObjectNumber: 64,
	}
}

// NewYOLOv5Seg returns an instance of the YOLOv5Seg post processor
func NewYOLOv5Seg(p YOLOv5SegParams) *YOLOv5Seg {
	return &YOLOv5Seg{
		Params: p,
		idGen:  NewIDGenerator(),
	}
}

const (
	protoChannel = 32
	protoHeight  = 160
	protoWeight  = 160
	protoSize    = protoChannel * protoHeight * protoWeight
)

// newStrideDataSeg returns an initialised instance of strideData
func newStrideDataSeg(outputs *rknnlite.Outputs) *strideData {

	in := outputs.InputAttributes()
	out := outputs.OutputAttributes()

	s := &strideData{
		filterBoxes:         make([]float32, 0),
		objProbs:            make([]float32, 0),
		classID:             make([]int, 0),
		outScales:           out.Scales,
		outZPs:              out.ZPs,
		height:              in.Height,
		width:               in.Width,
		filterSegments:      make([]float32, 0),
		filterSegmentsByNMS: make([]float32, 0),
		proto:               make([]float32, protoSize),
	}

	return s
}

// DetectObjects takes the RKNN outputs and runs the object detection process
// then returns the results
func (y *YOLOv5Seg) DetectObjects(outputs *rknnlite.Outputs,
	resizer *preprocess.Resizer) ([]DetectResult, SegMask) {

	// strides in protoype code
	data := newStrideDataSeg(outputs)

	validCount := 0

	// process outputs of rknn
	for i := 0; i < 7; i++ {

		// same as process_i8() in C code
		validCount += y.processStride(
			outputs,
			i,
			data,
		)
	}

	if validCount <= 0 {
		// no object detected
		return nil, SegMask{}
	}

	// indexArray is used to keep and index of detect objects contained in
	// the stride "data" variable
	var indexArray []int

	for i := 0; i < validCount; i++ {
		indexArray = append(indexArray, i)
	}

	quickSortIndiceInverse(data.objProbs, 0, validCount-1, indexArray)

	// create a unique set of ClassID (ie: eliminate any multiples found)
	classSet := make(map[int]bool)

	for _, id := range data.classID {
		classSet[id] = true
	}

	// for each classID in the classSet calculate the NMS
	for c := range classSet {
		nms(validCount, data.filterBoxes, data.classID, indexArray, c, y.Params.NMSThreshold)
	}

	// collate objects into a result for returning
	group := make([]DetectResult, 0)
	lastCount := 0

	for i := 0; i < validCount; i++ {
		if indexArray[i] == -1 || lastCount >= y.Params.MaxObjectNumber {
			continue
		}
		n := indexArray[i]

		x1 := data.filterBoxes[n*4+0]
		y1 := data.filterBoxes[n*4+1]
		x2 := x1 + data.filterBoxes[n*4+2]
		y2 := y1 + data.filterBoxes[n*4+3]
		id := data.classID[n]
		objConf := data.objProbs[i]

		for k := 0; k < protoChannel; k++ {
			data.filterSegmentsByNMS = append(data.filterSegmentsByNMS,
				data.filterSegments[n*protoChannel+k])
		}

		result := DetectResult{
			Box: BoxRect{
				// have left the clamps on here versus C code original
				Left:   int(clamp(x1, 0, data.width)),
				Top:    int(clamp(y1, 0, data.height)),
				Right:  int(clamp(x2, 0, data.width)),
				Bottom: int(clamp(y2, 0, data.height)),
			},
			Probability: objConf,
			Class:       id,
			ID:          y.idGen.GetNext(),
		}

		group = append(group, result)
		lastCount++
	}

	// handle segment masks
	boxesNum := len(group)
	filterBoxesByNMS := make([]int, boxesNum*4) // C code is float32
	boxIDs := make([]int64, boxesNum)

	for i := 0; i < boxesNum; i++ {
		// for crop mask
		filterBoxesByNMS[i*4+0] = group[i].Box.Left
		filterBoxesByNMS[i*4+1] = group[i].Box.Top
		filterBoxesByNMS[i*4+2] = group[i].Box.Right
		filterBoxesByNMS[i*4+3] = group[i].Box.Bottom
		boxIDs[i] = group[i].ID

		// get real box
		group[i].Box.Left = boxReverse(group[i].Box.Left, resizer.XPad(), resizer.ScaleFactor())
		group[i].Box.Top = boxReverse(group[i].Box.Top, resizer.YPad(), resizer.ScaleFactor())
		group[i].Box.Right = boxReverse(group[i].Box.Right, resizer.XPad(), resizer.ScaleFactor())
		group[i].Box.Bottom = boxReverse(group[i].Box.Bottom, resizer.YPad(), resizer.ScaleFactor())
	}

	// C code does not use USE_FP_RESIZE as uint8 is faster via CPU calculation
	// than using NPU

	// compute the mask through Matmul.  we have a parallel version of the code
	// which uses goroutines, but speed benefits are only gained from about
	// greater than 6 boxes. the parallel version has a negative consequence
	// in that it effects the performance of the resizeByOpenCVUint8() call
	// afterwards due to the overhead of the goroutines being cleaned up.
	var matmulOut []uint8
	if boxesNum > 6 {
		matmulOut = y.matmulUint8Parallel(data, boxesNum)
	} else {
		matmulOut = y.matmulUint8(data, boxesNum)
	}

	// resize to (boxes_num, model_in_width, model_in_height)
	segMask := make([]uint8, boxesNum*int(data.height*data.width))

	resizeByOpenCVUint8(matmulOut, protoWeight, protoHeight,
		boxesNum, segMask, int(data.width), int(data.height))

	// crop mask
	allMaskInOne := make([]uint8, data.height*data.width)
	cropMaskWithIDUint8(segMask, allMaskInOne, filterBoxesByNMS, boxesNum,
		int(data.height), int(data.width))

	// get real mask
	croppedHeight := int(data.height) - resizer.YPad()*2
	croppedWidth := int(data.width) - resizer.XPad()*2

	croppedSegMask := make([]uint8, croppedHeight*croppedWidth)
	realSegMask := make([]uint8, resizer.SrcHeight()*resizer.SrcWidth())

	segReverse(allMaskInOne, croppedSegMask, realSegMask,
		int(data.height), int(data.width), croppedHeight, croppedWidth,
		resizer.SrcHeight(), resizer.SrcWidth(), resizer.YPad(), resizer.XPad(),
	)

	return group, SegMask{realSegMask, boxIDs}
}

// matmulUint8 performs matrix multiplication using the CPU
func (y *YOLOv5Seg) matmulUint8(data *strideData, boxesNum int) []uint8 {

	A := data.filterSegmentsByNMS
	B := data.proto
	// C is matmulOut
	C := make([]uint8, boxesNum*protoHeight*protoWeight)

	rowsA := boxesNum
	colsA := protoChannel
	colsB := protoHeight * protoWeight

	var temp float32

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			temp = 0
			for k := 0; k < colsA; k++ {
				temp += A[i*colsA+k] * B[k*colsB+j]
			}
			if temp > 0 {
				C[i*colsB+j] = 4
			} else {
				C[i*colsB+j] = 0
			}
		}
	}

	return C
}

func (y *YOLOv5Seg) matmulUint8Parallel(data *strideData, boxesNum int) []uint8 {

	A := data.filterSegmentsByNMS
	B := data.proto
	C := make([]uint8, boxesNum*protoHeight*protoWeight)

	rowsA := boxesNum
	colsA := protoChannel
	colsB := protoHeight * protoWeight

	// use a worker pool based on available CPU cores
	numWorkers := runtime.NumCPU()
	rowCh := make(chan int, rowsA)

	// worker function for performing the matrix multiplication on a row
	worker := func() {
		for i := range rowCh {
			for j := 0; j < colsB; j++ {
				var temp float32
				for k := 0; k < colsA; k++ {
					temp += A[i*colsA+k] * B[k*colsB+j]
				}
				if temp > 0 {
					C[i*colsB+j] = 4
				} else {
					C[i*colsB+j] = 0
				}
			}
		}
	}

	// start the workers
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		go func() {
			defer wg.Done()
			worker()
		}()
	}

	// distribute rows to workers
	for i := 0; i < rowsA; i++ {
		rowCh <- i
	}
	close(rowCh)

	// wait for all workers to complete
	wg.Wait()

	return C
}

// processStride processes the given stride
func (y *YOLOv5Seg) processStride(outputs *rknnlite.Outputs, inputID int,
	data *strideData) int {

	gridH := int(outputs.OutputAttributes().DimHeights[inputID])
	gridW := int(outputs.OutputAttributes().DimWidths[inputID])
	stride := int(data.height) / gridH

	validCount := 0
	gridLen := gridH * gridW

	if inputID%2 == 1 {
		return validCount
	}

	if inputID == 6 {
		inputProto := outputs.Output[inputID].BufInt
		zpProto := data.outZPs[inputID]
		scaleProto := data.outScales[inputID]

		for i := 0; i < protoSize; i++ {
			data.proto[i] = deqntAffineToF32(inputProto[i], zpProto, scaleProto)
		}

		return validCount
	}

	input := outputs.Output[inputID].BufInt
	inputSeg := outputs.Output[inputID+1].BufInt
	zp := data.outZPs[inputID]
	scale := data.outScales[inputID]
	zpSeg := data.outZPs[inputID+1]
	scaleSeg := data.outScales[inputID+1]

	thresI8 := qntF32ToAffine(y.Params.BoxThreshold, zp, scale)

	for a := 0; a < 3; a++ {
		for i := 0; i < gridH; i++ {
			for j := 0; j < gridW; j++ {

				boxConfidence := input[(y.Params.ProbBoxSize*a+4)*gridLen+i*gridW+j]

				if boxConfidence >= thresI8 {

					offset := (y.Params.ProbBoxSize*a)*gridLen + i*gridW + j
					offsetSeg := (protoChannel*a)*gridLen + i*gridW + j
					inPtr := offset // Used as a starting index into input
					inPtrSeg := offsetSeg

					boxX := (deqntAffineToF32(input[inPtr], zp, scale))*2.0 - 0.5
					boxY := (deqntAffineToF32(input[inPtr+gridLen], zp, scale))*2.0 - 0.5
					boxW := (deqntAffineToF32(input[inPtr+2*gridLen], zp, scale)) * 2.0
					boxH := (deqntAffineToF32(input[inPtr+3*gridLen], zp, scale)) * 2.0

					boxX = (boxX + float32(j)) * float32(stride)
					boxY = (boxY + float32(i)) * float32(stride)
					boxW = boxW * boxW * float32(y.Params.Strides[inputID/2].Anchor[a*2])
					boxH = boxH * boxH * float32(y.Params.Strides[inputID/2].Anchor[a*2+1])
					boxX -= boxW / 2.0
					boxY -= boxH / 2.0

					maxClassProbs := input[inPtr+5*gridLen]
					maxClassID := 0

					for k := 1; k < y.Params.ObjectClassNum; k++ {
						prob := input[inPtr+(5+k)*gridLen]
						if prob > maxClassProbs {
							maxClassID = k
							maxClassProbs = prob
						}
					}

					boxConfF32 := deqntAffineToF32(boxConfidence, zp, scale)
					classProbF32 := deqntAffineToF32(maxClassProbs, zp, scale)
					limitScore := boxConfF32 * classProbF32

					if limitScore > y.Params.BoxThreshold {
						for k := 0; k < protoChannel; k++ {
							segElementFP := deqntAffineToF32(inputSeg[inPtrSeg+k*gridLen], zpSeg, scaleSeg)
							data.filterSegments = append(data.filterSegments, segElementFP)
						}

						data.objProbs = append(data.objProbs,
							deqntAffineToF32(maxClassProbs, zp, scale)*deqntAffineToF32(boxConfidence, zp, scale),
						)
						data.classID = append(data.classID, maxClassID)
						data.filterBoxes = append(data.filterBoxes, boxX, boxY, boxW, boxH)
						validCount++
					}
				}
			}
		}
	}

	return validCount
}
