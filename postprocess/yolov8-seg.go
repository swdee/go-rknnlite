package postprocess

import (
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/preprocess"
	"github.com/swdee/go-rknnlite/tracker"
)

// YOLOv8Seg defines the struct for YOLOv8Seg model inference post processing
type YOLOv8Seg struct {
	// Params are the Model configuration parameters
	Params YOLOv8SegParams
	// nextID is a counter that increments and provides the next number
	// for each detection result ID
	idGen *idGenerator
	// protoSize is the Prototype tensor size of the Segment Mask
	protoSize int
}

// YOLOv8SegParams defines the struct containing the YOLOv8Seg parameters to use
// for post processing operations
type YOLOv8SegParams struct {
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
	// MaxObjectNumber is the maximum number of objects detected that can be
	// returned
	MaxObjectNumber int
	// PrototypeChannel is the Prototype tensor defined in the Model used
	// for generating the Segment Mask.  This is the number of channels
	// generated
	PrototypeChannel int
	// PrototypeChannel is the Prototype tensor defined in the Model used
	// for generating the Segment Mask.  This is spatial resolution height
	PrototypeHeight int
	// PrototypeChannel is the Prototype tensor defined in the Model used
	// for generating the Segment Mask.  This is the spatial resolution weight
	PrototypeWeight int
}

// YOLOv8SegDefaultParams returns an instance of YOLOv8SegParams configured with
// default values for a Model trained on the COCO dataset featuring:
// - Object Classes: 80
// - Box Threshold: 0.25
// - NMS Threshold: 0.45
// - Maximum Object Number: 64
// - PrototypeChannel: 32
// - PrototypeHeight: 160
// - PrototypeWeight: 160
func YOLOv8SegCOCOParams() YOLOv8SegParams {
	return YOLOv8SegParams{
		BoxThreshold:     0.25,
		NMSThreshold:     0.45,
		ObjectClassNum:   80,
		MaxObjectNumber:  64,
		PrototypeChannel: 32,
		PrototypeHeight:  160,
		PrototypeWeight:  160,
	}
}

// NewYOLOv8 returns an instance of the YOLOv8Seg post processor
func NewYOLOv8Seg(p YOLOv8SegParams) *YOLOv8Seg {
	return &YOLOv8Seg{
		Params:    p,
		idGen:     NewIDGenerator(),
		protoSize: p.PrototypeChannel * p.PrototypeHeight * p.PrototypeWeight,
	}
}

// YOLOv8SegResult defines a struct used for object detection results
type YOLOv8SegResult struct {
	DetectResults []DetectResult
	SegmentData   SegmentData
}

// GetDetectResults returns the object detection results containing bounding
// boxes
func (r YOLOv8SegResult) GetDetectResults() []DetectResult {
	return r.DetectResults
}

func (r YOLOv8SegResult) GetSegmentData() SegmentData {
	return r.SegmentData
}

// DetectObjects takes the RKNN outputs and runs the object detection process
// then returns the results
func (y *YOLOv8Seg) DetectObjects(outputs *rknnlite.Outputs,
	resizer *preprocess.Resizer) DetectionResult {

	data := newStrideDataSeg(outputs, y.protoSize)

	validCount := 0

	// distribution focal loss (DFL)
	dflLen := int(outputs.OutputAttributes().DimForDFL / 4)

	// process outputs of rknn
	for i := 0; i < 13; i++ {

		// same as process_i8() in C code
		validCount += y.processStride(
			outputs,
			i,
			data,
			dflLen,
		)
	}

	if validCount <= 0 {
		// no object detected
		return nil
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
		nms(validCount, data.filterBoxes, data.classID, indexArray, c,
			y.Params.NMSThreshold, 4)
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

		for k := 0; k < y.Params.PrototypeChannel; k++ {
			data.filterSegmentsByNMS = append(data.filterSegmentsByNMS,
				data.filterSegments[n*y.Params.PrototypeChannel+k])
		}

		result := DetectResult{
			Box: BoxRect{
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

	boxesNum := len(group)
	segData := SegmentData{
		filterBoxesByNMS: make([]int, boxesNum*4),
		data:             data,
		boxesNum:         boxesNum,
	}

	for i := 0; i < boxesNum; i++ {
		// store filter boxes at their original size for segment mask calculations
		segData.filterBoxesByNMS[i*4+0] = group[i].Box.Left
		segData.filterBoxesByNMS[i*4+1] = group[i].Box.Top
		segData.filterBoxesByNMS[i*4+2] = group[i].Box.Right
		segData.filterBoxesByNMS[i*4+3] = group[i].Box.Bottom

		// resize detection boxes back to that of original image
		group[i].Box.Left = boxReverse(group[i].Box.Left, resizer.XPad(), resizer.ScaleFactor())
		group[i].Box.Top = boxReverse(group[i].Box.Top, resizer.YPad(), resizer.ScaleFactor())
		group[i].Box.Right = boxReverse(group[i].Box.Right, resizer.XPad(), resizer.ScaleFactor())
		group[i].Box.Bottom = boxReverse(group[i].Box.Bottom, resizer.YPad(), resizer.ScaleFactor())
	}

	res := YOLOv8SegResult{
		DetectResults: group,
		SegmentData:   segData,
	}

	return res
}

// processStride processes the given stride
func (y *YOLOv8Seg) processStride(outputs *rknnlite.Outputs, inputID int,
	data *strideData, dflLen int) int {

	gridH := int(outputs.OutputAttributes().DimHeights[inputID])
	gridW := int(outputs.OutputAttributes().DimWidths[inputID])
	stride := int(data.height) / gridH

	validCount := 0
	gridLen := gridH * gridW

	// skip if input id is not 0, 4, 8, or 12
	if inputID%4 != 0 {
		return validCount
	}

	if inputID == 12 {
		inputProto := outputs.Output[inputID].BufInt
		zpProto := data.outZPs[inputID]
		scaleProto := data.outScales[inputID]

		for i := 0; i < y.protoSize; i++ {
			data.proto[i] = deqntAffineToF32(inputProto[i], zpProto, scaleProto)
		}

		return validCount
	}

	boxTensor := outputs.Output[inputID].BufInt
	boxZP := data.outZPs[inputID]
	boxScale := data.outScales[inputID]

	scoreTensor := outputs.Output[inputID+1].BufInt
	scoreZP := data.outZPs[inputID+1]
	scoreScale := data.outScales[inputID+1]

	scoreSumTensor := outputs.Output[inputID+2].BufInt
	scoreSumZP := data.outZPs[inputID+2]
	scoreSumScale := data.outScales[inputID+2]

	segTensor := outputs.Output[inputID+3].BufInt
	segZP := data.outZPs[inputID+3]
	segScale := data.outScales[inputID+3]

	scoreThresI8 := qntF32ToAffine(y.Params.BoxThreshold, scoreZP, scoreScale)
	scoreSumThresI8 := qntF32ToAffine(y.Params.BoxThreshold, scoreSumZP, scoreSumScale)

	for i := 0; i < gridH; i++ {
		for j := 0; j < gridW; j++ {

			offset := i*gridW + j
			maxClassID := -1

			offsetSeg := i*gridW + j
			inPtrSeg := segTensor[offsetSeg:]

			// Quick filtering using score sum
			if scoreSumTensor != nil {
				if scoreSumTensor[offset] < scoreSumThresI8 {
					continue
				}
			}

			maxScore := int8(-scoreZP)

			for c := 0; c < y.Params.ObjectClassNum; c++ {
				if scoreTensor[offset] > scoreThresI8 && scoreTensor[offset] > maxScore {
					maxScore = scoreTensor[offset]
					maxClassID = c
				}
				offset += gridLen
			}

			// Compute box
			if maxScore > scoreThresI8 {

				for k := 0; k < y.Params.PrototypeChannel; k++ {
					segElementFP := deqntAffineToF32(inPtrSeg[k*gridLen], segZP, segScale)
					data.filterSegments = append(data.filterSegments, segElementFP)
				}

				offset = i*gridW + j
				beforeDFL := make([]float32, 4*dflLen)

				for k := 0; k < dflLen*4; k++ {
					beforeDFL[k] = deqntAffineToF32(boxTensor[offset], boxZP, boxScale)
					offset += gridLen
				}

				box := computeDFL(beforeDFL[:], dflLen)

				x1 := (-box[0] + float32(j) + 0.5) * float32(stride)
				y1 := (-box[1] + float32(i) + 0.5) * float32(stride)
				x2 := (box[2] + float32(j) + 0.5) * float32(stride)
				y2 := (box[3] + float32(i) + 0.5) * float32(stride)
				w := x2 - x1
				h := y2 - y1
				data.filterBoxes = append(data.filterBoxes, x1, y1, w, h)

				data.objProbs = append(data.objProbs, deqntAffineToF32(maxScore, scoreZP, scoreScale))
				data.classID = append(data.classID, maxClassID)
				validCount++
			}
		}
	}

	return validCount
}

// SegmentMask creates segment mask data for object detection results
func (y *YOLOv8Seg) SegmentMask(detectObjs DetectionResult,
	resizer *preprocess.Resizer) SegMask {

	// handle segment masks
	segData := detectObjs.(YOLOv8SegResult).GetSegmentData()
	boxesNum := segData.boxesNum

	// C code does not use USE_FP_RESIZE as uint8 is faster via CPU calculation
	// than using NPU

	// compute the mask through Matmul.  we have a parallel version of the code
	// which uses goroutines, but speed benefits are only gained from about
	// greater than 6 boxes. the parallel version has a negative consequence
	// in that it effects the performance of the resizeByOpenCVUint8() call
	// afterwards due to the overhead of the goroutines being cleaned up.
	//
	// also tried a version doing matmul in float32 using the ARM compute library
	// whilst it is faster on the matmul step the other Resize and Crop steps
	// are still to slow so no benefit is seen.
	var matmulOut []uint8

	if boxesNum > 6 {
		matmulOut = matmulUint8Parallel(segData.data, boxesNum,
			y.Params.PrototypeChannel, y.Params.PrototypeHeight,
			y.Params.PrototypeWeight)
	} else {
		matmulOut = matmulUint8(segData.data, boxesNum,
			y.Params.PrototypeChannel, y.Params.PrototypeHeight,
			y.Params.PrototypeWeight)
	}

	// resize the tensor mask outputs to (boxes_num, model_in_width, model_in_height)
	segMask := make([]uint8, boxesNum*int(segData.data.height*segData.data.width))

	resizeByOpenCVUint8(matmulOut, y.Params.PrototypeWeight, y.Params.PrototypeHeight,
		boxesNum, segMask, int(segData.data.width), int(segData.data.height))

	// crop mask takes all segment makes from inference and combines them into a single mask
	allMaskInOne := make([]uint8, segData.data.height*segData.data.width)
	cropMaskWithIDUint8(segMask, allMaskInOne, segData.filterBoxesByNMS, boxesNum,
		int(segData.data.height), int(segData.data.width), []int{})

	// get real mask
	croppedHeight := int(segData.data.height) - resizer.YPad()*2
	croppedWidth := int(segData.data.width) - resizer.XPad()*2

	croppedSegMask := make([]uint8, croppedHeight*croppedWidth)
	realSegMask := make([]uint8, resizer.SrcHeight()*resizer.SrcWidth())

	segReverse(allMaskInOne, croppedSegMask, realSegMask,
		int(segData.data.height), int(segData.data.width), croppedHeight, croppedWidth,
		resizer.SrcHeight(), resizer.SrcWidth(), resizer.YPad(), resizer.XPad(),
	)

	return SegMask{realSegMask}
}

// TrackMask creates segment mask data for tracked objects
func (y *YOLOv8Seg) TrackMask(detectObjs DetectionResult,
	trackObjs []*tracker.STrack, resizer *preprocess.Resizer) SegMask {

	// handle segment masks
	detectResults := detectObjs.(YOLOv8SegResult).GetDetectResults()
	segData := detectObjs.(YOLOv8SegResult).GetSegmentData()
	boxesNum := segData.boxesNum

	// the detection objects and tracked objects can be different, so we need
	// to adjust the segment mask to only have tracked object masks and strip
	// out the non-used ones
	trackObjIDs := make([]int64, 0)

	for _, trackObj := range trackObjs {
		trackObjIDs = append(trackObjIDs, trackObj.GetDetectionID())
	}

	// go through the detection results to find the object ID's we need to strip out
	stripObjs := make([]int, 0)

	for i, detResult := range detectResults {
		if !int64InSlice(detResult.ID, trackObjIDs) {
			stripObjs = append(stripObjs, i)
		}
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
		matmulOut = matmulUint8Parallel(segData.data, boxesNum,
			y.Params.PrototypeChannel, y.Params.PrototypeHeight,
			y.Params.PrototypeWeight)
	} else {
		matmulOut = matmulUint8(segData.data, boxesNum,
			y.Params.PrototypeChannel, y.Params.PrototypeHeight,
			y.Params.PrototypeWeight)
	}

	// resize the tensor mask outputs to (boxes_num, model_in_width, model_in_height)
	segMask := make([]uint8, boxesNum*int(segData.data.height*segData.data.width))

	resizeByOpenCVUint8(matmulOut, y.Params.PrototypeWeight, y.Params.PrototypeHeight,
		boxesNum, segMask, int(segData.data.width), int(segData.data.height))

	// crop mask takes all segment makes from inference and combines them into a single mask
	allMaskInOne := make([]uint8, segData.data.height*segData.data.width)
	cropMaskWithIDUint8(segMask, allMaskInOne, segData.filterBoxesByNMS, boxesNum,
		int(segData.data.height), int(segData.data.width), stripObjs)

	// get real mask
	croppedHeight := int(segData.data.height) - resizer.YPad()*2
	croppedWidth := int(segData.data.width) - resizer.XPad()*2

	croppedSegMask := make([]uint8, croppedHeight*croppedWidth)
	realSegMask := make([]uint8, resizer.SrcHeight()*resizer.SrcWidth())

	segReverse(allMaskInOne, croppedSegMask, realSegMask,
		int(segData.data.height), int(segData.data.width), croppedHeight, croppedWidth,
		resizer.SrcHeight(), resizer.SrcWidth(), resizer.YPad(), resizer.XPad(),
	)

	return SegMask{realSegMask}
}
