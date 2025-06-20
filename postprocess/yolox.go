package postprocess

import (
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/preprocess"
	"math"
)

// YOLOX defines the struct for YOLOX model inference post processing
type YOLOX struct {
	// Params are the Model configuration parameters
	Params YOLOXParams
	// nextID is a counter that increments and provides the next number
	// for each detection result ID
	idGen *idGenerator
}

// YOLOXParams defines the struct containing the YOLOX parameters to use
// for post processing operations
type YOLOXParams struct {
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

// YOLOXCOCOParams returns and instance of YOLOXParams configured with
// default values for a Model trained on the COCO dataset featuring:
// - Object Classes: 80
// - Strides of: 8, 16, 32
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
func YOLOXCOCOParams() YOLOXParams {
	return YOLOXParams{
		Strides: []YOLOStride{
			{
				Size: 8,
			},
			{
				Size: 16,
			},
			{
				Size: 32,
			},
		},
		BoxThreshold:    0.25,
		NMSThreshold:    0.45,
		ObjectClassNum:  80,
		ProbBoxSize:     85,
		MaxObjectNumber: 64,
	}
}

// NewYOLOX returns an instance of the YOLOX post processor
func NewYOLOX(p YOLOXParams) *YOLOX {
	return &YOLOX{
		Params: p,
		idGen:  NewIDGenerator(),
	}
}

// YOLOXResult defines a struct used for object detection results
type YOLOXResult struct {
	DetectResults []DetectResult
}

// GetDetectResults returns the object detection results containing bounding
// boxes
func (r YOLOXResult) GetDetectResults() []DetectResult {
	return r.DetectResults
}

// DetectObjects takes the RKNN outputs and runs the object detection process
// then returns the results
func (y *YOLOX) DetectObjects(outputs *rknnlite.Outputs,
	resizer *preprocess.Resizer) DetectionResult {

	// strides in protoype code
	data := newStrideData(outputs)

	validCount := 0

	// process each stride
	for i, stride := range y.Params.Strides {
		validCount += y.processStride(outputs.Output[i].BufInt, stride, data,
			data.outZPs[i], data.outScales[i])
	}

	if validCount <= 0 {
		// no object detected
		return YOLOXResult{}
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

		x1 := data.filterBoxes[n*4+0] - float32(resizer.XPad())
		y1 := data.filterBoxes[n*4+1] - float32(resizer.YPad())
		x2 := x1 + data.filterBoxes[n*4+2]
		y2 := y1 + data.filterBoxes[n*4+3]
		id := data.classID[n]
		objConf := data.objProbs[i]

		result := DetectResult{
			Box: BoxRect{
				Left:   int(clamp(x1, 0, data.width) / resizer.ScaleFactor()),
				Top:    int(clamp(y1, 0, data.height) / resizer.ScaleFactor()),
				Right:  int(clamp(x2, 0, data.width) / resizer.ScaleFactor()),
				Bottom: int(clamp(y2, 0, data.height) / resizer.ScaleFactor()),
			},
			Probability: objConf,
			Class:       id,
			ID:          y.idGen.GetNext(),
		}

		group = append(group, result)
		lastCount++
	}

	return YOLOXResult{
		DetectResults: group,
	}
}

// processStride processes the given stride
func (y *YOLOX) processStride(input []int8, stride YOLOStride,
	data *strideData, zp int32, scale float32) int {

	// calculate grid size
	gridH := int(data.height) / stride.Size
	gridW := int(data.width) / stride.Size

	validCount := 0
	gridLen := gridH * gridW
	thresI8 := qntF32ToAffine(y.Params.BoxThreshold, zp, scale)

	for i := 0; i < gridH; i++ {
		for j := 0; j < gridW; j++ {

			boxConfidence := input[4*gridLen+i*gridW+j]

			if boxConfidence >= thresI8 {

				offset := i*gridW + j
				inPtr := offset // Used as a starting index into input

				maxClassProbs := input[inPtr+5*gridLen]
				maxClassID := 0

				for k := 1; k < y.Params.ObjectClassNum; k++ {
					prob := input[inPtr+(5+k)*gridLen]
					if prob > maxClassProbs {
						maxClassID = k
						maxClassProbs = prob
					}
				}

				if maxClassProbs > thresI8 {

					boxX := (deqntAffineToF32(input[inPtr], zp, scale))
					boxY := (deqntAffineToF32(input[inPtr+gridLen], zp, scale))
					boxW := (deqntAffineToF32(input[inPtr+2*gridLen], zp, scale))
					boxH := (deqntAffineToF32(input[inPtr+3*gridLen], zp, scale))

					boxX = (boxX + float32(j)) * float32(stride.Size)
					boxY = (boxY + float32(i)) * float32(stride.Size)

					boxW = float32(math.Exp(float64(boxW))) * float32(stride.Size)
					boxH = float32(math.Exp(float64(boxH))) * float32(stride.Size)
					boxX -= boxW / 2.0
					boxY -= boxH / 2.0

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

	return validCount
}
