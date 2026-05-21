package postprocess

import (
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess/result"
	"github.com/swdee/go-rknnlite/preprocess"
)

// YOLO26 defines the struct for YOLO26 model inference post processing
type YOLO26 struct {
	// Params are the Model configuration parameters
	Params YOLO26Params
	// nextID is a counter that increments and provides the next number
	// for each detection result ID
	idGen *result.IDGenerator
}

// YOLO26Params defines the struct containing the YOLO26 parameters to use
// for post processing operations
type YOLO26Params struct {
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
}

// YOLO26DefaultParams returns an instance of YOLO26Params configured with
// default values for a Model trained on the COCO dataset featuring:
// - Object Classes: 80
// - Box Threshold: 0.25
// - NMS Threshold: 0.45
// - Maximum Object Number: 64
func YOLO26COCOParams() YOLO26Params {
	return YOLO26Params{
		BoxThreshold:    0.25,
		NMSThreshold:    0.45,
		ObjectClassNum:  80,
		MaxObjectNumber: 64,
	}
}

// NewYOLO26 returns an instance of the YOLO26 post processor
func NewYOLO26(p YOLO26Params) *YOLO26 {
	return &YOLO26{
		Params: p,
		idGen:  result.NewIDGenerator(),
	}
}

// YOLO26Result defines a struct used for object detection results
type YOLO26Result struct {
	DetectResults []result.DetectResult
}

// GetDetectResults returns the object detection results containing bounding
// boxes
func (r YOLO26Result) GetDetectResults() []result.DetectResult {
	return r.DetectResults
}

// DetectObjects takes the RKNN outputs and runs the object detection process
// then returns the results
func (y *YOLO26) DetectObjects(outputs *rknnlite.Outputs,
	resizer *preprocess.Resizer) result.DetectionResult {

	data := newStrideData(outputs)

	if outputs.OutputAttributes().IONumber != 6 {
		return YOLO26Result{}
	}

	validCount := 0
	outputPerBranch := 2

	for i := 0; i < 3; i++ {
		boxIdx := i * outputPerBranch
		scoreIdx := boxIdx + 1

		gridH := int(outputs.OutputAttributes().DimHeights[boxIdx])
		gridW := int(outputs.OutputAttributes().DimWidths[boxIdx])
		stride := int(data.height) / gridH

		validCount += y.processStride(
			outputs.Output[boxIdx].BufInt,
			outputs.OutputAttributes().ZPs[boxIdx],
			outputs.OutputAttributes().Scales[boxIdx],
			outputs.Output[scoreIdx].BufInt,
			outputs.OutputAttributes().ZPs[scoreIdx],
			outputs.OutputAttributes().Scales[scoreIdx],
			gridH, gridW, stride,
			data,
		)
	}

	if validCount <= 0 {
		return YOLO26Result{}
	}

	indexArray := make([]int, validCount)
	for i := 0; i < validCount; i++ {
		indexArray[i] = i
	}

	quickSortIndiceInverse(data.objProbs, 0, validCount-1, indexArray)

	classSet := make(map[int]bool)
	for _, id := range data.classID {
		classSet[id] = true
	}

	for c := range classSet {
		nms(validCount, data.filterBoxes, data.classID, indexArray, c,
			y.Params.NMSThreshold, 4)
	}

	group := make([]result.DetectResult, 0)
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

		detectResult := result.DetectResult{
			Box: result.BoxRect{
				Left:   int(clamp(x1, 0, data.width) / resizer.ScaleFactor()),
				Top:    int(clamp(y1, 0, data.height) / resizer.ScaleFactor()),
				Right:  int(clamp(x2, 0, data.width) / resizer.ScaleFactor()),
				Bottom: int(clamp(y2, 0, data.height) / resizer.ScaleFactor()),
			},
			Probability: data.objProbs[n],
			Class:       data.classID[n],
			ID:          y.idGen.GetNext(),
		}

		group = append(group, detectResult)
		lastCount++
	}

	return YOLO26Result{
		DetectResults: group,
	}
}

// processStride processes the given stride
func (y *YOLO26) processStride(boxTensor []int8, boxZP int32, boxScale float32,
	scoreTensor []int8, scoreZP int32, scoreScale float32,
	gridH int, gridW int, stride int,
	data *strideData) int {

	validCount := 0
	gridLen := gridH * gridW

	// Convert probability threshold to logit, then to INT8.
	// This lets us compare raw INT8 class scores without calling sigmoid
	// for every class at every grid cell.
	scoreThresLogit := unsigmoid(y.Params.BoxThreshold)
	scoreThresI8 := qntF32ToAffine(scoreThresLogit, scoreZP, scoreScale)

	for i := 0; i < gridH; i++ {
		for j := 0; j < gridW; j++ {

			baseOffset := i*gridW + j
			maxClassID := -1
			maxScoreI8 := scoreThresI8

			for c := 0; c < y.Params.ObjectClassNum; c++ {
				scoreOffset := baseOffset + c*gridLen
				score := scoreTensor[scoreOffset]

				if score > maxScoreI8 {
					maxScoreI8 = score
					maxClassID = c
				}
			}

			if maxClassID < 0 {
				continue
			}

			left := deqntAffineToF32(boxTensor[baseOffset+0*gridLen], boxZP, boxScale)
			top := deqntAffineToF32(boxTensor[baseOffset+1*gridLen], boxZP, boxScale)
			right := deqntAffineToF32(boxTensor[baseOffset+2*gridLen], boxZP, boxScale)
			bottom := deqntAffineToF32(boxTensor[baseOffset+3*gridLen], boxZP, boxScale)

			cx := float32(j) + 0.5
			cy := float32(i) + 0.5

			x1 := (cx - left) * float32(stride)
			y1 := (cy - top) * float32(stride)
			x2 := (cx + right) * float32(stride)
			y2 := (cy + bottom) * float32(stride)

			w := x2 - x1
			h := y2 - y1

			if w <= 0 || h <= 0 {
				continue
			}

			score := sigmoid(deqntAffineToF32(maxScoreI8, scoreZP, scoreScale))

			data.filterBoxes = append(data.filterBoxes, x1, y1, w, h)
			data.objProbs = append(data.objProbs, score)
			data.classID = append(data.classID, maxClassID)
			validCount++
		}
	}

	return validCount
}
