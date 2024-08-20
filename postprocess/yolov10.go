package postprocess

import (
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/preprocess"
)

// YOLOv10 defines the struct for YOLOv10 model inference post processing
type YOLOv10 struct {
	// Params are the Model configuration parameters
	Params YOLOv10Params
	// nextID is a counter that increments and provides the next number
	// for each detection result ID
	idGen *idGenerator
}

// YOLOv10Params defines the struct containing the YOLOv10 parameters to use
// for post processing operations
type YOLOv10Params struct {
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

// YOLOv10DefaultParams returns an instance of YOLOv10Params configured with
// default values for a Model trained on the COCO dataset featuring:
// - Object Classes: 80
// - Box Threshold: 0.25
// - NMS Threshold: 0.45
// - Maximum Object Number: 64
func YOLOv10COCOParams() YOLOv10Params {
	return YOLOv10Params{
		BoxThreshold:    0.25,
		NMSThreshold:    0.45,
		ObjectClassNum:  80,
		ProbBoxSize:     85,
		MaxObjectNumber: 64,
	}
}

// NewYOLOv10 returns an instance of the YOLOv10 post processor
func NewYOLOv10(p YOLOv10Params) *YOLOv10 {
	return &YOLOv10{
		Params: p,
		idGen:  NewIDGenerator(),
	}
}

// DetectObjects takes the RKNN outputs and runs the object detection process
// then returns the results
func (y *YOLOv10) DetectObjects(outputs *rknnlite.Outputs,
	resizer *preprocess.Resizer) []DetectResult {

	data := newStrideData(outputs)

	validCount := 0
	stride := 0

	// distribution focal loss (DFL)
	dflLen := int(outputs.OutputAttributes().DimForDFL / 4)

	outputPerBranch := int(outputs.OutputAttributes().IONumber / 3)

	for i := 0; i < 3; i++ {

		scoreSum := make([]int8, 0)
		scoreSumZp := int32(0)
		scoreSumScale := float32(1.0)

		if outputPerBranch == 3 {
			scoreSum = outputs.Output[i*outputPerBranch+2].BufInt
			scoreSumZp = outputs.OutputAttributes().ZPs[i*outputPerBranch+2]
			scoreSumScale = outputs.OutputAttributes().Scales[i*outputPerBranch+2]
		}

		boxIdx := i * outputPerBranch
		scoreIdx := i*outputPerBranch + 1

		gridH := int(outputs.OutputAttributes().DimHeights[boxIdx])
		gridW := int(outputs.OutputAttributes().DimWidths[boxIdx])

		stride = int(data.height) / gridH

		// same as process_i8 in C code
		validCount += y.processStride(
			outputs.Output[boxIdx].BufInt,
			outputs.OutputAttributes().ZPs[boxIdx],
			outputs.OutputAttributes().Scales[boxIdx],
			outputs.Output[scoreIdx].BufInt,
			outputs.OutputAttributes().ZPs[scoreIdx],
			outputs.OutputAttributes().Scales[scoreIdx],
			scoreSum, scoreSumZp, scoreSumScale,
			gridH, gridW, stride, dflLen,
			data,
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

	return group
}

// processStride processes the given stride
func (y *YOLOv10) processStride(boxTensor []int8, boxZP int32, boxScale float32,
	scoreTensor []int8, scoreZP int32, scoreScale float32,
	scoreSumTensor []int8, scoreSumZP int32, scoreSumScale float32,
	gridH int, gridW int, stride int, dflLen int,
	data *strideData) int {

	validCount := 0
	gridLen := gridH * gridW
	scoreThresI8 := qntF32ToAffine(y.Params.BoxThreshold, scoreZP, scoreScale)
	scoreSumThresI8 := qntF32ToAffine(y.Params.BoxThreshold, scoreSumZP, scoreSumScale)

	for i := 0; i < gridH; i++ {
	kLoop:
		for j := 0; j < gridW; j++ {

			offset := i*gridW + j
			maxClassID := -1

			// Quick filtering using score sum
			if scoreSumTensor != nil && len(scoreSumTensor) > 0 {
				if scoreSumTensor[offset] < scoreSumThresI8 {
					continue
				}
			}

			maxScore := int8(-scoreZP)

			for c := 0; c < y.Params.ObjectClassNum; c++ {

				scoreOffset := i*gridW + j + c*gridLen

				if scoreTensor[scoreOffset] > scoreThresI8 {
					maxScore = scoreTensor[scoreOffset]
					maxClassID = c
					beforeDFL := make([]float32, 4*dflLen)

					for k := 0; k < dflLen*4; k++ {

						// check offset for being out of bounds of boxTensor value
						// this check is not apart of the C++ code, however we found
						// sometimes the offset would be greater than key in boxTensor
						// when running inference on streaming so need to check
						// to stop panic.
						if offset > len(boxTensor) {
							break kLoop
						}

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
	}

	return validCount
}
