package postprocess

import (
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess/result"
	"github.com/swdee/go-rknnlite/preprocess"
)

// YOLOv8Pose defines the struct for YOLOv8 model inference post processing
type YOLOv8Pose struct {
	// Params are the Model configuration parameters
	Params YOLOv8PoseParams
	// nextID is a counter that increments and provides the next number
	// for each detection result ID
	idGen *idGenerator
}

// YOLOv8PoseParams defines the struct containing the YOLOv8 parameters to use
// for post processing operations
type YOLOv8PoseParams struct {
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
	// KeyPointsNumber is the number of COCO keypoints representing different parts
	// of the body the pose model is trained on
	KeyPointsNumber int
}

// YOLOv8PoseDefaultParams returns an instance of YOLOv8PoseParams configured with
// default values for a Model trained on the COCO dataset featuring:
// - Object Classes: 1
// - Box Threshold: 0.5
// - NMS Threshold: 0.4
// - Maximum Object Number: 64
// - KeyPoints Number: 17
func YOLOv8PoseCOCOParams() YOLOv8PoseParams {
	return YOLOv8PoseParams{
		BoxThreshold:    0.5,
		NMSThreshold:    0.4,
		ObjectClassNum:  1,
		MaxObjectNumber: 64,
		KeyPointsNumber: 17,
	}
}

// NewYOLOv8Pose returns an instance of the YOLOv8Pose post processor
func NewYOLOv8Pose(p YOLOv8PoseParams) *YOLOv8Pose {
	return &YOLOv8Pose{
		Params: p,
		idGen:  NewIDGenerator(),
	}
}

// YOLOv8PoseResult defines a struct used for object detection results
type YOLOv8PoseResult struct {
	DetectResults []result.DetectResult
	KeyPoints     [][]result.KeyPoint
}

// GetDetectResults returns the object detection results containing bounding
// boxes
func (r YOLOv8PoseResult) GetDetectResults() []result.DetectResult {
	return r.DetectResults
}

func (r YOLOv8PoseResult) GetKeyPoints() [][]result.KeyPoint {
	return r.KeyPoints
}

// DetectObjects takes the RKNN outputs and runs the object detection process
// then returns the results
func (y *YOLOv8Pose) DetectObjects(outputs *rknnlite.Outputs,
	resizer *preprocess.Resizer) result.DetectionResult {

	data := newStrideData(outputs)

	validCount := 0
	stride := 0
	index := 0

	for i := 0; i < 3; i++ {
		gridH := int(outputs.OutputAttributes().DimHeights[i])
		gridW := int(outputs.OutputAttributes().DimWidths[i])
		stride = int(data.height) / gridH

		// same as process_i8 in C code
		validCount += y.processStride(
			outputs.Output[i].BufInt,
			outputs.OutputAttributes().ZPs[i],
			outputs.OutputAttributes().Scales[i],
			gridH, gridW, stride,
			data, index,
		)

		index += gridH * gridW
	}

	if validCount <= 0 {
		// no object detected
		return YOLOv8PoseResult{}
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
			y.Params.NMSThreshold, 5)
	}

	// collate objects into a result for returning
	group := make([]result.DetectResult, 0)
	allKeyPoints := make([][]result.KeyPoint, 0)
	lastCount := 0

	for i := 0; i < validCount; i++ {
		if indexArray[i] == -1 || lastCount >= y.Params.MaxObjectNumber {
			continue
		}
		n := indexArray[i]

		x1 := data.filterBoxes[n*5+0] - float32(resizer.XPad())
		y1 := data.filterBoxes[n*5+1] - float32(resizer.YPad())
		x2 := x1 + data.filterBoxes[n*5+2]
		y2 := y1 + data.filterBoxes[n*5+3]
		keyPointsIdx := data.filterBoxes[n*5+4]

		keyPtData := make([]result.KeyPoint, 0)

		for j := 0; j < y.Params.KeyPointsNumber; j++ {
			/*
				kpX := outputs.Output[3].BufInt[j*3*8400+0*8400+int(keyPointsIdx)]
				kpY := outputs.Output[3].BufInt[j*3*8400+1*8400+int(keyPointsIdx)]
				kpScore := outputs.Output[3].BufInt[j*3*8400+2*8400+int(keyPointsIdx)]

				kp := KeyPoint{
					X:     int(float32(int(kpX)-resizer.XPad()) / resizer.ScaleFactor()),
					Y:     int(float32(int(kpY)-resizer.YPad()) / resizer.ScaleFactor()),
					Score: float32(kpScore),
				}
			*/

			// float16
			kpX := outputs.Output[3].BufFloat[j*3*8400+0*8400+int(keyPointsIdx)]
			kpY := outputs.Output[3].BufFloat[j*3*8400+1*8400+int(keyPointsIdx)]
			kpScore := outputs.Output[3].BufFloat[j*3*8400+2*8400+int(keyPointsIdx)]

			kp := result.KeyPoint{
				X:     int(float32(int(kpX)-resizer.XPad()) / resizer.ScaleFactor()),
				Y:     int(float32(int(kpY)-resizer.YPad()) / resizer.ScaleFactor()),
				Score: float32(kpScore),
			}

			keyPtData = append(keyPtData, kp)

		}

		allKeyPoints = append(allKeyPoints, keyPtData)

		id := data.classID[n]
		objConf := data.objProbs[i]

		result := result.DetectResult{
			Box: result.BoxRect{
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

	return YOLOv8PoseResult{
		DetectResults: group,
		KeyPoints:     allKeyPoints,
	}
}

// processStride processes the given stride
func (y *YOLOv8Pose) processStride(boxTensor []int8, boxZP int32, boxScale float32,
	gridH int, gridW int, stride int, data *strideData, index int) int {

	inputLocLen := 64
	validCount := 0

	thresI8 := qntF32ToAffine(unsigmoid(y.Params.BoxThreshold), boxZP, boxScale)

	for h := 0; h < gridH; h++ {
		for w := 0; w < gridW; w++ {
			for a := 0; a < y.Params.ObjectClassNum; a++ {

				offset := (inputLocLen+a)*gridW*gridH + h*gridW + w

				if boxTensor[offset] >= thresI8 {

					boxConfF32 := sigmoid(deqntAffineToF32(boxTensor[offset], boxZP, boxScale))

					// allocate space for loc array and fill it with dequantized values
					loc := make([]float32, inputLocLen)

					for i := 0; i < inputLocLen; i++ {
						loc[i] = deqntAffineToF32(boxTensor[i*gridW*gridH+h*gridW+w], boxZP, boxScale)
					}

					// apply softmax
					for i := 0; i < inputLocLen/16; i++ {
						softmax(loc[i*16:(i+1)*16], 16)
					}

					// process xywh
					var xywh_ [4]float32
					var xywh [4]float32

					for dfl := 0; dfl < 16; dfl++ {
						xywh_[0] += loc[dfl] * float32(dfl)
						xywh_[1] += loc[1*16+dfl] * float32(dfl)
						xywh_[2] += loc[2*16+dfl] * float32(dfl)
						xywh_[3] += loc[3*16+dfl] * float32(dfl)
					}

					// adjust xywh and calculate coordinates
					xywh_[0] = (float32(w) + 0.5) - xywh_[0]
					xywh_[1] = (float32(h) + 0.5) - xywh_[1]
					xywh_[2] = (float32(w) + 0.5) + xywh_[2]
					xywh_[3] = (float32(h) + 0.5) + xywh_[3]

					xywh[0] = ((xywh_[0] + xywh_[2]) / 2) * float32(stride)
					xywh[1] = ((xywh_[1] + xywh_[3]) / 2) * float32(stride)
					xywh[2] = (xywh_[2] - xywh_[0]) * float32(stride)
					xywh[3] = (xywh_[3] - xywh_[1]) * float32(stride)

					xywh[0] = xywh[0] - xywh[2]/2
					xywh[1] = xywh[1] - xywh[3]/2

					// append box coordinates and other information
					// xywh[0] is coordinate X
					// xywh[1] is coordinate Y
					// xywh[2] is Width
					// xywh[3] is Height
					// float32(index+(h*gridW)+w) is Pose keypoint index
					data.filterBoxes = append(data.filterBoxes, xywh[0], xywh[1], xywh[2], xywh[3], float32(index+(h*gridW)+w))
					data.objProbs = append(data.objProbs, boxConfF32)
					data.classID = append(data.classID, a)

					validCount++
				}
			}
		}
	}

	return validCount
}

// GetPoseEstimation returns the keypoints for the detection objects used in
// pose estimation
func (y *YOLOv8Pose) GetPoseEstimation(detectObjs result.DetectionResult) [][]result.KeyPoint {
	return detectObjs.(YOLOv8PoseResult).GetKeyPoints()
}
