package postprocess

import (
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess/result"
	"github.com/swdee/go-rknnlite/preprocess"
	"math"
)

// RetinaFace defines the struct for the RetinaFace model inference post processing
type RetinaFace struct {
	// Params are the Model configuration parameters
	Params RetinaFaceParams
	// nextID is a counter that increments and provides the next number
	// for each detection result ID
	idGen *idGenerator
}

// RetinaFaceParams defines the struct containing the RetinaFace parameters to use
// for post processing operations
type RetinaFaceParams struct {
	// ConfThreshold is the minimum probability score required for a bounding box
	// region to be considered for processing
	ConfThreshold float32
	// NMSThreshold is the Non-Maximum Suppression threshold used for defining
	// the maximum allowed Intersection Over Union (IoU) between two
	// bounding boxes for both to be kept
	NMSThreshold float32
	// VisThreshold is the Visualisation threshold
	VisThreshold float32
	// MaxObjectNumber is the maximum number of objects detected that can be
	// returned
	MaxObjectNumber int
	// KeyPointsNumber is the number of face landmark keypoints representing
	// different features of the face
	KeyPointsNumber int
}

// WiderFaceParams returns an instance of RetinaFaceParams configured with
// the default values for a Model trained on the WIDERFACE dataset featuring:
// - NMS Threshold: 0.4
// - ConfThreshold: 0.5
// - VisThreshold: 0.4
// - MaxObjectNumber: 128
// - KeyPointsNumber: 5
func WiderFaceParams() RetinaFaceParams {
	return RetinaFaceParams{
		NMSThreshold:    0.4,
		ConfThreshold:   0.5,
		VisThreshold:    0.4,
		MaxObjectNumber: 128,
		KeyPointsNumber: 5,
	}
}

// NewRetinaFace returns an instance of the RetinaFace post processor
func NewRetinaFace(p RetinaFaceParams) *RetinaFace {
	return &RetinaFace{
		Params: p,
		idGen:  NewIDGenerator(),
	}
}

// RetinaFaceResult defines a struct used for retina face detection results
type RetinaFaceResult struct {
	DetectResults []result.DetectResult
	KeyPoints     [][]result.KeyPoint
}

// GetDetectResults returns the object detection results containing bounding
// boxes
func (r RetinaFaceResult) GetDetectResults() []result.DetectResult {
	return r.DetectResults
}

// GetKeyPoints returns the keypoints of the detected faces landmark features
func (r RetinaFaceResult) GetKeyPoints() [][]result.KeyPoint {
	return r.KeyPoints
}

// DetectFaces takes the RKNN outputs and runs the face detection process
// then returns the result
func (r *RetinaFace) DetectFaces(outputs *rknnlite.Outputs,
	resizer *preprocess.Resizer) result.DetectionResult {

	location := outputs.Output[0].BufFloat
	scores := outputs.Output[1].BufFloat
	landms := outputs.Output[2].BufFloat

	modelWidth := outputs.InputAttributes().Width
	modelHeight := outputs.InputAttributes().Height

	var priorPtr [][4]float32
	var numPriors int

	// Determine priors based on model height
	switch modelHeight {
	case 320:
		numPriors = 4200
		priorPtr = retinaFaceBoxPriors320
	case 640:
		numPriors = 16800
		priorPtr = retinaFaceBoxPriors640
	default:
		// model shape error
		return nil
	}

	filterIndices := make([]int, numPriors)
	props := make([]float32, numPriors)

	// Filter valid results and apply NMS
	validCount := r.filterValidResult(scores, location, landms, priorPtr,
		filterIndices, props, r.Params.ConfThreshold, numPriors)

	quickSortIndiceInverse(props, 0, validCount-1, filterIndices)

	r.nms(validCount, location, filterIndices, r.Params.NMSThreshold,
		resizer.SrcWidth(), resizer.SrcHeight())

	// collate objects into a result for returning
	group := make([]result.DetectResult, 0)
	allKeyPoints := make([][]result.KeyPoint, 0)
	lastCount := 0

	for i := 0; i < validCount; i++ {
		if lastCount >= r.Params.MaxObjectNumber {
			break
		}

		if filterIndices[i] == -1 || props[i] < r.Params.VisThreshold {
			continue
		}

		n := filterIndices[i]

		x1 := location[n*4+0]*float32(modelWidth) - float32(resizer.XPad())
		y1 := location[n*4+1]*float32(modelHeight) - float32(resizer.YPad())
		x2 := location[n*4+2]*float32(modelWidth) - float32(resizer.XPad())
		y2 := location[n*4+3]*float32(modelHeight) - float32(resizer.YPad())

		keyPtData := make([]result.KeyPoint, 0)

		// Process facial landmark points
		for j := 0; j < r.Params.KeyPointsNumber; j++ {
			pointX := landms[n*10+2*j]*float32(modelWidth) - float32(resizer.XPad())
			pointY := landms[n*10+2*j+1]*float32(modelHeight) - float32(resizer.YPad())

			kp := result.KeyPoint{
				X: int(clamp(pointX, 0, modelWidth) / resizer.ScaleFactor()),
				Y: int(clamp(pointY, 0, modelHeight) / resizer.ScaleFactor()),
			}

			keyPtData = append(keyPtData, kp)
		}

		allKeyPoints = append(allKeyPoints, keyPtData)

		result := result.DetectResult{
			Box: result.BoxRect{
				Left:   int(clamp(x1, 0, modelWidth) / resizer.ScaleFactor()),
				Top:    int(clamp(y1, 0, modelHeight) / resizer.ScaleFactor()),
				Right:  int(clamp(x2, 0, modelWidth) / resizer.ScaleFactor()),
				Bottom: int(clamp(y2, 0, modelHeight) / resizer.ScaleFactor()),
			},
			Probability: props[i],
			ID:          r.idGen.GetNext(),
		}

		group = append(group, result)
		lastCount++
	}

	return RetinaFaceResult{
		DetectResults: group,
		KeyPoints:     allKeyPoints,
	}
}

// filterValidResult filters valid results based on a threshold and decodes
// the bounding boxes and landmarks
func (r *RetinaFace) filterValidResult(scores, loc, landms []float32,
	boxPriors [][4]float32, filterIndice []int,
	props []float32, threshold float32, numResults int) int {

	validCount := 0
	variances := [2]float32{0.1, 0.2}

	// Iterate through each result
	for i := 0; i < numResults; i++ {

		faceScore := scores[i*2+1]

		if faceScore > threshold {

			filterIndice[validCount] = i
			props[validCount] = faceScore

			// Decode location to original position
			xCenter := loc[i*4+0]*variances[0]*boxPriors[i][2] + boxPriors[i][0]
			yCenter := loc[i*4+1]*variances[0]*boxPriors[i][3] + boxPriors[i][1]
			w := float32(math.Exp(float64(loc[i*4+2]*variances[1]))) * boxPriors[i][2]
			h := float32(math.Exp(float64(loc[i*4+3]*variances[1]))) * boxPriors[i][3]

			xMin := xCenter - w*0.5
			yMin := yCenter - h*0.5
			xMax := xMin + w
			yMax := yMin + h

			loc[i*4+0] = xMin
			loc[i*4+1] = yMin
			loc[i*4+2] = xMax
			loc[i*4+3] = yMax

			// Decode landmarks
			for j := 0; j < 5; j++ {
				landms[i*10+2*j] = landms[i*10+2*j]*variances[0]*boxPriors[i][2] + boxPriors[i][0]
				landms[i*10+2*j+1] = landms[i*10+2*j+1]*variances[0]*boxPriors[i][3] + boxPriors[i][1]
			}

			validCount++
		}
	}

	return validCount
}

// nms implements a Non-Maximum Suppression (NMS) algorithm to filter
// overlapping bounding boxes
func (r *RetinaFace) nms(validCount int, outputLocations []float32, order []int,
	threshold float32, width, height int) {

	for i := 0; i < validCount; i++ {

		if order[i] == -1 {
			continue
		}

		n := order[i]

		for j := i + 1; j < validCount; j++ {

			m := order[j]

			if m == -1 {
				continue
			}

			// Calculate coordinates for bounding box n
			xmin0 := outputLocations[n*4+0] * float32(width)
			ymin0 := outputLocations[n*4+1] * float32(height)
			xmax0 := outputLocations[n*4+2] * float32(width)
			ymax0 := outputLocations[n*4+3] * float32(height)

			// Calculate coordinates for bounding box m
			xmin1 := outputLocations[m*4+0] * float32(width)
			ymin1 := outputLocations[m*4+1] * float32(height)
			xmax1 := outputLocations[m*4+2] * float32(width)
			ymax1 := outputLocations[m*4+3] * float32(height)

			// Calculate the IoU (Intersection over Union)
			iou := calculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1)

			// Suppress bounding box m if IoU exceeds the threshold
			if iou > threshold {
				order[j] = -1
			}
		}
	}
}

// GetFaceLandmarks returns the landmark keypoints for the detected faces
func (r *RetinaFace) GetFaceLandmarks(detectObjs result.DetectionResult) [][]result.KeyPoint {
	return detectObjs.(RetinaFaceResult).GetKeyPoints()
}
