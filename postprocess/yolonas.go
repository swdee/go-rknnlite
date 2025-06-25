package postprocess

import (
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess/result"
	"github.com/swdee/go-rknnlite/preprocess"
	"sort"
)

// YOLONAS defines the struct for YOLONAS model inference post processing
type YOLONAS struct {
	// Params are the Model configuration paramters
	Params YOLONASParams
	// idGen is the counter that increments and provides the next number
	// for each detection result ID
	idGen *result.IDGenerator
}

// YOLONASParams defines the struct containing the YOLONASParams parameters to use
// for post processing operations
type YOLONASParams struct {
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

// YOLONASDefaultParams returns an instance of YOLONASParams configured with
// default values for a Model trained on the COCO dataset featuring:
// - Object Classes: 80
// - Box Threshold: 0.25
// - NMS Threshold: 0.45
// - Maximum Object Number: 64
func YOLONASCOCOParams() YOLONASParams {
	return YOLONASParams{
		BoxThreshold:    0.25,
		NMSThreshold:    0.45,
		ObjectClassNum:  80,
		MaxObjectNumber: 64,
	}
}

// NewYOLONAS returns an instance of the YOLONAS post processor
func NewYOLONAS(p YOLONASParams) *YOLONAS {
	return &YOLONAS{
		Params: p,
		idGen:  result.NewIDGenerator(),
	}
}

// YOLONASResult defines a struct used for object detection results
type YOLONASResult struct {
	DetectResults []result.DetectResult
}

// GetDetectResults returns the object detection results containing bounding
// boxes
func (r YOLONASResult) GetDetectResults() []result.DetectResult {
	return r.DetectResults
}

// DetectObjects takes the RKNN outputs and runs the object detection process
// then returns the results
func (y *YOLONAS) DetectObjects(outputs *rknnlite.Outputs,
	resizer *preprocess.Resizer) result.DetectionResult {

	boxTensor := outputs.Output[0].BufInt
	scoreTensor := outputs.Output[1].BufInt

	dflLen := int(outputs.OutputAttributes().DimForDFL)
	numClasses := y.Params.ObjectClassNum

	boxZP := outputs.OutputAttributes().ZPs[0]
	boxScale := outputs.OutputAttributes().Scales[0]
	scoreZP := outputs.OutputAttributes().ZPs[1]
	scoreScale := outputs.OutputAttributes().Scales[1]

	inWidth := outputs.InputAttributes().Width
	inHeight := outputs.InputAttributes().Height

	// collate objects into a result for returning
	group := make([]result.DetectResult, 0)
	lastCount := 0

	// dequantize and decode each anchor
	for a := 0; a < dflLen; a++ {

		// check for maximum object number limit to restrict results to
		if lastCount >= y.Params.MaxObjectNumber {
			break
		}

		// dequantize raw box outputs, corner-based decode (x1,y1,x2,y2)
		x1p := deqntAffineToF32(boxTensor[a*4+0], boxZP, boxScale)
		y1p := deqntAffineToF32(boxTensor[a*4+1], boxZP, boxScale)
		x2p := deqntAffineToF32(boxTensor[a*4+2], boxZP, boxScale)
		y2p := deqntAffineToF32(boxTensor[a*4+3], boxZP, boxScale)

		// undo letterbox
		x1 := (x1p - float32(resizer.XPad())) / resizer.ScaleFactor()
		y1 := (y1p - float32(resizer.YPad())) / resizer.ScaleFactor()
		x2 := (x2p - float32(resizer.XPad())) / resizer.ScaleFactor()
		y2 := (y2p - float32(resizer.YPad())) / resizer.ScaleFactor()

		// find the class with highest probability (outputs are already quantized [0..1])
		bestProb := float32(0)
		bestClass := 0

		for c := 0; c < numClasses; c++ {
			// direct dequantitize a probability in [0..1]
			p := deqntAffineToF32(scoreTensor[a*numClasses+c], scoreZP, scoreScale)

			if p > bestProb {
				bestProb = p
				bestClass = c
			}
		}

		// threshold
		if bestProb < y.Params.BoxThreshold {
			continue
		}

		result := result.DetectResult{
			Box: result.BoxRect{
				Left:   int(clamp(x1, 0, inWidth)),
				Top:    int(clamp(y1, 0, inHeight)),
				Right:  int(clamp(x2, 0, inWidth)),
				Bottom: int(clamp(y2, 0, inHeight)),
			},
			Probability: bestProb,
			Class:       bestClass,
			ID:          y.idGen.GetNext(),
		}

		group = append(group, result)
		lastCount++
	}

	return YOLONASResult{
		DetectResults: y.nms(group, y.Params.NMSThreshold),
	}
}

// nms applies non maximum supression to detection results
func (y *YOLONAS) nms(dets []result.DetectResult, iouThresh float32) []result.DetectResult {

	// sort descending by Probability
	sort.Slice(dets, func(i, j int) bool {
		return dets[i].Probability > dets[j].Probability
	})

	keep := make([]result.DetectResult, 0, len(dets))
	used := make([]bool, len(dets))

	for i := range dets {
		if used[i] {
			continue
		}

		keep = append(keep, dets[i])

		for j := i + 1; j < len(dets); j++ {
			if used[j] {
				continue
			}

			xmin0 := float32(dets[i].Box.Left)
			ymin0 := float32(dets[i].Box.Top)
			xmax0 := float32(dets[i].Box.Right)
			ymax0 := float32(dets[i].Box.Bottom)

			xmin1 := float32(dets[j].Box.Left)
			ymin1 := float32(dets[j].Box.Top)
			xmax1 := float32(dets[j].Box.Right)
			ymax1 := float32(dets[j].Box.Bottom)

			iou := calculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1)

			if iou > iouThresh {
				used[j] = true
			}
		}
	}

	return keep
}
