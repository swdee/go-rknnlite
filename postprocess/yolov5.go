package postprocess

import (
	"github.com/swdee/go-rknnlite"
)

// YOLOv5 defines the struct for YOLOv5 model inference post processing
type YOLOv5 struct {
	// Params are the Model configuration parameters
	Params YOLOv5Params
}

// YOLOv5Params defines the struct containing the YOLOv5 parameters to use
// for post processing operations
type YOLOv5Params struct {
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

type YOLOStride struct {
	// Size is the number of pixels to use for input in each grid section of
	// the image
	Size int
	// Anchor are the Anchor Box presents for the YOLO model used in bounding
	// box predicition for objects
	Anchor []int
}

// YOLOv5DefaultParams returns an instance of YOLOv5Params configured with
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
func YOLOv5COCOParams() YOLOv5Params {
	return YOLOv5Params{
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

// NewYOLOv5 returns an instance of the YOLOv5 post processor
func NewYOLOv5(p YOLOv5Params) *YOLOv5 {
	return &YOLOv5{
		Params: p,
	}
}

// strideData is a struct used to hold all Stride data used during the post
// processing
type strideData struct {
	// filterBoxes is a slice of all objects detected's bounding box parameters
	filterBoxes []float32
	// objProbs is a slice of all the object probabilities detected
	objProbs []float32
	// classID is a slice of all the object class ID's detected.  These
	// correspond to the index/line of the name in the dataset labels
	classID []int
	// outScales are the Model output Scales
	outScales []float32
	// outZPs are the Model output Zero Points
	outZPs []int32
	// height is the pixel height of input image the Model was trained on
	height uint32
	// width is the pixel width of input image the Model was trained on
	width uint32
}

// newStrideData returns an initialised instance of strideData
func newStrideData(outputs *rknnlite.Outputs) *strideData {

	in := outputs.InputAttributes()
	out := outputs.ScalesAndZPs()

	s := &strideData{
		filterBoxes: make([]float32, 0),
		objProbs:    make([]float32, 0),
		classID:     make([]int, 0),
		outScales:   out.Scales,
		outZPs:      out.ZPs,
		height:      in.Height,
		width:       in.Width,
	}

	return s
}

// BoxRect are the dimensions of the bounding box of a detect object
type BoxRect struct {
	Left   int
	Right  int
	Top    int
	Bottom int
}

// DetectResult defines the attributes of a single object detected
type DetectResult struct {
	// Class is the line number in the labels file the Model was trained on
	// defining the Class of the detected object
	Class int
	// Box are the bounding box dimensions of the object location
	Box BoxRect
	// Probability is the confidence score of the object detected
	Probability float32
}

// DetectObjects takes the RKNN outputs and runs the object detection process
// then returns the results
func (y *YOLOv5) DetectObjects(outputs *rknnlite.Outputs) []DetectResult {

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

		result := DetectResult{
			Box: BoxRect{
				Left:   int(clamp(x1, 0, data.width)),
				Top:    int(clamp(y1, 0, data.height)),
				Right:  int(clamp(x2, 0, data.width)),
				Bottom: int(clamp(y2, 0, data.height)),
			},
			Probability: objConf,
			Class:       id,
		}

		group = append(group, result)
		lastCount++
	}

	return group
}

// processStride processes the given stride
func (y *YOLOv5) processStride(input []int8, stride YOLOStride,
	data *strideData, zp int32, scale float32) int {

	// calculate grid size
	gridH := int(data.height) / stride.Size
	gridW := int(data.width) / stride.Size

	validCount := 0
	gridLen := gridH * gridW
	thresI8 := qntF32ToAffine(y.Params.BoxThreshold, zp, scale)

	for a := 0; a < 3; a++ {
		for i := 0; i < gridH; i++ {
			for j := 0; j < gridW; j++ {

				boxConfidence := input[(y.Params.ProbBoxSize*a+4)*gridLen+i*gridW+j]

				if boxConfidence >= thresI8 {

					offset := (y.Params.ProbBoxSize*a)*gridLen + i*gridW + j
					inPtr := offset // Used as a starting index into input

					boxX := (deqntAffineToF32(input[inPtr], zp, scale))*2.0 - 0.5
					boxY := (deqntAffineToF32(input[inPtr+gridLen], zp, scale))*2.0 - 0.5
					boxW := (deqntAffineToF32(input[inPtr+2*gridLen], zp, scale)) * 2.0
					boxH := (deqntAffineToF32(input[inPtr+3*gridLen], zp, scale)) * 2.0

					boxX = (boxX + float32(j)) * float32(stride.Size)
					boxY = (boxY + float32(i)) * float32(stride.Size)
					boxW = boxW * boxW * float32(stride.Anchor[a*2])
					boxH = boxH * boxH * float32(stride.Anchor[a*2+1])
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

					if maxClassProbs > thresI8 {
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
