package rknnlite

import (
	"fmt"
	"log"
	"math"
)

const (
	// default confidence threshold
	BOX_THRESH = 0.25
	// default NMS (Non-maximum Suppression) threshold
	NMS_THRESH = 0.45

	OBJ_CLASS_NUM = 80
	PROP_BOX_SIZE = 5 + OBJ_CLASS_NUM

	OBJ_NAME_MAX_SIZE = 16
	OBJ_NUMB_MAX_SIZE = 64
)

var (
	anchor0 = []int{10, 13, 16, 30, 33, 23}
	anchor1 = []int{30, 61, 62, 45, 59, 119}
	anchor2 = []int{116, 90, 156, 198, 373, 326}
)

type StrideData struct {
	filterBoxes []float32
	objProbs    []float32
	classID     []int
}

/*
  post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
               box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
*/

// TODO: the BOX_THRES and NMS_THRES need to be passed in as paramters
func (r *Runtime) DetectObjects(outputs []Output, scaleW, scaleH float32) DetectResultGroup {

	// load coco labels from txt file into "labels"

	// detect_result_group

	// init, filterBoxes, objProbs, ClassId
	strides := &StrideData{
		filterBoxes: make([]float32, 0),
		objProbs:    make([]float32, 0),
		classID:     make([]int, 0),
	}

	// set default vars where inputAttr is NCHW
	//channel := r.inputAttrs[0].Dims[1]
	height := r.inputAttrs[0].Dims[2]
	width := r.inputAttrs[0].Dims[3]

	if r.inputAttrs[0].Fmt == TensorNHWC {
		height = r.inputAttrs[0].Dims[1]
		width = r.inputAttrs[0].Dims[2]
		//channel = r.inputAttrs[0].Dims[3]
	}

	// init scales and zero points
	outScales := make([]float32, 0)
	outZPs := make([]int32, 0)

	for i := 0; i < int(r.ioNum.NumberOutput); i++ {
		outScales = append(outScales, r.outputAttrs[i].Scale)
		outZPs = append(outZPs, r.outputAttrs[i].ZP)
	}

	// stride 8
	stride0 := 8
	gridH0 := int(height) / stride0
	gridW0 := int(width) / stride0
	//validCount0 := 0
	validCount0 := processStride(outputs[0].BufInt, anchor0, gridH0, gridW0,
		stride0, strides, BOX_THRESH,
		outZPs[0], outScales[0])

	// stride 16
	stride1 := 16
	gridH1 := int(height) / stride1
	gridW1 := int(width) / stride1
	validCount1 := processStride(outputs[1].BufInt, anchor1, gridH1, gridW1,
		stride1, strides, BOX_THRESH,
		outZPs[1], outScales[1])

	// stride 32
	stride2 := 32
	gridH2 := int(height) / stride2
	gridW2 := int(width) / stride2
	validCount2 := processStride(outputs[2].BufInt, anchor2, gridH2, gridW2,
		stride2, strides, BOX_THRESH,
		outZPs[2], outScales[2])

	log.Printf("validcount0=%d, validCount1=%d, validCount2=%d",
		validCount0, validCount1, validCount2)

	validCount := validCount0 + validCount1 + validCount2

	if validCount <= 0 {
		// no object detected
		log.Printf("No object detected")
		return DetectResultGroup{
			Count: 0,
		}
	}

	var indexArray []int

	for i := 0; i < validCount; i++ {
		indexArray = append(indexArray, i)
	}

	fmt.Println("Probabilities BEFORE:", strides.objProbs)

	quickSortIndiceInverse(strides.objProbs, 0, validCount-1, indexArray)

	fmt.Println("Sorted probabilities:", strides.objProbs)
	fmt.Println("Updated indices:", indexArray)

	// create a unique set of ClassID (ie: eliminate any multiples found)
	classSet := make(map[int]bool)

	for _, id := range strides.classID {
		classSet[id] = true
	}

	fmt.Println("class set=", classSet)
	fmt.Println("class ID's=", strides.classID)
	fmt.Println("validCount=", validCount)
	fmt.Println("BEFORE indexArry=", indexArray)

	// for each classID in the classSet calculate the NMS
	for c := range classSet {
		nms(validCount, strides.filterBoxes, strides.classID, indexArray, c, NMS_THRESH)
	}

	fmt.Println("AFTER indexarray=", indexArray)

	var group DetectResultGroup
	var lastCount int = 0
	group.Count = 0

	fmt.Printf("validCount=%d\n", validCount)

	for i := 0; i < validCount; i++ {
		if indexArray[i] == -1 || lastCount >= OBJ_NUMB_MAX_SIZE {
			continue
		}
		n := indexArray[i]

		x1 := strides.filterBoxes[n*4+0] //- float32(pads.Left) // used with letterbox resize
		y1 := strides.filterBoxes[n*4+1] //- float32(pads.Top)
		x2 := x1 + strides.filterBoxes[n*4+2]
		y2 := y1 + strides.filterBoxes[n*4+3]
		id := strides.classID[n]
		objConf := strides.objProbs[i]

		result := DetectResult{
			Box: BoxRect{
				Left:   int(clamp(x1, 0, width) / scaleW),
				Top:    int(clamp(y1, 0, height) / scaleH),
				Right:  int(clamp(x2, 0, width) / scaleW),
				Bottom: int(clamp(y2, 0, height) / scaleH),
			},
			Prop: objConf,
			//Name: labels[id],
			Name:    fmt.Sprintf("id=%d, arr idx=%d", id, n),
			IndexID: id,
		}
		/*
			// Ensure that the string copy does not exceed the max size
			if len(result.Name) > OBJ_NAME_MAX_SIZE {
				result.Name = result.Name[:OBJ_NAME_MAX_SIZE]
			}
		*/

		group.Results = append(group.Results, result)
		lastCount++
	}

	group.Count = lastCount

	fmt.Println("group=", group)
	return group
}

// nms implements a Non-Maximum Suppression (NMS) algorithm.
// function manipulates "order" variable which is the indexArray
func nms(validCount int, outputLocations []float32, classIds, order []int,
	filterId int, threshold float32) {

	for i := 0; i < validCount; i++ {

		if order[i] == -1 || classIds[i] != filterId {
			continue
		}

		n := order[i]

		log.Printf("DOING n, arr idx=%d\n", n)

		for j := i + 1; j < validCount; j++ {
			m := order[j]

			if m == -1 || classIds[i] != filterId {
				log.Printf("  j=%d, skipping, (arr idx) m=%d, classId[%d] != filterId[%d]\n", j, m, classIds[i], filterId)
				continue
			}

			xmin0 := outputLocations[n*4+0]
			ymin0 := outputLocations[n*4+1]
			xmax0 := xmin0 + outputLocations[n*4+2]
			ymax0 := ymin0 + outputLocations[n*4+3]

			xmin1 := outputLocations[m*4+0]
			ymin1 := outputLocations[m*4+1]
			xmax1 := xmin1 + outputLocations[m*4+2]
			ymax1 := ymin1 + outputLocations[m*4+3]

			iou := calculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1)

			log.Printf("  j=%d, cmp (arr idx) m=%d, filterId=%d, iou=%.2f\n", j, m, filterId, iou)

			if iou > threshold {
				log.Printf("      dropped due to threshold\n")
				order[j] = -1
			}
		}
	}
}

func calculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
	xmax1, ymax1 float32) float32 {

	w := math.Max(0.0, math.Min(float64(xmax0), float64(xmax1))-math.Max(float64(xmin0), float64(xmin1))+1.0)
	h := math.Max(0.0, math.Min(float64(ymax0), float64(ymax1))-math.Max(float64(ymin0), float64(ymin1))+1.0)
	intersection := w * h

	// Calculate the area of both rectangles with added 1.0 for inclusive pixel calculation
	area0 := (xmax0 - xmin0 + 1) * (ymax0 - ymin0 + 1)
	area1 := (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)

	// Calculate union
	union := area0 + area1 - float32(intersection)

	if union <= 0 {
		return 0.0
	}

	// Return Intersection of Union (IoU)
	return float32(intersection) / union
}

// quick sort algorithm that sorts the objProbs vector and synchronously
// updates the indices vector to track the reordering of elements
func quickSortIndiceInverse(input []float32, left int, right int, indices []int) int {
	var key float32
	var keyIndex int
	low := left
	high := right

	if left < right {
		keyIndex = indices[left]
		key = input[left]
		for low < high {
			for low < high && input[high] <= key {
				high--
			}
			input[low] = input[high]
			indices[low] = indices[high]
			for low < high && input[low] >= key {
				low++
			}
			input[high] = input[low]
			indices[high] = indices[low]
		}
		input[low] = key
		indices[low] = keyIndex
		quickSortIndiceInverse(input, left, low-1, indices)
		quickSortIndiceInverse(input, low+1, right, indices)
	}

	return low
}

func processStride(input []int8, anchor []int, gridH int, gridW int,
	stride int, strideData *StrideData, threshold float32, zp int32, scale float32) int {

	validCount := 0
	gridLen := gridH * gridW
	thresI8 := qntF32ToAffine(threshold, zp, scale)

	for a := 0; a < 3; a++ {
		for i := 0; i < gridH; i++ {
			for j := 0; j < gridW; j++ {

				boxConfidence := input[(PROP_BOX_SIZE*a+4)*gridLen+i*gridW+j]

				if boxConfidence >= thresI8 {

					offset := (PROP_BOX_SIZE*a)*gridLen + i*gridW + j
					inPtr := offset // Used as a starting index into input

					boxX := (deqntAffineToF32(input[inPtr], zp, scale))*2.0 - 0.5
					boxY := (deqntAffineToF32(input[inPtr+gridLen], zp, scale))*2.0 - 0.5
					boxW := (deqntAffineToF32(input[inPtr+2*gridLen], zp, scale)) * 2.0
					boxH := (deqntAffineToF32(input[inPtr+3*gridLen], zp, scale)) * 2.0

					boxX = (boxX + float32(j)) * float32(stride)
					boxY = (boxY + float32(i)) * float32(stride)
					boxW = boxW * boxW * float32(anchor[a*2])
					boxH = boxH * boxH * float32(anchor[a*2+1])
					boxX -= boxW / 2.0
					boxY -= boxH / 2.0

					maxClassProbs := input[inPtr+5*gridLen]
					maxClassID := 0

					for k := 1; k < OBJ_CLASS_NUM; k++ {
						prob := input[inPtr+(5+k)*gridLen]
						if prob > maxClassProbs {
							maxClassID = k
							maxClassProbs = prob
						}
					}

					if maxClassProbs > thresI8 {
						strideData.objProbs = append(strideData.objProbs, deqntAffineToF32(maxClassProbs, zp, scale)*deqntAffineToF32(boxConfidence, zp, scale))
						strideData.classID = append(strideData.classID, maxClassID)
						validCount++
						strideData.filterBoxes = append(strideData.filterBoxes, boxX, boxY, boxW, boxH)
					}

				}
			}
		}
	}

	return validCount
}

// deqntAffineToF32 converts a quantized int8 value back to a float32 using
// the provided zero point and scale
func deqntAffineToF32(qnt int8, zp int32, scale float32) float32 {
	return (float32(qnt) - float32(zp)) * scale
}

// qntF32ToAffine converts a float32 value to an int8 using quantization
// parameters: zero point and scale
func qntF32ToAffine(f32 float32, zp int32, scale float32) int8 {

	dstVal := (f32 / scale) + float32(zp)
	res := clip(dstVal, -128, 127)

	return int8(res)
}

// clip restricts the value x to be within the range min and max and converts
// the result to int
func clip(val, min, max float32) int {

	if val <= min {
		return int(min)
	}

	if val >= max {
		return int(max)
	}

	return int(val)
}

func clamp(val float32, min, max uint32) float32 {

	if val > float32(min) {

		if val < float32(max) {
			return val // casting the float to int after the comparison
		}

		return float32(max)
	}

	return float32(min)
}

/*
type DetectResult struct {
	Name string
	// LabelIndex  int32 - use instead of Name
	Probability float32
	// Dimensions of bounding box
	Left   int
	Right  int
	Top    int
	Bottom int
}

type DetectResultGroup struct {
	ID      int
	Count   int
	Results []DetectResult
}
*/

type BoxRect struct {
	Left   int
	Right  int
	Top    int
	Bottom int
}

type DetectResult struct {
	Name    string
	IndexID int
	Box     BoxRect
	Prop    float32
}

type DetectResultGroup struct {
	ID      int
	Count   int
	Results []DetectResult
}
