package postprocess

import (
	"math"
)

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

// clamp restricts the value x to be within the range min and max and converts
// the result to float32
func clamp(val float32, min, max uint32) float32 {

	if val > float32(min) {

		if val < float32(max) {
			return val // casting the float to int after the comparison
		}

		return float32(max)
	}

	return float32(min)
}

// quickSortIndiceInverse is a quick sort algorithm that sorts the objProbs
// vector and synchronously updates the indices vector to track the reordering
// of elements
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

// nms implements a Non-Maximum Suppression (NMS) algorithm
func nms(validCount int, outputLocations []float32, classIds, order []int,
	filterId int, threshold float32) {

	for i := 0; i < validCount; i++ {

		if order[i] == -1 || classIds[i] != filterId {
			continue
		}

		n := order[i]

		for j := i + 1; j < validCount; j++ {
			m := order[j]

			if m == -1 || classIds[i] != filterId {
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

			if iou > threshold {
				order[j] = -1
			}
		}
	}
}

// calculateOverlap works out the Intersection of Union (IoU) value of two
// boxes dimensions
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

// computeDFL calculates the Distribution Focal Loss (DFL)
func computeDFL(tensor []float32, dflLen int) []float32 {

	box := make([]float32, 4)

	for b := 0; b < 4; b++ {

		expT := make([]float32, dflLen)
		expSum := float32(0)
		accSum := float32(0)

		for i := 0; i < dflLen; i++ {
			expT[i] = float32(math.Exp(float64(tensor[i+b*dflLen])))
			expSum += expT[i]
		}

		for i := 0; i < dflLen; i++ {
			accSum += expT[i] / expSum * float32(i)
		}

		box[b] = accSum
	}

	return box
}
