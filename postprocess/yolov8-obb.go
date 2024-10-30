package postprocess

import (
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/preprocess"
	"math"
	"sort"
)

// YOLOv8obb defines the struct for YOLOv8-obb model inference post processing
type YOLOv8obb struct {
	// Params are the Model configuration parameters
	Params YOLOv8obbParams
	// nextID is a counter that increments and provides the next number
	// for each detection result ID
	idGen *idGenerator
}

// YOLOv8obbParams defines the struct containing the YOLOv8-obb parameters to use
// for post processing operations
type YOLOv8obbParams struct {
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

// YOLOv8obbDOTAv1Params returns an instance of YOLOv8obbParams configured
// with default values for a Model trained on the DOTAv1 dataset
// - Object Classes: 15
// - Box Threshold: 0.5
// - NMS Threshold: 0.4
// - Maximum Object Number: 64
func YOLOv8obbDOTAv1Params() YOLOv8obbParams {
	return YOLOv8obbParams{
		BoxThreshold:    0.5,
		NMSThreshold:    0.4,
		ObjectClassNum:  15,
		MaxObjectNumber: 64,
	}
}

// NewYOLOv8obb returns an instance of the YOLOv8obb post processor
func NewYOLOv8obb(p YOLOv8obbParams) *YOLOv8obb {
	return &YOLOv8obb{
		Params: p,
		idGen:  NewIDGenerator(),
	}
}

// YOLOv8obbResult defines a struct used for object detection results
type YOLOv8obbResult struct {
	DetectResults []DetectResult
}

// GetDetectResults returns the object detection results containing bounding
// boxes
func (r YOLOv8obbResult) GetDetectResults() []DetectResult {
	return r.DetectResults
}

// DetectObjects takes the RKNN outputs and runs the object detection process
// then returns the results
func (y *YOLOv8obb) DetectObjects(outputs *rknnlite.Outputs,
	resizer *preprocess.Resizer) DetectionResult {

	data := newStrideData(outputs)

	validCount := 0
	stride := 0
	index := 0

	for i := 0; i < 3; i++ {
		boxIdx := i

		gridH := int(outputs.OutputAttributes().DimHeights[boxIdx])
		gridW := int(outputs.OutputAttributes().DimWidths[boxIdx])

		stride = int(data.height) / gridH

		// same as process_i8 in C code
		validCount += y.processStride(
			outputs.Output[boxIdx].BufInt,
			outputs.Output[3].BufInt,
			gridH, gridW, stride, data,
			outputs.OutputAttributes().ZPs[boxIdx],
			outputs.OutputAttributes().Scales[boxIdx],
			outputs.OutputAttributes().ZPs[3],
			outputs.OutputAttributes().Scales[3],
			index,
		)

		index += gridH * gridW
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
		y.nms(validCount, data.filterBoxes, data.classID, indexArray, c,
			y.Params.NMSThreshold)
	}

	// collate objects into a result for returning
	group := make([]DetectResult, 0)
	lastCount := 0

	for i := 0; i < validCount; i++ {
		if indexArray[i] == -1 || lastCount >= y.Params.MaxObjectNumber {
			continue
		}
		n := indexArray[i]

		x1 := data.filterBoxes[n*5+0] - float32(resizer.XPad())
		y1 := data.filterBoxes[n*5+1] - float32(resizer.YPad())
		w := data.filterBoxes[n*5+2]
		h := data.filterBoxes[n*5+3]
		angle := data.filterBoxes[n*5+4]
		id := data.classID[n]
		objConf := data.objProbs[i]

		result := DetectResult{
			Box: BoxRect{
				X:      int(clamp(x1, 0, data.width) / resizer.ScaleFactor()),
				Y:      int(clamp(y1, 0, data.height) / resizer.ScaleFactor()),
				Width:  int(clamp(w, 0, data.width) / resizer.ScaleFactor()),
				Height: int(clamp(h, 0, data.height) / resizer.ScaleFactor()),
				Angle:  angle,
				Mode:   ModeXYWH,
			},
			Probability: objConf,
			Class:       id,
			ID:          y.idGen.GetNext(),
		}

		group = append(group, result)
		lastCount++
	}

	return YOLOv8obbResult{
		DetectResults: group,
	}
}

// processStride processes the given stride
func (y *YOLOv8obb) processStride(input []int8, angleFeature []int8,
	gridH int, gridW int, stride int, data *strideData,
	zp int32, scale float32, angleFeatureZp int32, angleFeatureScale float32,
	index int) int {

	inputLocLen := 64
	validCount := 0
	thresI8 := qntF32ToAffine(unsigmoid(y.Params.BoxThreshold), zp, scale)

	for h := 0; h < gridH; h++ {
		for w := 0; w < gridW; w++ {
			for a := 0; a < y.Params.ObjectClassNum; a++ {

				// calculate the index of the current element in the input tensor
				// [1,tensor_len,grid_h,grid_w]
				idx := (inputLocLen+a)*gridW*gridH + h*gridW + w

				// is object confidence above the threshold
				if input[idx] >= thresI8 {

					boxConfF32 := sigmoid(deqntAffineToF32(input[idx], zp, scale))

					loc := make([]float32, inputLocLen)

					for i := 0; i < inputLocLen; i++ {
						loc[i] = deqntAffineToF32(input[i*gridW*gridH+h*gridW+w], zp, scale)
					}

					for i := 0; i < inputLocLen/16; i++ {
						softmax(loc[i*16:i*16+16], 16)
					}

					xywh_ := [4]float32{0, 0, 0, 0}
					xywh := [4]float32{0, 0, 0, 0}

					for dfl := 0; dfl < 16; dfl++ {
						xywh_[0] += loc[dfl] * float32(dfl)
						xywh_[1] += loc[1*16+dfl] * float32(dfl)
						xywh_[2] += loc[2*16+dfl] * float32(dfl)
						xywh_[3] += loc[3*16+dfl] * float32(dfl)
					}

					xywhAdd := [2]float32{xywh_[0] + xywh_[2], xywh_[1] + xywh_[3]}
					xywhSub := [2]float32{(xywh_[2] - xywh_[0]) / 2, (xywh_[3] - xywh_[1]) / 2}

					angleFeatureVal := deqntAffineToF32(angleFeature[index+(h*gridW)+w],
						angleFeatureZp, angleFeatureScale)
					angleFeatureVal = (angleFeatureVal - 0.25) * 3.1415927410125732

					angleFeatureCos := float32(math.Cos(float64(angleFeatureVal)))
					angleFeatureSin := float32(math.Sin(float64(angleFeatureVal)))

					// calculate final box dimensions
					xyMul1 := xywhSub[0] * angleFeatureCos
					xyMul2 := xywhSub[1] * angleFeatureSin
					xyMul3 := xywhSub[0] * angleFeatureSin
					xyMul4 := xywhSub[1] * angleFeatureCos

					xywh_[0] = ((xyMul1 - xyMul2) + float32(w) + 0.5) * float32(stride)
					xywh_[1] = ((xyMul3 + xyMul4) + float32(h) + 0.5) * float32(stride)
					xywh_[2] = xywhAdd[0] * float32(stride)
					xywh_[3] = xywhAdd[1] * float32(stride)

					xywh[0] = xywh_[0] - xywh_[2]/2
					xywh[1] = xywh_[1] - xywh_[3]/2
					xywh[2] = xywh_[2]
					xywh[3] = xywh_[3]

					// update data results
					data.filterBoxes = append(data.filterBoxes,
						xywh[0],         // x
						xywh[1],         // y
						xywh[2],         // w
						xywh[3],         // h
						angleFeatureVal, // angle
					)
					data.objProbs = append(data.objProbs, boxConfF32)
					data.classID = append(data.classID, a)

					validCount++
				}
			}
		}
	}

	return validCount
}

// nms implements a Non-Maximum Suppression (NMS) algorithm
func (y *YOLOv8obb) nms(validCount int, outputLocations []float32, classIds, order []int,
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

			xmin0 := outputLocations[n*5+0]
			ymin0 := outputLocations[n*5+1]
			w0 := outputLocations[n*5+2]
			h0 := outputLocations[n*5+3]
			angle0 := outputLocations[n*5+4]

			xmin1 := outputLocations[m*5+0]
			ymin1 := outputLocations[m*5+1]
			w1 := outputLocations[m*5+2]
			h1 := outputLocations[m*5+3]
			angle1 := outputLocations[m*5+4]

			iou := y.calculateOverlap(xmin0, ymin0, w0, h0, angle0, xmin1, ymin1, w1, h1, angle1)

			if iou > threshold {
				order[j] = -1
			}
		}
	}
}

// calculateOverlap calculates the Intersection over Union (IoU) between two
// rotated bounding boxes
func (y *YOLOv8obb) calculateOverlap(x1, y1, w1, h1, angle1, x2, y2, w2, h2,
	angle2 float32) float32 {

	// Define the data for two boxes
	rbbox1 := []float32{x1, y1, w1, h1, angle1}
	rbbox2 := []float32{x2, y2, w2, h2, angle2}

	// Call function to get corner points of the boxes
	corners1 := rbboxToCorners(rbbox1)
	corners2 := rbboxToCorners(rbbox2)

	var pts [][]float32
	numPts := 0

	// Check if corners of the first box are inside the second box
	for i := 0; i < 4; i++ {
		pointX := corners1[2*i]
		pointY := corners1[2*i+1]
		if pointInQuadrilateral(pointX, pointY, corners2) {
			numPts++
			pts = append(pts, []float32{pointX, pointY})
		}
	}

	// Check if corners of the second box are inside the first box
	for i := 0; i < 4; i++ {
		pointX := corners2[2*i]
		pointY := corners2[2*i+1]
		if pointInQuadrilateral(pointX, pointY, corners1) {
			numPts++
			pts = append(pts, []float32{pointX, pointY})
		}
	}

	// Check intersections of line segments between the two boxes
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			var pointX, pointY float32
			ret := false
			lineSegmentIntersection(corners1, corners2, i, j, &ret, &pointX, &pointY)
			if ret {
				numPts++
				pts = append(pts, []float32{pointX, pointY})
			}
		}
	}

	sortVertexInConvexPolygon(pts, numPts)

	polygonAreaVal := polygonArea(pts, numPts)

	// Calculate area_union
	areaUnion := rbbox1[2]*rbbox1[3] + rbbox2[2]*rbbox2[3] - polygonAreaVal
	return polygonAreaVal / areaUnion
}

// rbboxToCorners converts a rotated bounding box to its corner coordinates
func rbboxToCorners(rbbox []float32) []float32 {
	// Calculate the center coordinates
	cx := rbbox[0] + rbbox[2]/2
	cy := rbbox[1] + rbbox[3]/2

	// Get the width, height, and rotation angle of the box
	xD := rbbox[2]
	yD := rbbox[3]
	angle := rbbox[4]

	// Calculate cosine and sine of the angle
	aCos := float32(math.Cos(float64(angle)))
	aSin := float32(math.Sin(float64(angle)))

	// Initialize an 8-element slice for the corners (4 points with x and y)
	corners := make([]float32, 8)

	// Define the initial corner positions (relative to center)
	cornersX := []float32{-xD / 2, -xD / 2, xD / 2, xD / 2}
	cornersY := []float32{-yD / 2, yD / 2, yD / 2, -yD / 2}

	// Calculate the rotated corner positions
	for i := 0; i < 4; i++ {
		corners[2*i] = aCos*cornersX[i] - aSin*cornersY[i] + cx   // X coordinate
		corners[2*i+1] = aSin*cornersX[i] + aCos*cornersY[i] + cy // Y coordinate
	}

	return corners
}

// pointInQuadrilateral checks if a point is inside a quadrilateral
func pointInQuadrilateral(ptX, ptY float32, corners []float32) bool {
	ab0 := corners[2] - corners[0]
	ab1 := corners[3] - corners[1]
	ad0 := corners[6] - corners[0]
	ad1 := corners[7] - corners[1]
	ap0 := ptX - corners[0]
	ap1 := ptY - corners[1]

	abab := ab0*ab0 + ab1*ab1
	abap := ab0*ap0 + ab1*ap1
	adad := ad0*ad0 + ad1*ad1
	adap := ad0*ap0 + ad1*ap1

	return abab >= abap && abap >= 0 && adad >= adap && adap >= 0
}

// lineSegmentIntersection checks for intersection between line segments and calculates the intersection point
func lineSegmentIntersection(pts1, pts2 []float32, i, j int, ret *bool, pointX, pointY *float32) {
	// pts1, pts2 represent the corners of two boxes
	// i, j represent the index of points, taking the point and the next one to form a line segment

	A := []float32{pts1[2*i], pts1[2*i+1]}
	B := []float32{pts1[2*((i+1)%4)], pts1[2*((i+1)%4)+1]}
	C := []float32{pts2[2*j], pts2[2*j+1]}
	D := []float32{pts2[2*((j+1)%4)], pts2[2*((j+1)%4)+1]}

	BA0 := B[0] - A[0]
	BA1 := B[1] - A[1]
	DA0 := D[0] - A[0]
	CA0 := C[0] - A[0]
	DA1 := D[1] - A[1]
	CA1 := C[1] - A[1]

	// Check directions using cross product
	acd := DA1*CA0 > CA1*DA0
	bcd := (D[1]-B[1])*(C[0]-B[0]) > (C[1]-B[1])*(D[0]-B[0])

	if acd != bcd {
		abc := CA1*BA0 > BA1*CA0
		abd := DA1*BA0 > BA1*DA0

		// Check directions
		if abc != abd {
			DC0 := D[0] - C[0]
			DC1 := D[1] - C[1]
			ABBA := A[0]*B[1] - B[0]*A[1]
			CDDC := C[0]*D[1] - D[0]*C[1]
			DH := BA1*DC0 - BA0*DC1
			Dx := ABBA*DC0 - BA0*CDDC
			Dy := ABBA*DC1 - BA1*CDDC
			*pointX = Dx / DH
			*pointY = Dy / DH
			*ret = true
			return
		}
	}
	*ret = false
}

// sortVertexInConvexPolygon sorts the vertices of a convex polygon
func sortVertexInConvexPolygon(pts [][]float32, numOfInter int) {
	if numOfInter > 0 {
		center := []float32{0, 0}
		for i := 0; i < numOfInter; i++ {
			center[0] += pts[i][0]
			center[1] += pts[i][1]
		}
		center[0] /= float32(numOfInter)
		center[1] /= float32(numOfInter)

		sort.Slice(pts, func(i, j int) bool {
			return comparePoints(pts[i], pts[j], center)
		})
	}
}

// comparePoints is a comparison function used for sorting
func comparePoints(pt1, pt2, center []float32) bool {
	vx1 := pt1[0] - center[0]
	vy1 := pt1[1] - center[1]
	vx2 := pt2[0] - center[0]
	vy2 := pt2[1] - center[1]
	d1 := float32(math.Sqrt(float64(vx1*vx1 + vy1*vy1)))
	d2 := float32(math.Sqrt(float64(vx2*vx2 + vy2*vy2)))
	vx1 /= d1
	vy1 /= d1
	vx2 /= d2
	vy2 /= d2
	if vy1 < 0 {
		vx1 = -2 - vx1
	}
	if vy2 < 0 {
		vx2 = -2 - vx2
	}
	return vx1 < vx2
}

// triangleArea calculates the area of a triangle
func triangleArea(a, b, c []float32) float32 {
	return float32(math.Abs(float64((a[0]-c[0])*(b[1]-c[1])-(a[1]-c[1])*(b[0]-c[0])))) / 2.0
}

// polygonArea calculates the area of a polygon by decomposing it into triangles
func polygonArea(intPts [][]float32, numOfInter int) float32 {
	areaVal := float32(0.0)
	for i := 1; i < numOfInter-1; i++ {
		areaVal += triangleArea(intPts[0], intPts[i], intPts[i+1])
	}
	return areaVal
}
