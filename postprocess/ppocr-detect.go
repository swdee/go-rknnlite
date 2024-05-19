package postprocess

import (
	"fmt"
	clipper "github.com/ctessum/go.clipper"
	"github.com/swdee/go-rknnlite"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"math"
	"sort"
)

const (
	MaxContours = 1000
)

// PPOCRDetect defines the struct for the PPOCR Detection model inference
// post processing
type PPOCRDetect struct {
	Params PPOCRDetectParams
}

// PPOCRDetectParams defines the struct containing the PPOCRDetect parameters
// to use for post preocessing operations
type PPOCRDetectParams struct {
	Threshold    float32
	BoxThreshold float32
	Dilation     bool
	BoxType      string // poly|quad
	UnclipRatio  float32
	ScoreMode    string // slow|fast
	ModelWidth   int
	ModelHeight  int
}

// PPOCRPoint represents a point with x and y coordinates
type PPOCRPoint struct {
	X, Y int
}

// PPOCRBox represents a bounding box with corners and a score
type PPOCRBox struct {
	LeftTop     PPOCRPoint
	RightTop    PPOCRPoint
	RightBottom PPOCRPoint
	LeftBottom  PPOCRPoint
	Score       float32
}

// PPOCRDetectResult is the bounding box result for an area of text detected
type PPOCRDetectResult struct {
	Box []PPOCRBox
}

// NewPPOCRDetect returns and instance of the PPOCRDetect post processor
func NewPPOCRDetect(param PPOCRDetectParams) *PPOCRDetect {
	p := &PPOCRDetect{
		Params: param,
	}

	return p
}

// Detect takes the RKNN oiutputs and converts them to co-ordinates for the
// bounding boxes of each area of text detected.
func (p *PPOCRDetect) Detect(outputs *rknnlite.Outputs, scaleW float32,
	scaleH float32) []PPOCRDetectResult {

	results := make([]PPOCRDetectResult, len(outputs.Output))
	for idx, output := range outputs.Output {
		det, _ := p.detectText(output, scaleW, scaleH)

		results[idx] = det
	}

	return results
}

// detectText takes a single RKNN Output and returns the co-ordinates of the
// area of text detected
func (p *PPOCRDetect) detectText(output rknnlite.Output, scaleW float32,
	scaleH float32) (PPOCRDetectResult, error) {

	results := PPOCRDetectResult{
		Box: make([]PPOCRBox, 0, MaxContours),
	}

	// prepare bitmap

	// convert output to uint8
	cbuf := p.float32ToUint8(output.BufFloat)

	// create Mat for Uint8 values
	cbufMap := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(0, 0, 0, 0),
		p.Params.ModelHeight, p.Params.ModelWidth, gocv.MatTypeCV8UC1)
	defer cbufMap.Close()

	dataPtrUint8, err := cbufMap.DataPtrUint8()

	if err != nil {
		return results, fmt.Errorf("error getting data pointer for cbufMap: %w", err)
	}

	copy(dataPtrUint8, cbuf)

	// create Mat for float32 prediction values
	predMap := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(0, 0, 0, 0),
		p.Params.ModelHeight, p.Params.ModelWidth, gocv.MatTypeCV32F)

	defer predMap.Close()

	dataPtrFloat32, err := predMap.DataPtrFloat32()

	if err != nil {
		return results, fmt.Errorf("error getting data pointer for predMap: %w", err)
	}

	copy(dataPtrFloat32, output.BufFloat)

	threshold := p.Params.Threshold * 255
	maxvalue := float32(255.0)

	bitMap := gocv.NewMat()
	defer bitMap.Close()

	gocv.Threshold(cbufMap, &bitMap, threshold, maxvalue, gocv.ThresholdBinary)

	if p.Params.Dilation {
		dilaEle := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(2, 2))
		defer dilaEle.Close()
		gocv.Dilate(bitMap, &bitMap, dilaEle)
	}

	// find polygon contours
	minSize := 3
	contours := gocv.FindContours(bitMap, gocv.RetrievalList, gocv.ChainApproxSimple)

	numContours := contours.Size()
	if numContours > MaxContours {
		numContours = MaxContours
	}

	var boxes [][][]int
	var scores []float32

	for i := 0; i < numContours; i++ {
		// ensure we have 4 or more contours needed to make a poly/quad shape
		contour := contours.At(i)

		if contour.Size() < 3 {
			continue
		}

		var score float32

		if p.Params.BoxType == "poly" {

			epsilon := 0.002 * gocv.ArcLength(contour, true)
			points := gocv.ApproxPolyDP(contour, epsilon, true)

			if points.Size() < 4 {
				continue
			}

			pointVector := points.ToPoints()
			score = p.polygonScoreAcc(pointVector, predMap)

			if score < p.Params.BoxThreshold {
				continue
			}

			var boxForUnclip [][]float32

			for _, pt := range pointVector {
				boxForUnclip = append(boxForUnclip, []float32{float32(pt.X), float32(pt.Y)})
			}

			// start of unclip
			clipbox := p.unClip(boxForUnclip, p.Params.UnclipRatio)

			if clipbox.Height < 2 && clipbox.Width < 2 {
				continue
			}
			// end unclip

			ssid := int(math.Max(float64(clipbox.Width), float64(clipbox.Height)))

			cliparray := p.getMiniBoxes(clipbox)

			if ssid < minSize+2 {
				continue
			}

			var intcliparray [][]int

			for _, pt := range cliparray {
				intcliparray = append(intcliparray,
					[]int{int(p.clampf32(pt[0], 0, float32(p.Params.ModelWidth))), int(p.clampf32(pt[1], 0, float32(p.Params.ModelHeight)))})
			}

			boxes = append(boxes, intcliparray)

		} else {
			// quad

			box := gocv.MinAreaRect(contour)
			boxForUnclip := p.getMiniBoxes(box)

			ssid := int(math.Max(float64(box.Width), float64(box.Height)))

			if ssid < minSize {
				continue
			}

			if p.Params.ScoreMode == "slow" {
				pointVector := contour.ToPoints()
				score = p.polygonScoreAcc(pointVector, predMap)
			} else {
				score = p.boxScoreFast(boxForUnclip, predMap)
			}

			if score < p.Params.BoxThreshold {
				continue
			}

			// start of unclip
			clipbox := p.unClip(boxForUnclip, p.Params.UnclipRatio)

			if clipbox.Height < 2 && clipbox.Width < 2 {
				continue
			}
			// end of unclip

			ssid = int(math.Max(float64(clipbox.Width), float64(clipbox.Height)))

			cliparray := p.getMiniBoxes(clipbox)

			if ssid < minSize+2 {
				continue
			}

			var intcliparray [][]int

			for _, pt := range cliparray {
				intcliparray = append(intcliparray,
					[]int{int(p.clampf32(pt[0], 0, float32(p.Params.ModelWidth))), int(p.clampf32(pt[1], 0, float32(p.Params.ModelHeight)))})
			}

			boxes = append(boxes, intcliparray)

		}

		scores = append(scores, score)
	}

	var rootPoints [][][]int
	var rootScores []float32

	for n := range boxes {

		boxes[n] = p.orderPointsClockwise(boxes[n])

		for m := range boxes[n] {
			boxes[n][m][0] = p.minInt(p.maxInt(boxes[n][m][0], 0), p.Params.ModelWidth-1)
			boxes[n][m][1] = p.minInt(p.maxInt(boxes[n][m][1], 0), p.Params.ModelHeight-1)
		}

		rectWidth := int(math.Sqrt(float64(
			(boxes[n][0][0]-boxes[n][1][0])*(boxes[n][0][0]-boxes[n][1][0]) +
				(boxes[n][0][1]-boxes[n][1][1])*(boxes[n][0][1]-boxes[n][1][1]))))

		rectHeight := int(math.Sqrt(float64(
			(boxes[n][0][0]-boxes[n][3][0])*(boxes[n][0][0]-boxes[n][3][0]) +
				(boxes[n][0][1]-boxes[n][3][1])*(boxes[n][0][1]-boxes[n][3][1]))))

		if rectWidth <= 4 || rectHeight <= 4 {
			continue
		}

		rootPoints = append(rootPoints, boxes[n])
		rootScores = append(rootScores, scores[n])
	}

	for n := 0; n < len(rootPoints); n++ {
		if n >= MaxContours {
			break
		}

		rootPoints[n] = p.orderPointsClockwise(rootPoints[n])

		box := PPOCRBox{
			LeftTop: PPOCRPoint{X: int(float32(rootPoints[n][0][0]) * scaleW),
				Y: int(float32(rootPoints[n][0][1]) * scaleH)},
			RightTop: PPOCRPoint{X: int(float32(rootPoints[n][1][0]) * scaleW),
				Y: int(float32(rootPoints[n][1][1]) * scaleH)},
			RightBottom: PPOCRPoint{X: int(float32(rootPoints[n][2][0]) * scaleW),
				Y: int(float32(rootPoints[n][2][1]) * scaleH)},
			LeftBottom: PPOCRPoint{X: int(float32(rootPoints[n][3][0]) * scaleW),
				Y: int(float32(rootPoints[n][3][1]) * scaleH)},
			Score: rootScores[n],
		}

		results.Box = append(results.Box, box)
	}

	return results, nil
}

// float32ToUint8 converts a float32 slice to uint8 slice
func (p *PPOCRDetect) float32ToUint8(src []float32) []uint8 {

	dst := make([]uint8, len(src))

	for i, v := range src {
		dst[i] = uint8(v * 255)
	}

	return dst
}

// polygonScoreAcc calculates the average score (or mean value) of the pixel
// intensities within a specified polygonal contour on a given prediction map
func (p *PPOCRDetect) polygonScoreAcc(contour []image.Point, pred gocv.Mat) float32 {

	width := pred.Cols()
	height := pred.Rows()
	boxX := make([]float32, len(contour))
	boxY := make([]float32, len(contour))

	// extract the x and y coordinates of the contour points
	for i := 0; i < len(contour); i++ {
		boxX[i] = float32(contour[i].X)
		boxY[i] = float32(contour[i].Y)
	}

	// calculate the bounding box of the contour
	xmin := p.clamp(int(math.Floor(float64(p.minF32(boxX)))), 0, width-1)
	xmax := p.clamp(int(math.Ceil(float64(p.maxF32(boxX)))), 0, width-1)
	ymin := p.clamp(int(math.Floor(float64(p.minF32(boxY)))), 0, height-1)
	ymax := p.clamp(int(math.Ceil(float64(p.maxF32(boxY)))), 0, height-1)

	// create a mask of the bounding box size
	mask := gocv.NewMatWithSize(ymax-ymin+1, xmax-xmin+1, gocv.MatTypeCV8UC1)
	defer mask.Close()

	// shift the contour points to the bounding box coordinate system
	rookPoints := make([]image.Point, len(contour))

	for i := 0; i < len(contour); i++ {
		rookPoints[i] = image.Point{X: int(boxX[i]) - xmin, Y: int(boxY[i]) - ymin}
	}

	// convert rookPoints to a PointsVector
	ptsVector := gocv.NewPointsVectorFromPoints([][]image.Point{rookPoints})
	defer ptsVector.Close()

	// fill the polygon on the mask
	gocv.FillPoly(&mask, ptsVector, color.RGBA{R: 255, G: 255, B: 255, A: 255})

	// crop the prediction map to the bounding box
	croppedImg := pred.Region(image.Rect(xmin, ymin, xmax+1, ymax+1))
	defer croppedImg.Close()

	// ensure the mask and cropped image have the same dimensions
	if mask.Rows() != croppedImg.Rows() || mask.Cols() != croppedImg.Cols() {
		// Error: mask and cropped image dimensions do not match
		return -1.0
	}

	// calculate and return the mean value within the masked area
	score := croppedImg.MeanWithMask(mask)
	return float32(score.Val1)
}

// min finds the minimum value in a slice
func (p *PPOCRDetect) minF32(arr []float32) float32 {

	if len(arr) == 0 {
		return float32(math.Inf(1))
	}

	minVal := arr[0]

	for _, val := range arr {
		if val < minVal {
			minVal = val
		}
	}

	return minVal
}

// max finds the maximum value in a slice
func (p *PPOCRDetect) maxF32(arr []float32) float32 {

	if len(arr) == 0 {
		return float32(math.Inf(-1))
	}

	maxVal := arr[0]

	for _, val := range arr {
		if val > maxVal {
			maxVal = val
		}
	}

	return maxVal
}

// minInt returns the minimum Int from two values
func (p *PPOCRDetect) minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// maxInt returns the maximum Int from two values
func (p *PPOCRDetect) maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// clamp restricts a value between a minimum and maximum
func (p *PPOCRDetect) clamp(x, min, max int) int {

	if x < min {
		return min
	} else if x > max {
		return max
	}

	return x
}

// clampf32 clamps a float value between a minimum and maximum value
func (p *PPOCRDetect) clampf32(x, min, max float32) float32 {

	if x < min {
		return min
	} else if x > max {
		return max
	}

	return x
}

// minF32Val finds the minimum value in a slice
func (p *PPOCRDetect) minF32Val(values ...float32) float32 {
	// sort in ascending order
	sort.Slice(values, func(i, j int) bool {
		return values[i] < values[j]
	})
	return values[0]
}

// maxF32Val finds the maximum value in a slice
func (p *PPOCRDetect) maxF32Val(values ...float32) float32 {
	// sort in descending order
	sort.Slice(values, func(i, j int) bool {
		return values[i] > values[j]
	})
	return values[0]
}

// unClip method to perform the unclip operation
func (p *PPOCRDetect) unClip(box [][]float32, unclipRatio float32) gocv.RotatedRect {

	// calculate contour area and get distance
	distance := p.GetContourArea(box, unclipRatio)

	// convert the box points to Clipper Path
	var path clipper.Path

	for _, pt := range box {
		path = append(path, &clipper.IntPoint{X: clipper.CInt(pt[0]), Y: clipper.CInt(pt[1])})
	}

	// create a ClipperOffset object and add the path
	co := clipper.NewClipperOffset()
	co.AddPath(path, clipper.JtRound, clipper.EtClosedPolygon)

	// execute the offset operation
	solution := co.Execute(float64(distance))

	// convert the solution back to points
	var points []image.Point

	for _, sol := range solution {
		for _, pt := range sol {
			points = append(points, image.Point{X: int(pt.X), Y: int(pt.Y)})
		}
	}

	// create and return the RotatedRect result
	var res gocv.RotatedRect

	if len(points) == 0 {
		res = gocv.RotatedRect{
			Points: []image.Point{image.Pt(0, 0), image.Pt(0, 1),
				image.Pt(1, 1), image.Pt(1, 0)},
			BoundingRect: image.Rect(0, 0, 1, 1),
			Center:       image.Pt(0, 0),
			Width:        1,
			Height:       1,
			Angle:        0,
		}

	} else {
		// convert points to gocv.PointVector
		pointVector := gocv.NewPointVectorFromPoints(points)
		defer pointVector.Close()

		res = gocv.MinAreaRect(pointVector)
	}

	return res
}

// GetContourArea calculates the area of the contour and returns the distance
// based on the unclip ratio
func (p *PPOCRDetect) GetContourArea(box [][]float32, unclipRatio float32) float32 {

	ptsNum := len(box)
	area := float32(0.0)
	dist := float32(0.0)

	for i := 0; i < ptsNum; i++ {
		area += box[i][0]*box[(i+1)%ptsNum][1] - box[i][1]*box[(i+1)%ptsNum][0]
		dist += float32(math.Sqrt(float64((box[i][0]-box[(i+1)%ptsNum][0])*(box[i][0]-box[(i+1)%ptsNum][0]) +
			(box[i][1]-box[(i+1)%ptsNum][1])*(box[i][1]-box[(i+1)%ptsNum][1]))))
	}

	area = float32(math.Abs(float64(area / 2.0)))
	return area * unclipRatio / dist
}

// getMiniBoxes extracts and sorts the four corners of the RotatedRect
func (p *PPOCRDetect) getMiniBoxes(rect gocv.RotatedRect) [][]float32 {

	array := make([][]float32, 4)

	for i, pt := range rect.Points {
		array[i] = []float32{float32(pt.X), float32(pt.Y)}
	}

	// sort points by x-coordinates
	p.xsortFp32(array)

	// Sort points by x-coordinates
	sort.Slice(array, func(i, j int) bool {
		return array[i][0] < array[j][0]
	})

	idx1, idx2, idx3, idx4 := array[0], array[1], array[2], array[3]

	if array[3][1] <= array[2][1] {
		idx2 = array[3]
		idx3 = array[2]
	} else {
		idx2 = array[2]
		idx3 = array[3]
	}

	if array[1][1] <= array[0][1] {
		idx1 = array[1]
		idx4 = array[0]
	} else {
		idx1 = array[0]
		idx4 = array[1]
	}

	array[0] = idx1
	array[1] = idx2
	array[2] = idx3
	array[3] = idx4

	return array
}

// xsortFp32 sorts points based on their x-coordinates
func (p *PPOCRDetect) xsortFp32(points [][]float32) {
	sort.Slice(points, func(i, j int) bool {
		return points[i][0] < points[j][0]
	})
}

// xsortInt sorts points based on their x-coordinates
func (p *PPOCRDetect) xsortInt(points [][]int) {
	sort.Slice(points, func(i, j int) bool {
		return points[i][0] < points[j][0]
	})
}

// orderPointsClockwise orders points in a clockwise manner
func (p *PPOCRDetect) orderPointsClockwise(pts [][]int) [][]int {

	box := make([][]int, len(pts))
	copy(box, pts)
	p.xsortInt(box)

	leftmost := [][]int{box[0], box[1]}
	rightmost := [][]int{box[2], box[3]}

	if leftmost[0][1] > leftmost[1][1] {
		leftmost[0], leftmost[1] = leftmost[1], leftmost[0]
	}

	if rightmost[0][1] > rightmost[1][1] {
		rightmost[0], rightmost[1] = rightmost[1], rightmost[0]
	}

	rect := [][]int{leftmost[0], rightmost[0], rightmost[1], leftmost[1]}
	return rect
}

// boxScoreFast calculates the score of a box
func (p *PPOCRDetect) boxScoreFast(boxArray [][]float32, pred gocv.Mat) float32 {

	width := pred.Cols()
	height := pred.Rows()

	boxX := []float32{boxArray[0][0], boxArray[1][0], boxArray[2][0], boxArray[3][0]}
	boxY := []float32{boxArray[0][1], boxArray[1][1], boxArray[2][1], boxArray[3][1]}

	xmin := p.clamp(int(math.Floor(float64(p.minF32Val(boxX...)))), 0, width-1)
	xmax := p.clamp(int(math.Ceil(float64(p.maxF32Val(boxX...)))), 0, width-1)
	ymin := p.clamp(int(math.Floor(float64(p.minF32Val(boxY...)))), 0, height-1)
	ymax := p.clamp(int(math.Ceil(float64(p.maxF32Val(boxY...)))), 0, height-1)

	mask := gocv.NewMatWithSize(ymax-ymin+1, xmax-xmin+1, gocv.MatTypeCV8UC1)
	defer mask.Close()

	rootPoint := []image.Point{
		{X: int(boxArray[0][0]) - xmin, Y: int(boxArray[0][1]) - ymin},
		{X: int(boxArray[1][0]) - xmin, Y: int(boxArray[1][1]) - ymin},
		{X: int(boxArray[2][0]) - xmin, Y: int(boxArray[2][1]) - ymin},
		{X: int(boxArray[3][0]) - xmin, Y: int(boxArray[3][1]) - ymin},
	}

	ptsVector := gocv.NewPointsVectorFromPoints([][]image.Point{rootPoint})
	defer ptsVector.Close()

	gocv.FillPoly(&mask, ptsVector, color.RGBA{R: 255, G: 255, B: 255, A: 255})

	croppedImg := pred.Region(image.Rect(xmin, ymin, xmax+1, ymax+1))
	defer croppedImg.Close()

	// ensure the mask and cropped image have the same dimensions
	if mask.Rows() != croppedImg.Rows() || mask.Cols() != croppedImg.Cols() {
		// Error: mask and cropped image dimensions do not match
		return -1.0
	}

	mean := croppedImg.MeanWithMask(mask)
	return float32(mean.Val1)
}
