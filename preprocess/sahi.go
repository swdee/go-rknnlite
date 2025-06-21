package preprocess

import (
	"errors"
	"github.com/swdee/go-rknnlite/postprocess/result"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"math"
	"sort"
)

// SAHI defines the struct used for Slicing Aided Hyper Inference
type SAHI struct {
	// sliceWidth is the width of each image slice
	sliceWidth int
	// sliceHeight is the height of each image slice
	sliceHeight int
	// overlapWidth is a ratio from 0.0 to 1.0 to represent the number of pixels
	// to overlap each slice.  A value of 0.2 represents 20% of sliceWidth's pixels
	overlapWidth float32
	// overlapHeight is a ratio from 0.0 to 1.0 to represent the number of pixels
	// to overlap each slice.  A value of 0.2 represents 20% of sliceHeight's pixels
	overlapHeight float32
	// results stores a slices object detection results
	results []sahiResult
	// nextID is a counter that increments and provides the next number
	// for each detection result ID
	idGen *result.IDGenerator
}

// sahiResult defines a struct to store a slice and its detection results
type sahiResult struct {
	slice Slice
	det   []result.DetectResult
}

// Slice defines the struct used to store the coordinates for a slice of the
// source image
type Slice struct {
	// X is the coordinate of the slices left corner
	X int
	// Y is the coordinate of the slices top corner
	Y int
	// X2 is the coordinate of the slices right corner
	X2 int
	// Y2 is the coordinate of the slices bottom corner
	Y2 int
	// slice is the sliced image Mat
	slice gocv.Mat
	// resizer is an instance of the image resizer
	resizer *Resizer
	// destMat is the destination Mat after crop and resize of the slice
	destMat gocv.Mat
}

// NewSAHI returns a SAHI (Slicing Aided Hyper Inference) instance for slicing
// a source image into a series of tiles for inference.  The sliceWidth and
// sliceHeight should be the same size as the Models input tensor dimensions.
func NewSAHI(sliceWidth, sliceHeight int, overlapWidth, overlapHeight float32) *SAHI {
	s := &SAHI{
		sliceWidth:    sliceWidth,
		sliceHeight:   sliceHeight,
		overlapWidth:  overlapWidth,
		overlapHeight: overlapHeight,
		results:       make([]sahiResult, 0),
		idGen:         result.NewIDGenerator(),
	}

	return s
}

// computePositions returns the start‐coordinates (0‐based) of each tile
// along one axis, and the computed tile length.  It guarantees:
//
//   - you get the smallest n tiles so that n*tileLen – (n−1)*step >= srcLen
//   - step = (srcLen−tileLen)/(n−1) is =< sliceLen
//   - thus overlap = tileLen − step >= sliceLen*overlapRatio
//
// any leftover pixels to cover the image get spread evenly via rounding.
// function returns a slice of positions and tileLen
func (s *SAHI) computePositions(srcLen, sliceLen int, overlapRatio float32) ([]int, int) {

	// minimum pixel‐overlap
	minOv := int(math.Ceil(float64(sliceLen) * float64(overlapRatio)))

	// tile length = sliceLen + that minimum overlap
	tileLen := sliceLen + minOv

	// how many tiles you'd need if you stepped by sliceLen each time?
	//    this ensures step =< sliceLen and so overlap >= minOv
	n := int(math.Ceil(float64(srcLen-tileLen)/float64(sliceLen))) + 1
	if n < 1 {
		n = 1
	}

	// actual step (evenly spread)
	denom := n - 1
	var step float64

	if denom > 0 {
		step = float64(srcLen-tileLen) / float64(denom)
	} else {
		step = 0
	}

	// build positions, rounding to distribute leftovers
	positions := make([]int, n)

	for i := 0; i < n; i++ {
		p := int(math.Round(step * float64(i)))

		// clamp to [0, srcLen-tileLen]
		if p < 0 {
			p = 0
		} else if p > srcLen-tileLen {
			p = srcLen - tileLen
		}

		positions[i] = p
	}

	return positions, tileLen
}

// Slice slices the given input image into a series of tiles
func (s *SAHI) Slice(src gocv.Mat) []Slice {
	// get dimensions of source image
	srcH, srcW := src.Rows(), src.Cols()

	// compute X starts and tileW
	xs, tileW := s.computePositions(srcW, s.sliceWidth, s.overlapWidth)
	// compute Y starts and tileH
	ys, tileH := s.computePositions(srcH, s.sliceHeight, s.overlapHeight)

	slices := make([]Slice, 0, len(xs)*len(ys))

	for _, y := range ys {
		for _, x := range xs {
			rect := image.Rect(x, y, x+tileW, y+tileH)

			// letter‐box this tile down to sliceWidth x sliceHeight
			resizer := NewResizer(tileW, tileH, s.sliceWidth, s.sliceHeight)

			slices = append(slices, Slice{
				X:       x,
				Y:       y,
				X2:      x + tileW,
				Y2:      y + tileH,
				slice:   src.Region(rect),
				resizer: resizer,
				destMat: gocv.NewMat(),
			})
		}
	}

	return slices
}

// AddResult adds the slice and its detection result
func (s *SAHI) AddResult(slice Slice, res []result.DetectResult) {
	s.results = append(s.results, sahiResult{
		slice: slice,
		det:   res,
	})
}

// GetDetectResults returns the global detection results for the source image
// made up from the combination of all slices detect results.
//   - iouThreshold is the intersection-over-union value (NMSThreshold) used in the YOLO
//     processor parameters.
//   - smallBoxOverlapThreshold is the percentage represented from a value of
//     0 to 1 that small boxes need to overlap by to be discarded which occurs when
//     and object sits on the slice/tile overlap boundary
func (s *SAHI) GetDetectResults(iouThreshold, smallBoxOverlapThresh float32) []result.DetectResult {

	// collate objects into a result for returning
	group := make([]result.DetectResult, 0)

	// for each slice
	for _, sr := range s.results {

		// for each object/detection in the slice
		for _, dr := range sr.det {
			// remap the coordinates of the detection result to the global coordinates
			// of the source image
			gresult := result.DetectResult{
				Box: result.BoxRect{
					Left:   sr.slice.X + dr.Box.Left,
					Top:    sr.slice.Y + dr.Box.Top,
					Right:  sr.slice.X + dr.Box.Right,
					Bottom: sr.slice.Y + dr.Box.Bottom,
				},
				Probability: dr.Probability,
				Class:       dr.Class,
				ID:          s.idGen.GetNext(),
			}

			group = append(group, gresult)
		}
	}

	// Sort by descending probability
	sort.Slice(group, func(i, j int) bool {
		return group[i].Probability > group[j].Probability
	})

	// run class-aware NMS
	return s.nmsCluster(group, iouThreshold, smallBoxOverlapThresh)
}

// FreeResults releases stored detection results.  This should be called after
// processing of each image to clear out results added by AddResult().
func (s *SAHI) FreeResults() {
	s.results = s.results[:0]
}

// iou computes the Intersection-over-Union of two boxes.
func (s *SAHI) iou(a, b result.BoxRect) float32 {
	x1 := max(a.Left, b.Left)
	y1 := max(a.Top, b.Top)
	x2 := min(a.Right, b.Right)
	y2 := min(a.Bottom, b.Bottom)

	w := float32(max(0, x2-x1))
	h := float32(max(0, y2-y1))
	inter := w * h

	areaA := float32((a.Right - a.Left) * (a.Bottom - a.Top))
	areaB := float32((b.Right - b.Left) * (b.Bottom - b.Top))

	return inter / (areaA + areaB - inter)
}

// max returns max between two numbers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// min returns minimum between two numbers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// NMS runs non-maximum suppression on a sorted slice of detections
// Assumes detections are sorted descending by Probability.
func (s *SAHI) nms(detections []result.DetectResult,
	iouThreshold, smallBoxOverlapThresh float32) []result.DetectResult {

	keep := make([]result.DetectResult, 0, len(detections))

	// track by class, so we don’t suppress across different classes
	for _, det := range detections {
		skip := false

		for _, kept := range keep {
			if det.Class != kept.Class {
				continue
			}

			// do IoU check
			if s.iou(det.Box, kept.Box) > iouThreshold {
				skip = true
				break
			}

			// do partial box check, if the intersection covers most of the small box
			inter := s.intersectionArea(det.Box, kept.Box)
			areaDet := s.boxArea(det.Box)
			if areaDet > 0 && float32(inter)/float32(areaDet) > smallBoxOverlapThresh {
				skip = true
				break
			}
		}

		if !skip {
			keep = append(keep, det)
		}
	}

	return keep
}

// nmsCluster picks one box per overlapping cluster (class‐agnostic),
// choosing the larghest area (tie‐break on confidence), and uses both IoU
// and small‐box overlap tests to form clusters.
//   - detections must be sorted descending by Probability before calling.
//   - iouThreshold is your overlap cutoff (e.g. 0.45)
//   - smallBoxOverlapThresh is the fraction of the small box’s area that must
//     be overlapped to consider it a duplicate (e.g. 0.7)
func (s *SAHI) nmsCluster(dets []result.DetectResult,
	iouThreshold, smallBoxOverlapThresh float32) []result.DetectResult {

	n := len(dets)
	suppressed := make([]bool, n)
	keep := make([]result.DetectResult, 0, n)

	for i, base := range dets {
		if suppressed[i] {
			continue
		}

		// start a new cluster with "base"
		cluster := []result.DetectResult{base}
		suppressed[i] = true

		for j := i + 1; j < n; j++ {
			if suppressed[j] {
				continue
			}

			other := dets[j]

			// decide if "other" belongs in this cluster
			inCluster := false

			// IoU test
			if s.iou(base.Box, other.Box) > iouThreshold {
				inCluster = true
			} else {
				// small‐box self‐overlap test
				inter := s.intersectionArea(base.Box, other.Box)
				areaOther := s.boxArea(other.Box)
				if areaOther > 0 && float32(inter)/float32(areaOther) > smallBoxOverlapThresh {
					inCluster = true
				}
			}

			if !inCluster {
				continue
			}

			// assign to this cluster
			suppressed[j] = true
			cluster = append(cluster, other)
		}

		// pick the single largest‐area box (tie‐break on probability)
		best := cluster[0]
		bestArea := s.boxArea(best.Box)

		for _, c := range cluster[1:] {
			a := s.boxArea(c.Box)

			if a > bestArea || (a == bestArea && c.Probability > best.Probability) {
				best = c
				bestArea = a
			}
		}

		keep = append(keep, best)
	}

	return keep
}

// intersectionArea returns the raw pixel-area of overlap between two boxes.
func (s *SAHI) intersectionArea(a, b result.BoxRect) int {
	x1 := max(a.Left, b.Left)
	y1 := max(a.Top, b.Top)
	x2 := min(a.Right, b.Right)
	y2 := min(a.Bottom, b.Bottom)

	if x2 <= x1 || y2 <= y1 {
		return 0
	}

	return (x2 - x1) * (y2 - y1)
}

// boxArea returns the pixel‐area of a single box.
func (s *SAHI) boxArea(a result.BoxRect) int {
	return max(0, a.Right-a.Left) * max(0, a.Bottom-a.Top)
}

// Mat returns the slices Mat after cropping and resizing to be used
// for running inference on
func (s *Slice) Mat() *gocv.Mat {
	s.resizer.LetterBoxResize(s.slice, &s.destMat, color.RGBA{R: 0, G: 0, B: 0, A: 255})
	return &s.destMat
}

// Resizer returns the slices letter box resize
func (s *Slice) Resizer() *Resizer {
	return s.resizer
}

// Free releases the slice from memory
func (s *Slice) Free() error {
	err := s.resizer.Close()
	err2 := s.slice.Close()
	err3 := s.destMat.Close()

	return errors.Join(err, err2, err3)
}
