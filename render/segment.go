package render

import (
	"fmt"
	"github.com/swdee/go-rknnlite/postprocess/result"
	"github.com/swdee/go-rknnlite/tracker"
	"gocv.io/x/gocv"
	"image"
	"image/color"
)

// SegmentMask renders the provided segment masks as a transparent overlay on
// top of the whole image
func SegmentMask(img *gocv.Mat, segMask []uint8, alpha float32) {

	// get pointer to image Mat so we can directly manipulate its pixels
	buf, err := img.DataPtrUint8() // length == total*3 (BGR)

	if err != nil {
		return
	}

	invA := 1.0 - alpha

	// increment through all pixels in segment mask
	for i, cls := range segMask {

		// skip pixels that have no segment mask
		if cls == 0 {
			continue
		}

		// pixel position in buffer
		pixelPos := i * 3

		// get original pixel colors directly from the byte slice
		b := float32(buf[pixelPos+0])
		g := float32(buf[pixelPos+1])
		r := float32(buf[pixelPos+2])

		// overlay colour to use
		col := classColors[cls%uint8(len(classColors))]

		// calculate blended colors based on alpha transparency
		buf[pixelPos+0] = uint8(b*invA + float32(col.B)*alpha)
		buf[pixelPos+1] = uint8(g*invA + float32(col.G)*alpha)
		buf[pixelPos+2] = uint8(r*invA + float32(col.R)*alpha)
	}
}

// boxLabel defines where the detection object label should be rendered on
// source image
type boxLabel struct {
	rect    image.Rectangle
	clr     color.RGBA
	text    string
	textPos image.Point
}

// findTopPoint finds the highest point (Y axis) of the given point vector
func findTopPoint(approx gocv.PointVector) image.Point {
	topPoint := approx.At(0)
	for i := 1; i < approx.Size(); i++ {
		pt := approx.At(i)
		if pt.Y < topPoint.Y {
			topPoint = pt
		}
	}
	return topPoint
}

// isContourInsideTrackerRect checks if the bounding box of a contour fits
// inside the bounding box of the detection result plus a pad
func isContourInsideTrackerRect(contourRect image.Rectangle,
	bbox *tracker.Rect, pad int) bool {

	return contourRect.Min.X >= int(bbox.TLX())-pad &&
		contourRect.Min.Y >= int(bbox.TLY())-pad &&
		contourRect.Max.X <= int(bbox.BRX())+pad &&
		contourRect.Max.Y <= int(bbox.BRY())+pad
}

// SegmentOutline renders the provided segment masks object outline for all
// objects.  The minArea is the value required for the masks minimum area for
// it to be used, this is needed to filter out small amounts of noise/artifacts
// contained in the mask from inferencing.
func SegmentOutline(img *gocv.Mat, segMask []uint8,
	detectResults []result.DetectResult, minArea float64,
	classNames []string, font Font, lineThickness int) error {

	width := img.Cols()
	height := img.Rows()
	boxesNum := len(detectResults)

	// create a Mat from the segMask once
	maskMat, err := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8U, segMask)

	if err != nil {
		return fmt.Errorf("error creating mask Mat: %w", err)
	}

	defer maskMat.Close()

	// one Mat for threshold results
	objMask := gocv.NewMat()
	defer objMask.Close()

	// keep a record of all box labels for later rendering
	boxLabels := make([]boxLabel, 0, boxesNum)

	// iterate over each unique object ID to isolate the mask
	for idx, dr := range detectResults {

		// clamp and build ROI from the detection box
		bb := dr.Box

		roiRect := image.Rect(
			max(0, bb.Left),
			max(0, bb.Top),
			min(width, bb.Right),
			min(height, bb.Bottom),
		)

		if roiRect.Empty() {
			continue
		}

		// crop the mask to just this ROI
		roi := maskMat.Region(roiRect)

		// threshold for the single object ID (idx+1)
		lowerBound := gocv.NewScalar(float64(idx+1), 0, 0, 0)
		upperBound := gocv.NewScalar(float64(idx+1), 0, 0, 0)
		gocv.InRangeWithScalar(roi, lowerBound, upperBound, &objMask)

		// Find contours for this object
		contours := gocv.FindContours(objMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)

		// Get the color for this object
		useClr := classColors[idx%len(classColors)]

		// Get the label from the detectResults
		labelText := classNames[dr.Class]

		// Calculate the horizontal center of the bounding box
		centerX := (bb.Left + bb.Right) / 2

		// Draw contours in this ROI
		for i := 0; i < contours.Size(); i++ {
			contour := contours.At(i)

			// filter out small contours picked up from aliasing/noise in binary mask
			if gocv.ContourArea(contour) < minArea {
				continue
			}

			// approximate to reduce vertex count
			approx := gocv.ApproxPolyDP(contour, 3, true)

			// translate approx.points from ROI to full sized image coords
			pts := approx.ToPoints()
			dx, dy := roiRect.Min.X, roiRect.Min.Y
			for j := range pts {
				pts[j].X += dx
				pts[j].Y += dy
			}

			// build a PointsVector just for this contour
			ptsVec := gocv.NewPointsVector()

			// Add our approximated PointVector to PointsVector
			ptsVec.Append(gocv.NewPointVectorFromPoints(pts))

			// draw it—coords auto‑offset since ROI is a view on maskMat
			gocv.Polylines(img, ptsVec, true, useClr, lineThickness)

			// find topmost point for label placement
			top := findTopPoint(approx)
			top.X += roiRect.Min.X
			top.Y += roiRect.Min.Y

			// create text for label
			text := fmt.Sprintf("%s %.2f", labelText, dr.Probability)
			textSize := gocv.GetTextSize(text, font.Face, font.Scale, font.Thickness)

			// Adjust the label position so the text is centered horizontally
			labelPosition := image.Pt(centerX-textSize.X/2, top.Y-font.BottomPad)

			// create box for placing text on
			bRect := image.Rect(
				centerX-textSize.X/2-font.LeftPad,
				top.Y-textSize.Y-font.TopPad-font.BottomPad,
				centerX+textSize.X/2+font.RightPad,
				top.Y,
			)

			// record label rendering details
			nextLabel := boxLabel{
				rect:    bRect,
				clr:     useClr,
				text:    text,
				textPos: labelPosition,
			}
			boxLabels = append(boxLabels, nextLabel)

			approx.Close()
			ptsVec.Close()
		}

		contours.Close()
		roi.Close()
	}

	drawBoxLabels(img, boxLabels, font)

	return nil
}

// PaintSegmentToFile paints the segment mask to and image file
func PaintSegmentToFile(filename string, height, width int,
	segMask []uint8, alpha float32) error {

	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)
	defer img.Close()

	SegmentMask(&img, segMask, alpha)

	if gocv.IMWrite(filename, img) {
		return nil
	}

	return fmt.Errorf("Failed to write to file")
}

// getSegMaskIDFromDetectionID
func getSegMaskIDFromDetectionID(detectID int64,
	detectResults []result.DetectResult) int {

	for segMaskID, detectResult := range detectResults {
		if detectID == detectResult.ID {
			// add one as 0 is the background
			return segMaskID + 1
		}
	}

	// not found
	return 0
}

// TrackerOutlines draws the object segmentation outlines around tracker objects.
// The minArea is the value required for the masks minimum area for
// it to be used, this is needed to filter out small amounts of noise/artifacts
// contained in the mask from inferencing.  The epsilon value effects the shape
// of the polygon outline.   The higher the value the more round it becomes.
func TrackerOutlines(img *gocv.Mat, segMask []uint8,
	trackResults []*tracker.STrack, detectResults []result.DetectResult,
	minArea float64, classNames []string, font Font, lineThickness int,
	epsilon float64) error {

	width := img.Cols()
	height := img.Rows()

	// create a Mat from the segMask once
	maskMat, err := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8U, segMask)

	if err != nil {
		return fmt.Errorf("error creating mask Mat: %w", err)
	}

	defer maskMat.Close()

	// one Mat for threshold results
	objMask := gocv.NewMat()
	defer objMask.Close()

	// keep a record of all box labels for later rendering
	boxLabels := make([]boxLabel, 0)

	// loop over each track result
	for _, tr := range trackResults {

		segMaskID := getSegMaskIDFromDetectionID(tr.GetDetectionID(), detectResults)

		// clamp & build ROI in full‑image coords
		bb := tr.GetRect()

		roi := image.Rect(
			max(0, int(bb.TLX())),
			max(0, int(bb.TLY())),
			min(width, int(bb.BRX())),
			min(height, int(bb.BRY())),
		)

		if roi.Empty() {
			continue
		}

		// crop the mask to just this ROI
		roiMat := maskMat.Region(roi)
		lowerBound := gocv.NewScalar(float64(segMaskID), 0, 0, 0)
		upperBound := gocv.NewScalar(float64(segMaskID), 0, 0, 0)
		gocv.InRangeWithScalar(roiMat, lowerBound, upperBound, &objMask)

		// Find contours for this object
		contours := gocv.FindContours(objMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)

		// Get the color for this object
		useClr := classColors[tr.GetTrackID()%len(classColors)]

		// create text for label
		text := fmt.Sprintf("%s %d", classNames[tr.GetLabel()], tr.GetTrackID())
		textSize := gocv.GetTextSize(text, font.Face, font.Scale, font.Thickness)

		// Calculate the horizontal center of the bounding box
		centerX := int((bb.TLX() + bb.BRX()) / 2)

		usedContours := 0

		// draw each contour
		for i := 0; i < contours.Size(); i++ {
			contour := contours.At(i)

			// filter out small contours picked up from aliasing/noise in binary mask
			if gocv.ContourArea(contour) < minArea {
				continue
			}

			// Check if the contour's bounding rectangle is inside the object's bounding box
			// get the bounding box in ROI coords…
			contourRect := gocv.BoundingRect(contour)
			// shift it into full‑image coords
			contourRect = contourRect.Add(image.Pt(roi.Min.X, roi.Min.Y))

			if !isContourInsideTrackerRect(contourRect, bb, 10) {
				continue
			}

			usedContours++

			approx := gocv.ApproxPolyDP(contour, epsilon, true)

			// translate co-ordinates into full sized image space
			pts := approx.ToPoints()
			dx, dy := roi.Min.X, roi.Min.Y
			for j := range pts {
				pts[j].X += dx
				pts[j].Y += dy
			}

			// Create a PointsVector to hold our PointVector
			ptsVec := gocv.NewPointsVector()

			// Add our approximated PointVector to PointsVector
			ptsVec.Append(gocv.NewPointVectorFromPoints(pts))

			// Draw polygon lines using PointsVector
			gocv.Polylines(img, ptsVec, true, useClr, lineThickness)

			ptsVec.Close()
			approx.Close()
		}

		contours.Close()
		roiMat.Close()

		// draw rectangle around detected object if contour not found
		if usedContours == 0 {
			rect := image.Rect(
				int(bb.TLX()),
				int(bb.TLY()),
				int(bb.BRX()),
				int(bb.BRY()),
			)
			gocv.Rectangle(img, rect, useClr, lineThickness)
		}

		// Find the topmost point of the contour
		topY := int(bb.TLY())

		// Adjust the label position so the text is centered horizontally
		labelPos := image.Pt(centerX-textSize.X/2, topY-font.BottomPad)

		// create box for placing text on
		bRect := image.Rect(
			centerX-textSize.X/2-font.LeftPad,
			topY-textSize.Y-font.TopPad-font.BottomPad,
			centerX+textSize.X/2+font.RightPad,
			topY,
		)

		// record label rendering details
		nextLabel := boxLabel{
			rect:    bRect,
			clr:     useClr,
			text:    text,
			textPos: labelPos,
		}
		boxLabels = append(boxLabels, nextLabel)
	}

	drawBoxLabels(img, boxLabels, font)

	return nil
}

// TrackerMask renders the provided segment masks as a transparent overlay on
// top of the whole image.  alpha is the amount of opacity to apply to the mask
// overlay.
func TrackerMask(img *gocv.Mat, segMask []uint8,
	trackResults []*tracker.STrack, detectResults []result.DetectResult,
	alpha float32) {

	boxesNum := len(trackResults)
	segMaskIDs := make([]int, boxesNum+1)

	// get pointer to image Mat so we can directly manipulate its pixels
	buf, err := img.DataPtrUint8()

	if err != nil {
		return
	}

	invA := 1.0 - alpha

	// increment through all pixels in segment mask
	for i, cls := range segMask {

		// skip pixels that have no segment mask
		if cls == 0 || int(cls) > boxesNum {
			continue
		}
		id := int(cls)

		// check if track ID is cached for pixel color
		trackID := segMaskIDs[id]

		if trackID == 0 {
			detID := detectResults[id-1].ID
			tID := getTrackIDFromDetectID(detID, trackResults)

			if tID == -1 {
				continue
			}

			trackID = tID
			segMaskIDs[id] = trackID
		}

		// pixel position in buffer
		pixelPos := i * 3

		b := float32(buf[pixelPos+0])
		g := float32(buf[pixelPos+1])
		r := float32(buf[pixelPos+2])

		// overlay colour to use
		col := classColors[trackID%len(classColors)]

		// calculate blended colors based on alpha transparency
		buf[pixelPos+0] = uint8(b*invA + float32(col.B)*alpha)
		buf[pixelPos+1] = uint8(g*invA + float32(col.G)*alpha)
		buf[pixelPos+2] = uint8(r*invA + float32(col.R)*alpha)
	}
}

// getTrackIDFromDetectID
func getTrackIDFromDetectID(detectID int64, trackResults []*tracker.STrack) int {

	for _, tResults := range trackResults {
		if tResults.GetDetectionID() == detectID {
			return tResults.GetTrackID()
		}
	}

	return -1
}
