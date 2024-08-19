package render

import (
	"fmt"
	"github.com/swdee/go-rknnlite/postprocess"
	"gocv.io/x/gocv"
	"image"
	"image/color"
)

// SegmentMask renders the provided segment masks as a transparent overlay on
// top of the whole image
func SegmentMask(img *gocv.Mat, segMask []uint8, alpha float32) {

	// get dimensions
	width := img.Cols()
	height := img.Rows()

	// it is too slow to manipulate pixel by pixel using GoCV due to slowness
	// over CGO.  So we copy the bytes from the source image and manipulate
	// the bytes directly before copying back to a Mat
	imgData := img.ToBytes()

	// iterate over each pixel in the segmentation mask
	for j := 0; j < height; j++ {
		for k := 0; k < width; k++ {

			idx := j*width + k

			if segMask[idx] != 0 {

				classIndex := segMask[idx] % uint8(len(classColors))
				color := classColors[classIndex]

				// calculate position in the byte slice
				pixelPos := j*width*3 + k*3

				// get original pixel colors directly from the byte slice
				b, g, r := imgData[pixelPos+0], imgData[pixelPos+1], imgData[pixelPos+2]

				// calculate blended colors based on alpha transparency
				imgData[pixelPos+0] = uint8(float32(b)*(1-alpha) + float32(color.B)*alpha)
				imgData[pixelPos+1] = uint8(float32(g)*(1-alpha) + float32(color.G)*alpha)
				imgData[pixelPos+2] = uint8(float32(r)*(1-alpha) + float32(color.R)*alpha)
			}
		}
	}

	// copy back to the original mat
	tmpImg, _ := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8UC3, imgData)
	defer tmpImg.Close()
	tmpImg.CopyTo(img)
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

// SegmentOutline renders the provided segment masks object outline for all
// objects
func SegmentOutline(img *gocv.Mat, segMask []uint8,
	detectResults []postprocess.DetectResult, minArea float64,
	classNames []string, font Font, lineThickness int) error {

	width := img.Cols()
	height := img.Rows()
	boxesNum := len(detectResults)

	// create a Mat from the segMask
	maskMat, err := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8U, segMask)

	if err != nil {
		return fmt.Errorf("error creating mask Mat: %w", err)
	}

	defer maskMat.Close()

	// keep a record of all box labels for later rendering
	boxLabels := make([]boxLabel, 0)

	// iterate over each unique object ID to isolate the mask
	for objID := 1; objID < boxesNum+1; objID++ {

		// Create a binary mask for the current object (isolate the object by objID)
		objMask := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
		lowerBound := gocv.Scalar{Val1: float64(objID)}
		upperBound := gocv.Scalar{Val1: float64(objID)}
		gocv.InRangeWithScalar(maskMat, lowerBound, upperBound, &objMask)
		defer objMask.Close()

		// Find contours for this object
		contours := gocv.FindContours(objMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
		defer contours.Close()

		// Get the color for this object
		colorIndex := (objID - 1) % len(classColors)
		useClr := classColors[colorIndex]

		// Get the label from the detectResults
		label := classNames[detectResults[objID-1].Class]

		// Calculate the horizontal center of the bounding box
		boundingBox := detectResults[objID-1].Box
		centerX := (boundingBox.Left + boundingBox.Right) / 2

		// Draw contours
		for i := 0; i < contours.Size(); i++ {
			c := contours.At(i)

			// filter out small contours picked up from aliasing/noise in binary mask
			area := gocv.ContourArea(c)

			if area < minArea {
				continue
			}

			approx := gocv.ApproxPolyDP(c, 3, true)

			// Create a PointsVector to hold our PointVector
			ptsVec := gocv.NewPointsVector()

			// Add our approximated PointVector to PointsVector
			ptsVec.Append(approx)

			// Draw polygon lines using PointsVector
			gocv.Polylines(img, ptsVec, true, useClr, lineThickness)

			// Find the topmost point of the contour
			topPoint := findTopPoint(approx)

			// create text for label
			text := fmt.Sprintf("%s %.2f", label, detectResults[objID-1].Probability)
			textSize := gocv.GetTextSize(text, font.Face, font.Scale, font.Thickness)

			// Adjust the label position so the text is centered horizontally
			labelPosition := image.Pt(centerX-textSize.X/2, topPoint.Y-font.BottomPad)

			// create box for placing text on
			bRect := image.Rect(centerX-textSize.X/2-font.LeftPad,
				topPoint.Y-textSize.Y-font.TopPad-font.BottomPad,
				centerX+textSize.X/2+font.RightPad, topPoint.Y)

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
	}

	// draw all precalculated box labels so they are the top most layer on the
	// image and don't get overlapped with segment contour lines
	for _, box := range boxLabels {
		// draw box text gets written on
		gocv.Rectangle(img, box.rect, box.clr, -1)

		// Draw the label over box
		gocv.PutTextWithParams(img, box.text, box.textPos,
			font.Face, font.Scale, font.Color, font.Thickness,
			font.LineType, false)
	}

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
