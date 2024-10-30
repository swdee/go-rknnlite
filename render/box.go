package render

import (
	"fmt"
	"github.com/swdee/go-rknnlite/postprocess"
	"github.com/swdee/go-rknnlite/tracker"
	"gocv.io/x/gocv"
	"image"
	"math"
)

// DetectionBoxes renders the bounding boxes around the object detected
func DetectionBoxes(img *gocv.Mat, detectResults []postprocess.DetectResult,
	classNames []string, font Font, lineThickness int) {

	// keep a record of all box labels for later rendering
	boxLabels := make([]boxLabel, 0)

	// draw detection boxes
	for i, detResult := range detectResults {

		// Get the color for this object
		colorIndex := i % len(classColors)
		useClr := classColors[colorIndex]

		// draw rectangle around detected object
		rect := image.Rect(detResult.Box.Left, detResult.Box.Top, detResult.Box.Right,
			detResult.Box.Bottom)
		gocv.Rectangle(img, rect, useClr, lineThickness)

		// create text for label
		text := fmt.Sprintf("%s %.2f", classNames[detResult.Class], detResult.Probability)
		textSize := gocv.GetTextSize(text, font.Face, font.Scale, font.Thickness)

		// Calculate the alignment of text label
		var centerX int

		switch font.Alignment {
		case Center:
			centerX = (detResult.Box.Left + detResult.Box.Right) / 2

		case Right:
			centerX = detResult.Box.Right - (textSize.X / 2) - font.RightPad + (lineThickness / 2)

		case Left:
			fallthrough
		default:
			centerX = detResult.Box.Left + (textSize.X / 2) + font.LeftPad - (lineThickness / 2)
		}

		// Adjust the label position so the text is centered horizontally
		labelPosition := image.Pt(centerX-textSize.X/2, detResult.Box.Top-font.BottomPad)

		// create box for placing text on
		bRect := image.Rect(centerX-textSize.X/2-font.LeftPad,
			detResult.Box.Top-textSize.Y-font.TopPad-font.BottomPad,
			centerX+textSize.X/2+font.RightPad, detResult.Box.Top)

		// record label rendering details
		nextLabel := boxLabel{
			rect:    bRect,
			clr:     useClr,
			text:    text,
			textPos: labelPosition,
		}

		boxLabels = append(boxLabels, nextLabel)
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
}

// TrackerBoxes renders the bounding boxes around the object detected for
// tracker results
func TrackerBoxes(img *gocv.Mat, trackResults []*tracker.STrack,
	classNames []string, font Font, lineThickness int) {

	// keep a record of all box labels for later rendering
	boxLabels := make([]boxLabel, 0)

	for _, tResult := range trackResults {

		// calculate the coordinates in the original image
		boxLeft := int(tResult.GetRect().TLX())
		boxTop := int(tResult.GetRect().TLY())
		boxRight := int(tResult.GetRect().BRX())
		boxBottom := int(tResult.GetRect().BRY())

		// Get the color for this object
		colorIndex := tResult.GetTrackID() % len(classColors)
		useClr := classColors[colorIndex]

		// draw rectangle around detected object
		rect := image.Rect(boxLeft, boxTop, boxRight, boxBottom)
		gocv.Rectangle(img, rect, useClr, lineThickness)

		// create text for label
		text := fmt.Sprintf("%s %d", classNames[tResult.GetLabel()], tResult.GetTrackID())
		textSize := gocv.GetTextSize(text, font.Face, font.Scale, font.Thickness)

		// Calculate the alignment of text label
		var centerX int

		switch font.Alignment {
		case Center:
			centerX = (boxLeft + boxRight) / 2

		case Right:
			centerX = boxRight - (textSize.X / 2) - font.RightPad + (lineThickness / 2)

		case Left:
			fallthrough
		default:
			centerX = boxLeft + (textSize.X / 2) + font.LeftPad - (lineThickness / 2)
		}

		// Adjust the label position so the text is centered horizontally
		labelPosition := image.Pt(centerX-textSize.X/2, boxTop-font.BottomPad)

		// create box for placing text on
		bRect := image.Rect(centerX-textSize.X/2-font.LeftPad,
			boxTop-textSize.Y-font.TopPad-font.BottomPad,
			centerX+textSize.X/2+font.RightPad, boxTop)

		// record label rendering details
		nextLabel := boxLabel{
			rect:    bRect,
			clr:     useClr,
			text:    text,
			textPos: labelPosition,
		}
		boxLabels = append(boxLabels, nextLabel)

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
}

// OrientedBoundingBoxes renders the oriented bounding boxes around the object
// detected
func OrientedBoundingBoxes(img *gocv.Mat, detectResults []postprocess.DetectResult,
	classNames []string, font Font, lineThickness int) {

	// keep a record of all box labels for later rendering
	boxLabels := make([]boxLabel, 0)

	for i, detResult := range detectResults {

		// Get the color for this object
		colorIndex := i % len(classColors)
		useClr := classColors[colorIndex]

		// Convert rotated bounding box to corner points
		corners := obbToCorners(
			detResult.Box.X,
			detResult.Box.Y,
			detResult.Box.Width,
			detResult.Box.Height,
			detResult.Box.Angle,
		)

		// Draw lines between the corners to form the rotated rectangle
		for n := 0; n < 4; n++ {
			// Get the indices of the current corner and the next corner
			index1 := n * 2
			index2 := ((n + 1) % 4) * 2

			// Draw the line between corners
			pt1 := image.Point{X: corners[index1], Y: corners[index1+1]}
			pt2 := image.Point{X: corners[index2], Y: corners[index2+1]}
			gocv.Line(img, pt1, pt2, useClr, lineThickness)
		}

		// create text for label
		text := fmt.Sprintf("%s %.2f", classNames[detResult.Class], detResult.Probability)
		textSize := gocv.GetTextSize(text, font.Face, font.Scale, font.Thickness)

		// Calculate the alignment of text label
		minX, centerX, maxX, topY := calculateCornerTextAlignment(corners)

		var useX int

		switch font.Alignment {
		case Center:
			useX = centerX

		case Right:
			useX = maxX - (textSize.X / 2) - font.RightPad + (lineThickness / 2)

		case Left:
			fallthrough
		default:
			useX = minX + (textSize.X / 2) + font.LeftPad - (lineThickness / 2)
		}

		// Adjust the label position so the text is centered horizontally
		labelPosition := image.Pt(useX-textSize.X/2, topY-font.BottomPad)

		// create box for placing text on
		bRect := image.Rect(useX-textSize.X/2-font.LeftPad,
			topY-textSize.Y-font.TopPad-font.BottomPad,
			useX+textSize.X/2+font.RightPad, topY)

		// record label rendering details
		nextLabel := boxLabel{
			rect:    bRect,
			clr:     useClr,
			text:    text,
			textPos: labelPosition,
		}
		boxLabels = append(boxLabels, nextLabel)
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
}

// obbToCorners converts oritented bounding box to corners
func obbToCorners(x, y, w, h int, angle float32) [8]int {

	// Calculate center coordinates
	cx := float32(x + w/2)
	cy := float32(y + h/2)

	// Half dimensions
	xD := float32(w / 2)
	yD := float32(h / 2)

	// Calculate cosine and sine of angle
	aCos := float32(math.Cos(float64(angle)))
	aSin := float32(math.Sin(float64(angle)))

	// Define the initial corners (relative to center)
	cornersX := [4]float32{-xD, -xD, xD, xD}
	cornersY := [4]float32{-yD, yD, yD, -yD}

	// Calculate the rotated corners
	var outCorners [8]int

	for i := 0; i < 4; i++ {
		outCorners[2*i] = int(aCos*cornersX[i] - aSin*cornersY[i] + cx)   // X-coordinate
		outCorners[2*i+1] = int(aSin*cornersX[i] + aCos*cornersY[i] + cy) // Y-coordinate
	}

	return outCorners
}

// calculateCornerTextAlignment analyzes the corners of the oriented bounding box
// and returns the left-most X, center X, right-most X, and top Y coordinates
func calculateCornerTextAlignment(corners [8]int) (int, int, int, int) {

	// Initialize minX and maxX to the first corner's X value,
	// minY to the first corner's Y value
	minX := corners[0]
	maxX := corners[0]
	minY := corners[1]

	// Iterate through the remaining corners
	for i := 0; i < 4; i++ {
		x := corners[2*i]
		y := corners[2*i+1]

		// Update min and max X
		if x < minX {
			minX = x
		}
		if x > maxX {
			maxX = x
		}

		// Update min Y
		if y < minY {
			minY = y
		}
	}

	// Calculate the center X as the average of minX and maxX
	centerX := float64(minX+maxX) / 2.0

	// Top Y is the minimum Y value
	topY := minY

	return minX, int(centerX), maxX, topY
}
