package render

import (
	"fmt"
	"github.com/swdee/go-rknnlite/postprocess"
	"github.com/swdee/go-rknnlite/tracker"
	"gocv.io/x/gocv"
	"image"
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
