package render

import (
	"github.com/swdee/go-rknnlite/tracker"
	"gocv.io/x/gocv"
	"image"
	"image/color"
)

// TrailStyle defines the parameters used for rendering the trail style
type TrailStyle struct {
	// LineSame defines if the color of the trail line should be the
	// same color as that of the bounding box.  If set to false then use
	// the color specified at LineColor
	LineSame      bool
	LineColor     color.RGBA
	LineThickness int
	// CircleSame defines if the color of the midpoint circle should be the
	// same color as that of the bounding box.  If set to false then use
	// the color specified at CircleColor
	CircleSame   bool
	CircleColor  color.RGBA
	CircleRadius int
}

// DefaultTrailStyle returns default trail style settings
func DefaultTrailStyle() TrailStyle {
	return TrailStyle{
		LineSame:      false,
		LineColor:     Yellow,
		LineThickness: 1,
		CircleSame:    true,
		CircleColor:   Pink,
		CircleRadius:  3,
	}
}

// Trail draws the tracker trail lines on the source image.
func Trail(img *gocv.Mat, trackResults []*tracker.STrack,
	trail *tracker.Trail, style TrailStyle) {

	// draw trail
	for _, tResult := range trackResults {

		// Get the color for this object
		colorIndex := tResult.GetTrackID() % len(classColors)
		objClr := classColors[colorIndex]

		// determine style colors to use
		lineClr := objClr
		circleClr := objClr

		if !style.LineSame {
			lineClr = style.LineColor
		}

		if !style.CircleSame {
			circleClr = style.CircleColor
		}

		// draw trail line showing tracking history
		points := trail.GetPoints(tResult.GetTrackID())

		if len(points) > 2 {
			// draw trail
			for i := 1; i < len(points); i++ {
				// draw line segment of trail
				gocv.Line(img,
					image.Pt(points[i-1].X, points[i-1].Y),
					image.Pt(points[i].X, points[i].Y),
					lineClr, style.LineThickness,
				)

				if i == len(points)-1 {
					// draw center point circle on current rect/box
					gocv.Circle(img, image.Pt(points[i].X, points[i].Y),
						style.CircleRadius, circleClr, -1)
				}
			}
		}
	}
}
