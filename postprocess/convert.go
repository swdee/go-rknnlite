package postprocess

import "github.com/swdee/go-rknnlite/tracker"

// DetectionsToObjects takes a postprocess object detection results and
// converts it into a tracker object
func DetectionsToObjects(dets []DetectResult) []tracker.Object {

	var objs []tracker.Object
	var x, y, width, height float32

	for _, det := range dets {

		if det.Box.Mode == ModeXYWH {
			x = float32(det.Box.X)
			y = float32(det.Box.Y)
			width = float32(det.Box.Width)
			height = float32(det.Box.Height)

		} else {
			x = float32(det.Box.Left)
			y = float32(det.Box.Top)
			width = float32(det.Box.Right - det.Box.Left)
			height = float32(det.Box.Bottom - det.Box.Top)
		}

		objs = append(objs, tracker.Object{
			Rect:  tracker.NewRect(float32(x), float32(y), float32(width), float32(height)),
			Label: det.Class,
			Prob:  det.Probability,
			ID:    det.ID,
		})
	}

	return objs
}
