package tracker

import "github.com/swdee/go-rknnlite/postprocess"

// DetectionsToObjects takes a postprocess object detection results and
// converts it into a tracker object
func DetectionsToObjects(dets []postprocess.DetectResult) []Object {
	var objs []Object

	for _, det := range dets {

		x := det.Box.Left
		y := det.Box.Top
		width := det.Box.Right - det.Box.Left
		height := det.Box.Bottom - det.Box.Top

		objs = append(objs, Object{
			Rect:  NewRect(float32(x), float32(y), float32(width), float32(height)),
			Label: det.Class,
			Prob:  det.Probability,
			ID:    det.ID,
		})
	}

	return objs
}
