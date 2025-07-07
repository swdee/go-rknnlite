package tracker

// Object represents an object detected in ByteTrack
type Object struct {
	// Rect is the bounding box representation of the detected object
	Rect Rect
	// Label is the class label of the object detected
	Label int
	// Prob is the confidence/probability of the object detected
	Prob float32
	// ID is a unique ID to give this object which can be used to match
	// the input detection object and tracked object
	ID int64
	// Feature is a ReID embedding feature
	Feature []float32
}

// NewObject is a constructor function for the Object struct
func NewObject(rect Rect, label int, prob float32, id int64) Object {
	return Object{
		Rect:  rect,
		Label: label,
		Prob:  prob,
		ID:    id,
	}
}
