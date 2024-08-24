package postprocess

type DetectionResult interface {
	GetDetectResults() []DetectResult
}

// BoxRect are the dimensions of the bounding box of a detect object
type BoxRect struct {
	Left   int
	Right  int
	Top    int
	Bottom int
}

// DetectResult defines the attributes of a single object detected
type DetectResult struct {
	// Class is the line number in the labels file the Model was trained on
	// defining the Class of the detected object
	Class int
	// Box are the bounding box dimensions of the object location
	Box BoxRect
	// Probability is the confidence score of the object detected
	Probability float32
	// ID is a unique ID assigned to the detection result
	ID int64
}
