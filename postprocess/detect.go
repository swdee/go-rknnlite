package postprocess

type DetectionResult interface {
	GetDetectResults() []DetectResult
}

// BoxRectMode indicates how the BoxRect was set
type BoxRectMode int

const (
	ModeLTRB BoxRectMode = 0 // (Default) Left, Top, Right, Bottom mode
	ModeXYWH BoxRectMode = 1 // X, Y, Width, Height, Angle mode
)

// BoxRect are the dimensions of the bounding box of a detect object
type BoxRect struct {
	Left   int // Left boundary of the bounding box
	Right  int // Right boundary of the bounding box
	Top    int // Top boundary of the bounding box
	Bottom int // Bottom boundary of the bounding box

	X      int     // X coordinate of the bounding box center
	Y      int     // Y coordinate of the bounding box center
	Width  int     // Width of the bounding box
	Height int     // Height of the bounding box
	Angle  float32 // Rotation angle of the bounding box in radians

	Mode BoxRectMode // Mode indicates how the BoxRect was set
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

// Keypoint is used for specifying the X, Y coordinates and confidence score
// of an individual point used in Pose Estimatation
type KeyPoint struct {
	X     int
	Y     int
	Score float32
}
