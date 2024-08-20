package tracker

import "sync"

// Point represents the x,y coordinates of the center box of a tracking
// rect/bounding box results
type Point struct {
	X, Y int
}

// Track represents a track history
type Track struct {
	points []Point
}

// Trail is the struct to keep a history of Track results used for drawing
// a trail
type Trail struct {
	// size is the maximum number of most recent points to keep in history
	size int
	// history of tracked points
	history map[int]*Track
	sync.Mutex
}

// NewTrail returns a new trail history track instance.  Size is the number
// of most recent trails to keep and specifies the maximum length of the trail
// to maintain
func NewTrail(size int) *Trail {
	return &Trail{
		size:    size,
		history: make(map[int]*Track),
	}
}

// Reset clears all history
func (t *Trail) Reset() {
	t.Lock()
	defer t.Unlock()

	t.history = make(map[int]*Track)
}

// Add a track to the history
func (t *Trail) Add(strack *STrack) {
	t.Lock()
	defer t.Unlock()

	// init map if no history exists yet for track id
	if _, exists := t.history[strack.GetTrackID()]; !exists {
		t.history[strack.GetTrackID()] = &Track{}
	}

	// add bounding box/rect's center point to track history
	track := t.history[strack.GetTrackID()]

	// find center point
	x := (strack.GetRect().TLX() +
		(strack.GetRect().Width() / 2))

	y := (strack.GetRect().TLY() +
		(strack.GetRect().Height() / 2))

	track.points = append(track.points, Point{
		X: int(x),
		Y: int(y),
	})

	// check if history is exceeded and drop oldest point
	if len(track.points) > t.size {
		track.points = track.points[1:]
	}
}

// GetPoints gets the point history for a specific track id
func (t *Trail) GetPoints(id int) []Point {
	t.Lock()
	defer t.Unlock()

	if _, exists := t.history[id]; exists {
		return t.history[id].points
	}

	// no history yet
	return nil
}
