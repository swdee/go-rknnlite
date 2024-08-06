package tracker

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

// STrackState represents the state of a tracked object
type STrackState int

const (
	// Object is newly detected
	New STrackState = 0
	// Object is currently being tracked
	Tracked STrackState = 1
	// Object has been lost
	Lost STrackState = 2
	// Object has been removed
	Removed STrackState = 3
)

// STrack represents a single track of an object
type STrack struct {
	// Kalman filter used for tracking
	kalmanFilter *KalmanFilter
	// Mean state vector
	mean StateMean
	// Covariance matrix
	covariance StateCov
	// Bounding box of the tracked object
	rect Rect
	// Current state of the track
	state STrackState
	// Whether the track is activated
	isActivated bool
	// Detection score
	score float32
	// Unique ID for the track
	trackID int
	// Current frame ID
	frameID int
	// Frame ID when the track started
	startFrameID int
	// Length of the tracklet
	trackletLen int
	// Unique ID for the detection
	detectionID int64
	// label is the object label/class from yolo inference
	label int
}

// NewSTrack creates a new STrack
func NewSTrack(rect Rect, score float32, detectionID int64, label int) *STrack {
	return &STrack{
		kalmanFilter: NewKalmanFilter(1.0/20, 1.0/160),
		mean:         make(StateMean, 8),
		covariance:   StateCov{mat.NewDense(8, 8, nil)},
		rect:         rect,
		state:        New,
		isActivated:  false,
		score:        score,
		trackID:      0,
		frameID:      0,
		startFrameID: 0,
		trackletLen:  0,
		detectionID:  detectionID,
		label:        label,
	}
}

// GetRect returns the bounding box of the tracked object
func (s *STrack) GetRect() *Rect {
	return &s.rect
}

// GetSTrackState returns the current state of the track
func (s *STrack) GetSTrackState() STrackState {
	return s.state
}

// IsActivated returns whether the track is activated
func (s *STrack) IsActivated() bool {
	return s.isActivated
}

// GetScore returns the detection score
func (s *STrack) GetScore() float32 {
	return s.score
}

// GetTrackID returns the unique ID for the track
func (s *STrack) GetTrackID() int {
	return s.trackID
}

// GetFrameID returns the current frame ID
func (s *STrack) GetFrameID() int {
	return s.frameID
}

// GetDetectionID returns the unique ID for the detection
func (s *STrack) GetDetectionID() int64 {
	return s.detectionID
}

// GetLabel returns the object label/class from YOLO inference
func (s *STrack) GetLabel() int {
	return s.label
}

// GetStartFrameID returns the frame ID when the track started
func (s *STrack) GetStartFrameID() int {
	return s.startFrameID
}

// GetTrackletLength returns the length of the tracklet
func (s *STrack) GetTrackletLength() int {
	return s.trackletLen
}

// Activate initializes the track with the given frame ID and track ID
func (s *STrack) Activate(frameID, trackID int) {

	s.kalmanFilter.Initiate(s.mean, &s.covariance, DetectBox(s.rect.GetXyah()))

	s.updateRect()

	s.state = Tracked

	if frameID == 1 {
		s.isActivated = true
	}

	s.trackID = trackID
	s.frameID = frameID
	s.startFrameID = frameID
	s.trackletLen = 0
}

// ReActivate reinitializes the track with a new detection
func (s *STrack) ReActivate(newTrack *STrack, frameID, newTrackID int) {

	s.kalmanFilter.Update(s.mean, &s.covariance, DetectBox(newTrack.GetRect().GetXyah()))

	s.updateRect()

	s.state = Tracked
	s.isActivated = true
	s.score = newTrack.GetScore()
	s.detectionID = newTrack.GetDetectionID()

	if newTrackID >= 0 {
		s.trackID = newTrackID
	}

	s.frameID = frameID
	s.trackletLen = 0
}

// Predict predicts the next state of the track
func (s *STrack) Predict() {
	if s.state != Tracked {
		s.mean[7] = 0
	}

	s.kalmanFilter.Predict(s.mean, &s.covariance)
}

// Update updates the track with a new detection
func (s *STrack) Update(newTrack *STrack, frameID int) error {

	err := s.kalmanFilter.Update(s.mean, &s.covariance,
		DetectBox(newTrack.GetRect().GetXyah()))

	if err != nil {
		return fmt.Errorf("error updating: %w", err)
	}

	s.updateRect()

	s.state = Tracked
	s.isActivated = true
	s.score = newTrack.GetScore()
	s.detectionID = newTrack.GetDetectionID()
	s.frameID = frameID
	s.trackletLen++

	return nil
}

// MarkAsLost marks the track as lost
func (s *STrack) MarkAsLost() {
	s.state = Lost
}

// MarkAsRemoved marks the track as removed
func (s *STrack) MarkAsRemoved() {
	s.state = Removed
}

// updateRect updates the bounding box of the tracked object based on the state mean.
func (s *STrack) updateRect() {
	s.rect.SetWidth(s.mean[2] * s.mean[3])
	s.rect.SetHeight(s.mean[3])
	s.rect.SetX(s.mean[0] - s.rect.Width()/2)
	s.rect.SetY(s.mean[1] - s.rect.Height()/2)
}
