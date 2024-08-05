package tracker

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

type STrackState int

const (
	New     STrackState = 0
	Tracked STrackState = 1
	Lost    STrackState = 2
	Removed STrackState = 3
)

type STrack struct {
	kalmanFilter *KalmanFilter
	mean         StateMean
	covariance   StateCov
	rect         Rect
	state        STrackState
	isActivated  bool
	score        float32
	trackID      int
	frameID      int
	startFrameID int
	trackletLen  int
	detectionID  int64
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

func (s *STrack) GetRect() *Rect {
	return &s.rect
}

func (s *STrack) GetSTrackState() STrackState {
	return s.state
}

func (s *STrack) IsActivated() bool {
	return s.isActivated
}

func (s *STrack) GetScore() float32 {
	return s.score
}

func (s *STrack) GetTrackID() int {
	return s.trackID
}

func (s *STrack) GetFrameID() int {
	return s.frameID
}

func (s *STrack) GetDetectionID() int64 {
	return s.detectionID
}

func (s *STrack) GetLabel() int {
	return s.label
}

func (s *STrack) GetStartFrameID() int {
	return s.startFrameID
}

func (s *STrack) GetTrackletLength() int {
	return s.trackletLen
}

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

func (s *STrack) Predict() {
	if s.state != Tracked {
		s.mean[7] = 0
	}

	s.kalmanFilter.Predict(s.mean, &s.covariance)
}

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

func (s *STrack) MarkAsLost() {
	s.state = Lost
}

func (s *STrack) MarkAsRemoved() {
	s.state = Removed
}

func (s *STrack) updateRect() {
	s.rect.SetWidth(s.mean[2] * s.mean[3])
	s.rect.SetHeight(s.mean[3])
	s.rect.SetX(s.mean[0] - s.rect.Width()/2)
	s.rect.SetY(s.mean[1] - s.rect.Height()/2)
}
