package tracker

import (
	"fmt"
	"math"
)

// BYTETracker represents the BYTE Tracker
type BYTETracker struct {
	// Threshold for tracking objects
	trackThresh float32
	// High threshold for tracking objects
	highThresh float32
	// Matching threshold for associations
	matchThresh float32
	// Maximum time an object can be lost before being remove
	maxTimeLost int
	// Current frame ID
	frameID int
	// Counter for assigning unique track IDs
	trackIDCount int
	// List of currently tracked objects
	trackedStracks []*STrack
	// List of lost objects
	lostStracks []*STrack
	// List of removed objects
	removedStracks []*STrack
}

// NewBYTETracker initializes and returns a new BYTETracker
func NewBYTETracker(frameRate int, trackBuffer int, trackThresh float32,
	highThresh float32, matchThresh float32) *BYTETracker {

	return &BYTETracker{
		trackThresh: trackThresh,
		highThresh:  highThresh,
		matchThresh: matchThresh,
		maxTimeLost: int(float32(frameRate) / 30.0 * float32(trackBuffer)),
	}
}

// Reset clears the tracked data and resets everything
func (bt *BYTETracker) Reset() {
	bt.frameID = 0
	bt.trackIDCount = 0
	bt.trackedStracks = make([]*STrack, 0)
	bt.lostStracks = make([]*STrack, 0)
	bt.removedStracks = make([]*STrack, 0)
}

// Update updates the tracker with new detections
func (bt *BYTETracker) Update(objects []Object) ([]*STrack, error) {

	// Step 1: Get detections
	bt.frameID++

	// create new STracks using the result of object detection
	var detStracks, detLowStracks []*STrack

	for _, object := range objects {

		strack := NewSTrack(NewRect(object.Rect.X(), object.Rect.Y(), object.Rect.Width(), object.Rect.Height()),
			object.Prob, object.ID, object.Label)

		if object.Prob >= bt.trackThresh {
			detStracks = append(detStracks, strack)
		} else {
			detLowStracks = append(detLowStracks, strack)
		}
	}

	// create lists of existing STrack
	var activeStracks, nonActiveStracks, strackPool []*STrack

	for _, trackedStrack := range bt.trackedStracks {
		if !trackedStrack.IsActivated() {
			nonActiveStracks = append(nonActiveStracks, trackedStrack)
		} else {
			activeStracks = append(activeStracks, trackedStrack)
		}
	}

	strackPool = bt.jointStracks(activeStracks, bt.lostStracks)

	// predict current pose by KF
	for _, strack := range strackPool {
		strack.Predict()
	}

	// Step 2: First association, with IoU
	var currentTrackedStracks, remainTrackedStracks, remainDetStracks, refindStracks []*STrack

	matchesIdx, unmatchTrackIdx, unmatchDetectionIdx, err := bt.linearAssignment(
		bt.calcIouDistance(strackPool, detStracks),
		len(strackPool), len(detStracks), bt.matchThresh,
	)

	if err != nil {
		return nil, fmt.Errorf("fatal error in linearAssignment call, step 2: %w", err)
	}

	for _, matchIdx := range matchesIdx {

		track := strackPool[matchIdx[0]]
		det := detStracks[matchIdx[1]]

		if track.GetSTrackState() == Tracked {
			err := track.Update(det, bt.frameID)
			if err != nil {
				return nil, fmt.Errorf("error updating track, step 2: %w", err)
			}
			currentTrackedStracks = append(currentTrackedStracks, track)
		} else {
			track.ReActivate(det, bt.frameID, -1) // Providing the default new track ID
			refindStracks = append(refindStracks, track)
		}
	}

	for _, unmatchIdx := range unmatchDetectionIdx {
		remainDetStracks = append(remainDetStracks, detStracks[unmatchIdx])
	}

	for _, unmatchIdx := range unmatchTrackIdx {
		if strackPool[unmatchIdx].GetSTrackState() == Tracked {
			remainTrackedStracks = append(remainTrackedStracks, strackPool[unmatchIdx])
		}
	}

	// Step 3: Second association, using low score dets
	var currentLostStracks []*STrack

	matchesIdx, unmatchTrackIdx, unmatchDetectionIdx, err = bt.linearAssignment(
		bt.calcIouDistance(remainTrackedStracks, detLowStracks),
		len(remainTrackedStracks), len(detLowStracks), 0.5,
	)

	if err != nil {
		return nil, fmt.Errorf("fatal error in linearAssignment call, step 3: %w", err)
	}

	for _, matchIdx := range matchesIdx {
		track := remainTrackedStracks[matchIdx[0]]
		det := detLowStracks[matchIdx[1]]
		if track.GetSTrackState() == Tracked {
			err := track.Update(det, bt.frameID)
			if err != nil {
				return nil, fmt.Errorf("error updating track, step 3: %w", err)
			}
			currentTrackedStracks = append(currentTrackedStracks, track)
		} else {
			track.ReActivate(det, bt.frameID, -1) // Providing the default new track ID
			refindStracks = append(refindStracks, track)
		}
	}

	for _, unmatchTrack := range unmatchTrackIdx {
		track := remainTrackedStracks[unmatchTrack]
		if track.GetSTrackState() != Lost {
			track.MarkAsLost()
			currentLostStracks = append(currentLostStracks, track)
		}
	}

	// Step 4: Init new stracks
	var currentRemovedStracks []*STrack

	matchesIdx, unmatchUnconfirmedIdx, unmatchDetectionIdx, err := bt.linearAssignment(
		bt.calcIouDistance(nonActiveStracks, remainDetStracks),
		len(nonActiveStracks), len(remainDetStracks), 0.7,
	)

	if err != nil {
		return nil, fmt.Errorf("fatal error in linearAssignment call, step 4: %w", err)
	}

	for _, matchIdx := range matchesIdx {
		err := nonActiveStracks[matchIdx[0]].Update(remainDetStracks[matchIdx[1]], bt.frameID)
		if err != nil {
			return nil, fmt.Errorf("error updating track, step 4: %w", err)
		}
		currentTrackedStracks = append(currentTrackedStracks, nonActiveStracks[matchIdx[0]])
	}

	for _, unmatchIdx := range unmatchUnconfirmedIdx {
		track := nonActiveStracks[unmatchIdx]
		track.MarkAsRemoved()
		currentRemovedStracks = append(currentRemovedStracks, track)
	}

	for _, unmatchIdx := range unmatchDetectionIdx {
		track := remainDetStracks[unmatchIdx]
		if track.GetScore() < bt.highThresh {
			continue
		}
		bt.trackIDCount++
		track.Activate(bt.frameID, bt.trackIDCount)
		currentTrackedStracks = append(currentTrackedStracks, track)
	}

	// Step 5: Update state
	for _, lostStrack := range bt.lostStracks {
		if bt.frameID-lostStrack.GetFrameID() > bt.maxTimeLost {
			lostStrack.MarkAsRemoved()
			currentRemovedStracks = append(currentRemovedStracks, lostStrack)
		}
	}

	bt.trackedStracks = bt.jointStracks(currentTrackedStracks, refindStracks)
	bt.lostStracks = bt.subStracks(bt.jointStracks(bt.subStracks(bt.lostStracks, bt.trackedStracks), currentLostStracks), bt.removedStracks)
	bt.removedStracks = bt.jointStracks(bt.removedStracks, currentRemovedStracks)

	var trackedStracksOut, lostStracksOut []*STrack
	bt.removeDuplicateStracks(bt.trackedStracks, bt.lostStracks, &trackedStracksOut, &lostStracksOut)
	bt.trackedStracks = trackedStracksOut
	bt.lostStracks = lostStracksOut

	var outputStracks []*STrack
	for _, track := range bt.trackedStracks {
		if track.IsActivated() {
			outputStracks = append(outputStracks, track)
		}
	}

	return outputStracks, nil
}

// jointStracks combines two lists of tracks, avoiding duplicates
func (bt *BYTETracker) jointStracks(aTlist []*STrack, bTlist []*STrack) []*STrack {

	// create a map to track the existence of track IDs
	exists := make(map[int]bool)
	var res []*STrack

	// add all tracks from aTlist to the result list and mark their IDs as existing
	for _, track := range aTlist {
		exists[track.GetTrackID()] = true
		res = append(res, track)
	}

	// add tracks from bTlist to the result list if their IDs do not already exist
	for _, track := range bTlist {
		tid := track.GetTrackID()

		if !exists[tid] {
			exists[tid] = true
			res = append(res, track)
		}
	}

	return res
}

// subStracks subtracts bTlist from aTlist and returns the result
func (bt *BYTETracker) subStracks(aTlist []*STrack, bTlist []*STrack) []*STrack {
	stracks := make(map[int]*STrack)
	for _, track := range aTlist {
		stracks[track.GetTrackID()] = track
	}
	for _, track := range bTlist {
		delete(stracks, track.GetTrackID())
	}
	var res []*STrack
	for _, track := range stracks {
		res = append(res, track)
	}
	return res
}

// removeDuplicateStracks removes duplicate tracks
func (bt *BYTETracker) removeDuplicateStracks(aStracks []*STrack, bStracks []*STrack, aRes *[]*STrack, bRes *[]*STrack) {
	ious := bt.calcIouDistance(aStracks, bStracks)
	overlappingCombinations := [][2]int{}
	for i := range ious {
		for j := range ious[i] {
			if ious[i][j] < 0.15 {
				overlappingCombinations = append(overlappingCombinations, [2]int{i, j})
			}
		}
	}
	aOverlapping := make([]bool, len(aStracks))
	bOverlapping := make([]bool, len(bStracks))
	for _, combo := range overlappingCombinations {
		timep := aStracks[combo[0]].GetFrameID() - aStracks[combo[0]].GetStartFrameID()
		timeq := bStracks[combo[1]].GetFrameID() - bStracks[combo[1]].GetStartFrameID()
		if timep > timeq {
			bOverlapping[combo[1]] = true
		} else {
			aOverlapping[combo[0]] = true
		}
	}
	for i, overlapping := range aOverlapping {
		if !overlapping {
			*aRes = append(*aRes, aStracks[i])
		}
	}
	for i, overlapping := range bOverlapping {
		if !overlapping {
			*bRes = append(*bRes, bStracks[i])
		}
	}
}

// linearAssignment performs linear assignment using the Hungarian algorithm
func (bt *BYTETracker) linearAssignment(costMatrix [][]float32, costMatrixSize,
	costMatrixSizeSize int, thresh float32) (matchesIdx [][2]int,
	unmatchTrackIdx, unmatchDetectionIdx []int, fatalErr error) {

	if len(costMatrix) == 0 {
		for i := 0; i < costMatrixSize; i++ {
			unmatchTrackIdx = append(unmatchTrackIdx, i)
		}
		for i := 0; i < costMatrixSizeSize; i++ {
			unmatchDetectionIdx = append(unmatchDetectionIdx, i)
		}
		return
	}

	rowsol, colsol, _, fatalErr := bt.execLapjv(costMatrix, true, thresh)

	if fatalErr != nil {
		return
	}

	for i, sol := range rowsol {
		if sol >= 0 {
			matchesIdx = append(matchesIdx, [2]int{i, sol})
		} else {
			unmatchTrackIdx = append(unmatchTrackIdx, i)
		}
	}
	for i, sol := range colsol {
		if sol < 0 {
			unmatchDetectionIdx = append(unmatchDetectionIdx, i)
		}
	}

	return
}

// calcIous calculates the Intersection over Union (IoU) between two sets of rectangles
func (bt *BYTETracker) calcIous(aRects, bRects []Rect) [][]float32 {

	var ious [][]float32
	if len(aRects)*len(bRects) == 0 {
		return ious
	}

	ious = make([][]float32, len(aRects))
	for i := range ious {
		ious[i] = make([]float32, len(bRects))
	}

	for bi := range bRects {
		for ai := range aRects {
			ious[ai][bi] = bRects[bi].CalcIoU(aRects[ai])
		}
	}
	return ious
}

// calcIouDistance calculates the IoU distance between two sets of tracks
func (bt *BYTETracker) calcIouDistance(aTracks, bTracks []*STrack) [][]float32 {

	var aRects, bRects []Rect
	for _, track := range aTracks {
		aRects = append(aRects, *track.GetRect())
	}

	for _, track := range bTracks {
		bRects = append(bRects, *track.GetRect())
	}

	ious := bt.calcIous(aRects, bRects)

	var costMatrix [][]float32

	for _, iouRow := range ious {
		var iou []float32

		for _, iouValue := range iouRow {
			iou = append(iou, 1-iouValue)
		}

		costMatrix = append(costMatrix, iou)
	}

	return costMatrix
}

// execLapjv executes the LAPJV algorithm for linear assignment problem solving
func (bt *BYTETracker) execLapjv(cost [][]float32, extendCost bool,
	costLimit float32) (rowsol []int, colsol []int, opt float64, err error) {

	// Default value for returnCost. This was a parameter passed to the function
	// but was unused
	returnCost := false

	// Copy cost matrix
	costC := make([][]float32, len(cost))
	for i := range cost {
		costC[i] = make([]float32, len(cost[i]))
		copy(costC[i], cost[i])
	}

	nRows := len(cost)
	nCols := len(cost[0])
	rowsol = make([]int, nRows)
	colsol = make([]int, nCols)

	n := 0
	if nRows == nCols {
		n = nRows
	} else {
		if !extendCost {
			return nil, nil, 0, fmt.Errorf("The `extend_cost` variable should be set to True")
		}
	}

	if extendCost || costLimit < float32(math.MaxFloat32) {
		n = nRows + nCols
		costCExtended := make([][]float32, n)
		for i := range costCExtended {
			costCExtended[i] = make([]float32, n)
		}

		if costLimit < float32(math.MaxFloat32) {
			for i := range costCExtended {
				for j := range costCExtended[i] {
					costCExtended[i][j] = costLimit / 2.0
				}
			}
		} else {
			costMax := float32(-1)
			for i := range costC {
				for j := range costC[i] {
					if costC[i][j] > costMax {
						costMax = costC[i][j]
					}
				}
			}
			for i := range costCExtended {
				for j := range costCExtended[i] {
					costCExtended[i][j] = costMax + 1
				}
			}
		}

		for i := nRows; i < len(costCExtended); i++ {
			for j := nCols; j < len(costCExtended[i]); j++ {
				costCExtended[i][j] = 0
			}
		}
		for i := 0; i < nRows; i++ {
			for j := 0; j < nCols; j++ {
				costCExtended[i][j] = costC[i][j]
			}
		}

		costC = costCExtended
	}

	costPtr := make([][]float64, n)
	for i := range costPtr {
		costPtr[i] = make([]float64, n)
		for j := range costPtr[i] {
			costPtr[i][j] = float64(costC[i][j])
		}
	}

	xC := make([]int, n)
	yC := make([]int, n)

	ret, err := lapjvInternal(n, costPtr, xC, yC)
	if ret != 0 || err != nil {
		return nil, nil, 0, fmt.Errorf("The result of lapjvInternal() is invalid: %w", err)
	}

	opt = 0.0

	if n != nRows {
		for i := 0; i < n; i++ {
			if xC[i] >= nCols {
				xC[i] = -1
			}
			if yC[i] >= nRows {
				yC[i] = -1
			}
		}
		for i := 0; i < nRows; i++ {
			rowsol[i] = xC[i]
		}
		for i := 0; i < nCols; i++ {
			colsol[i] = yC[i]
		}

		if returnCost {
			for i := 0; i < len(rowsol); i++ {
				if rowsol[i] != -1 {
					opt += costPtr[i][rowsol[i]]
				}
			}
		}
	} else if returnCost {
		for i := 0; i < len(rowsol); i++ {
			opt += costPtr[i][rowsol[i]]
		}
	}

	return rowsol, colsol, opt, nil
}
