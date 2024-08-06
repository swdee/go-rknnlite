package tracker

import (
	"math"
	"testing"
)

// convertDetections takes the matrix of YOLO object detections and converts
// them into the bytetracker object format
func convertDetections(detections []detection, useLabel int) []Object {
	var objects []Object

	for _, det := range detections {

		x := det.x1
		y := det.y1
		width := det.x2 - det.x1
		height := det.y2 - det.y1
		score := det.score
		id := det.detectionID

		objects = append(objects, Object{
			Rect:  NewRect(x, y, width, height),
			Label: useLabel,
			Prob:  score,
			ID:    id,
		})
	}

	return objects
}

// almostEqual checks if two float32 values are approximately equal
func almostEqual(a, b, tolerance float32) bool {
	return float32(math.Abs(float64(a)-float64(b))) <= tolerance
}

// detectionFrame holds detection data for a single frame.
type detectionFrame struct {
	frameIdx       int
	detections     []detection
	expectedTracks []struct {
		trackID     int
		tlx, tly    float32
		brx, bry    float32
		prob        float32
		detectionID int64
	}
}

// Detection represents the detection data from YOLO model
type detection struct {
	x1, y1, x2, y2, score float32
	detectionID           int64
}

// TestBYTETracker tests tracker result from given detection inputs
func TestBYTETracker(t *testing.T) {

	// tolerance for float comparisons
	const tolerance = 1e-2

	// Initialize the BYTETracker
	bt := NewBYTETracker(30, 30, 0.5, 0.6, 0.8)

	// Define detection data and expected outputs for each frame
	frames := []detectionFrame{
		{
			frameIdx: 0,
			detections: []detection{
				{79, 205, 169, 609, 85.10, 1},
				{196, 222, 258, 451, 83.98, 2},
				{270, 247, 331, 456, 82.81, 3},
				{471, 205, 584, 638, 82.61, 4},
				{158, 302, 201, 506, 78.12, 5},
				{328, 234, 381, 445, 76.65, 6},
				{364, 218, 434, 450, 76.12, 7},
				{347, 148, 378, 238, 46.30, 8},
				{296, 184, 342, 408, 43.97, 9},
				{132, 201, 176, 319, 41.19, 10},
				{69, 191, 120, 391, 31.02, 11},
				{627, 237, 640, 284, 24.46, 12},
			},
			expectedTracks: []struct {
				trackID     int
				tlx, tly    float32
				brx, bry    float32
				prob        float32
				detectionID int64
			}{
				{1, 79.00000, 205.00000, 169.00000, 609.00000, 85.10, 1},
				{2, 196.00000, 222.00000, 258.00000, 451.00000, 83.98, 2},
				{3, 270.00000, 247.00000, 331.00000, 456.00000, 82.81, 3},
				{4, 471.00000, 205.00000, 584.00000, 638.00000, 82.61, 4},
				{5, 158.00000, 302.00000, 201.00000, 506.00000, 78.12, 5},
				{6, 328.00000, 234.00000, 381.00000, 445.00000, 76.65, 6},
				{7, 364.00000, 218.00000, 434.00000, 450.00000, 76.12, 7},
				{8, 347.00000, 148.00000, 378.00000, 238.00000, 46.30, 8},
				{9, 296.00000, 184.00000, 342.00000, 408.00000, 43.97, 9},
				{10, 132.00000, 201.00000, 176.00000, 319.00000, 41.19, 10},
				{11, 69.00000, 191.00000, 120.00000, 391.00000, 31.02, 11},
				{12, 627.00000, 237.00000, 640.00000, 284.00000, 24.46, 12},
			},
		},
		{
			frameIdx: 1,
			detections: []detection{
				{471, 212, 584, 633, 83.76, 13},
				{197, 219, 259, 453, 83.59, 14},
				{271, 242, 331, 457, 81.64, 15},
				{83, 220, 166, 610, 78.91, 16},
				{157, 303, 204, 502, 77.43, 17},
				{364, 218, 434, 450, 74.97, 18},
				{327, 232, 383, 446, 73.54, 19},
				{346, 149, 377, 238, 50.58, 20},
				{70, 181, 125, 397, 43.71, 21},
				{297, 185, 343, 416, 42.02, 22},
				{133, 206, 178, 319, 37.11, 23},
				{589, 280, 639, 554, 34.46, 24},
			},
			expectedTracks: []struct {
				trackID     int
				tlx, tly    float32
				brx, bry    float32
				prob        float32
				detectionID int64
			}{
				{1, 80.82532, 218.01653, 168.04245, 609.86774, 78.91, 16},
				{2, 196.29364, 219.39668, 259.44189, 452.73553, 83.59, 14},
				{3, 269.70096, 242.66116, 332.16684, 456.86777, 81.64, 15},
				{4, 472.32794, 211.07437, 582.67206, 633.66113, 83.76, 13},
				{5, 159.27533, 302.86774, 201.46021, 502.52890, 77.43, 17},
				{6, 328.08496, 232.26445, 381.78284, 445.86774, 73.54, 19},
				{7, 364.00000, 218.00000, 434.00000, 450.00000, 74.97, 18},
				{8, 346.27829, 148.86777, 376.98615, 238.00000, 50.58, 20},
				{9, 296.25809, 184.86777, 343.47742, 414.94214, 42.02, 22},
				{10, 134.08234, 205.33885, 176.52097, 319.00000, 37.11, 23},
				{11, 69.83383, 182.32233, 124.37277, 396.20660, 43.71, 21},
			},
		},
		{
			frameIdx: 2,
			detections: []detection{
				{472, 204, 584, 637, 85.21, 25},
				{199, 221, 260, 450, 81.64, 26},
				{158, 303, 205, 502, 78.59, 27},
				{84, 228, 167, 609, 77.73, 28},
				{269, 240, 332, 458, 77.34, 29},
				{363, 218, 433, 450, 75.57, 30},
				{329, 233, 381, 445, 73.63, 31},
				{139, 206, 179, 321, 46.31, 32},
				{78, 181, 134, 385, 44.66, 33},
				{296, 185, 346, 411, 42.80, 34},
				{589, 263, 640, 571, 38.81, 35},
				{346, 149, 377, 236, 33.45, 36},
			},
			expectedTracks: []struct {
				trackID     int
				tlx, tly    float32
				brx, bry    float32
				prob        float32
				detectionID int64
			}{
				{1, 82.73601, 226.55103, 167.85870, 609.22614, 77.73, 28},
				{2, 198.03925, 220.49619, 260.31458, 450.71359, 81.64, 26},
				{3, 268.93002, 240.37213, 332.31552, 457.78845, 77.34, 29},
				{4, 471.73502, 205.81052, 584.05249, 636.07104, 85.21, 25},
				{5, 160.21704, 303.01587, 202.38788, 501.93652, 78.59, 27},
				{6, 328.31689, 232.74213, 381.69986, 445.24115, 73.63, 31},
				{7, 363.22046, 218.00000, 433.22046, 450.00000, 75.57, 30},
				{8, 346.41031, 149.01614, 376.55737, 236.43452, 33.45, 36},
				{9, 297.41403, 185.01706, 344.16153, 412.28278, 42.80, 34},
				{10, 136.95946, 206.07745, 179.62943, 320.58356, 46.31, 32},
				{11, 77.51575, 180.81949, 130.46681, 388.02057, 44.66, 33},
				{13, 586.94348, 266.39999, 641.85657, 567.59998, 38.81, 35},
			},
		},
	}

	// Process each frame's detections
	for _, frame := range frames {

		// hard coded classification label for now as data was restricted
		// to the same class (person)
		objects := convertDetections(frame.detections, 0)

		trackedObjects, err := bt.Update(objects)

		if err != nil {
			t.Errorf("error updating bt: %v", err)
			continue
		}

		// Check if the output matches the expected values
		if len(trackedObjects) != len(frame.expectedTracks) {
			t.Errorf("Frame %d: expected %d tracked objects, got %d", frame.frameIdx, len(frame.expectedTracks), len(trackedObjects))
			continue
		}

		for i, track := range trackedObjects {

			expectedTrack := frame.expectedTracks[i]

			if track.GetTrackID() != expectedTrack.trackID ||
				!almostEqual(track.GetRect().TLX(), expectedTrack.tlx, tolerance) ||
				!almostEqual(track.GetRect().TLY(), expectedTrack.tly, tolerance) ||
				!almostEqual(track.GetRect().BRX(), expectedTrack.brx, tolerance) ||
				!almostEqual(track.GetRect().BRY(), expectedTrack.bry, tolerance) ||
				!almostEqual(track.GetScore(), expectedTrack.prob, tolerance) ||
				track.GetDetectionID() != expectedTrack.detectionID {

				t.Errorf("Frame %d: expected track %v, got track %v", frame.frameIdx, expectedTrack, track)
			}
		}

		/*
			// Print the tracked objects for each frame
			fmt.Printf("Processing frame %d\n", frame.frameIdx)

			for _, track := range trackedObjects {
				fmt.Printf("Frame %d, Track ID: %d, Rect: [%.5f, %.5f, %.5f, %.5f], Prob: %.2f, Detection ID: %d\n",
					frame.frameIdx, track.GetTrackID(), track.GetRect().TLX(), track.GetRect().TLY(),
					track.GetRect().BRX(), track.GetRect().BRY(), track.GetScore(), track.GetDetectionID())
			}
		*/
	}
}
