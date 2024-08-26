package render

import (
	"github.com/swdee/go-rknnlite/postprocess"
	"gocv.io/x/gocv"
	"image"
)

/* skeleton keypoints
0: Nose
1: Left Eye
2: Right Eye
3: Left Ear
4: Right Ear
5: Left Shoulder
6: Right Shoulder
7: Left Elbow
8: Right Elbow
9: Left Wrist
10: Right Wrist
11: Left Hip
12: Right Hip
13: Left Knee
14: Right Knee
15: Left Ankle
16: Right Ankle
*/

var (
	// skeleton defines the pose skeleton points to draw lines between.  The numbers
	// are paired, so (16,14) means draw line from right ankle to right knee.
	skeleton = [38]int{16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
		7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7}
	// keyPointsTotal is the number of keypoints in a skeleton
	keyPointsTotal = 17
)

// PoseKeyPoints renders the provided pose estimation keypoints for all objects
func PoseKeyPoints(img *gocv.Mat, keyPoints [][]postprocess.KeyPoint,
	lineThickness int) {

	// for each object
	for i := 0; i < len(keyPoints); i++ {

		// an individual object's key points
		keyPoint := keyPoints[i]

		// draw skeleton lines
		for j := 0; j < len(skeleton)/2; j++ {
			x1 := keyPoint[skeleton[2*j]-1].X
			y1 := keyPoint[skeleton[2*j]-1].Y
			x2 := keyPoint[skeleton[2*j+1]-1].X
			y2 := keyPoint[skeleton[2*j+1]-1].Y

			gocv.Line(img, image.Pt(x1, y1), image.Pt(x2, y2), limbColors[j], lineThickness)
		}

		// draw circles at skeleton joints
		for j := 0; j < keyPointsTotal; j++ {
			gocv.Circle(img, image.Pt(keyPoint[j].X, keyPoint[j].Y),
				3, keyPointColors[j], -1) // style.CircleRadius
		}
	}
}
