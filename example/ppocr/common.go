package main

import (
	"gocv.io/x/gocv"
	"image"
)

// resizeKeepAspectRatio resizes an image to a desired width and height while
// maintaining the aspect ratio. The resulting image is centered with black
// letterboxing where necessary.
func resizeKeepAspectRatio(srcImg gocv.Mat, dstImg *gocv.Mat, width, height int) {

	// calculate the ratio of the original image
	srcWidth := srcImg.Cols()
	srcHeight := srcImg.Rows()
	srcRatio := float64(srcWidth) / float64(srcHeight)

	// calculate the ratio of the new dimensions
	dstRatio := float64(width) / float64(height)

	newWidth, newHeight := width, height

	// adjust dimensions to maintain aspect ratio
	if srcRatio > dstRatio {
		newHeight = int(float64(width) / srcRatio)
	} else {
		newWidth = int(float64(height) * srcRatio)
	}

	// resize the original image to the new size that fits within the desired dimensions
	resizedImg := gocv.NewMat()
	gocv.Resize(srcImg, &resizedImg, image.Pt(newWidth, newHeight), 0, 0, gocv.InterpolationLinear)
	defer resizedImg.Close()

	// ensure destination Mat is the correct size and type
	if dstImg.Empty() {
		*dstImg = gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)
	}

	// create a black image
	dstImg.SetTo(gocv.NewScalar(0, 0, 0, 0))

	// find the top-left corner coordinates to center the resized image
	//x := (width - newWidth) / 2
	y := (height - newHeight) / 2
	x := 0

	// define a region of interest (ROI) within the final image where the
	// resized image will be placed
	roi := dstImg.Region(image.Rect(x, y, x+newWidth, y+newHeight))
	resizedImg.CopyTo(&roi)
	roi.Close()
}
