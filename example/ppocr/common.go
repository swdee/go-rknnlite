package main

import (
	"fmt"
	"github.com/swdee/go-rknnlite"
	"gocv.io/x/gocv"
	"image"
	"log"
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

func optionalQueries(rt *rknnlite.Runtime) ([]rknnlite.TensorAttr, []rknnlite.TensorAttr) {

	// get SDK version
	ver, err := rt.SDKVersion()

	if err != nil {
		log.Fatal("Error initializing RKNN runtime: ", err)
	}

	fmt.Printf("Driver Version: %s, API Version: %s\n", ver.DriverVersion, ver.APIVersion)

	// get model input and output numbers
	num, err := rt.QueryModelIONumber()

	if err != nil {
		log.Fatal("Error querying IO Numbers: ", err)
	}

	log.Printf("Model Input Number: %d, Ouput Number: %d\n", num.NumberInput, num.NumberOutput)

	// query Input tensors
	inputAttrs, err := rt.QueryInputTensors()

	if err != nil {
		log.Fatal("Error querying Input Tensors: ", err)
	}

	log.Println("Input tensors:")

	for _, attr := range inputAttrs {
		log.Printf("  %s\n", attr.String())
	}

	// query Output tensors
	outputAttrs, err := rt.QueryOutputTensors()

	if err != nil {
		log.Fatal("Error querying Output Tensors: ", err)
	}

	log.Println("Output tensors:")

	for _, attr := range outputAttrs {
		log.Printf("  %s\n", attr.String())
	}

	return inputAttrs, outputAttrs
}
