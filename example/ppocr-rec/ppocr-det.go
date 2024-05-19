package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"log"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/ppocrv4_det-rk3588.rknn", "RKNN compiled model file")
	imgFile := flag.String("i", "../data/ppocr-det-test.png", "Image file to run inference on")
	saveFile := flag.String("o", "../data/ppocr-det-out.png", "The output PNG file with object detection markers")
	flag.Parse()

	err := rknnlite.SetCPUAffinity(rknnlite.RK3588FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %w", err)
	}

	// create rknn runtime instance
	rt, err := rknnlite.NewRuntime(*modelFile, rknnlite.NPUCoreAuto)

	if err != nil {
		log.Fatal("Error initializing RKNN runtime: ", err)
	}

	// optional querying of model file tensors and SDK version.  not necessary
	// for production inference code
	inputAttrs, _ := optionalQueries(rt)

	// create PPOCR post processor
	ppocrProcessor := postprocess.NewPPOCRDetect(postprocess.PPOCRDetectParams{
		Threshold:    0.3,
		BoxThreshold: 0.6,
		Dilation:     false,
		BoxType:      "poly", //"poly",
		UnclipRatio:  1.5,
		ScoreMode:    "slow", // slow
		ModelWidth:   int(inputAttrs[0].Dims[2]),
		ModelHeight:  int(inputAttrs[0].Dims[1]),
	})

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// resize image to 480x480 and keep aspect ratio, centered with black letterboxing
	resizedImg := gocv.NewMat()
	resizeKeepAspectRatio(img, &resizedImg, int(inputAttrs[0].Dims[2]), int(inputAttrs[0].Dims[1]))

	defer img.Close()
	defer resizedImg.Close()

	start := time.Now()

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{resizedImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	endInference := time.Now()

	// work out scale ratio between source imnage and resized image
	scaleW := float32(img.Cols()) / float32(resizedImg.Cols())
	scaleH := float32(img.Rows()) / float32(resizedImg.Rows())

	results := ppocrProcessor.Detect(outputs, scaleW, scaleH)

	endDetect := time.Now()

	log.Printf("Model first run speed: inference=%s, post processing=%s, total time=%s\n",
		endInference.Sub(start).String(),
		endDetect.Sub(endInference).String(),
		endDetect.Sub(start).String(),
	)

	//
	lineColor := color.RGBA{R: 255, G: 0, B: 0, A: 255} // Red color
	thickness := 2

	for _, result := range results {
		for i, box := range result.Box {
			fmt.Printf("[%d]: [(%d, %d), (%d, %d), (%d, %d), (%d, %d)] %f\n",
				i,
				box.LeftTop.X, box.LeftTop.Y,
				box.RightTop.X, box.RightTop.Y,
				box.RightBottom.X, box.RightBottom.Y,
				box.LeftBottom.X, box.LeftBottom.Y,
				box.Score)

			// draw onto the source image the bounding box lines
			topLeft := image.Pt(box.LeftTop.X, box.LeftTop.Y)
			topRight := image.Pt(box.RightTop.X, box.RightTop.Y)
			bottomRight := image.Pt(box.RightBottom.X, box.RightBottom.Y)
			bottomLeft := image.Pt(box.LeftBottom.X, box.LeftBottom.Y)

			gocv.Line(&img, topLeft, topRight, lineColor, thickness)
			gocv.Line(&img, topRight, bottomRight, lineColor, thickness)
			gocv.Line(&img, bottomRight, bottomLeft, lineColor, thickness)
			gocv.Line(&img, bottomLeft, topLeft, lineColor, thickness)
		}
	}

	log.Printf("Saved image to %s\n", *saveFile)
	gocv.IMWrite(*saveFile, img)

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

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
