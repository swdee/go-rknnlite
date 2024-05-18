/*
Example code showing how to perform OCR on an image using PaddleOCR recognition
*/
package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"gocv.io/x/gocv"
	"image"
	"log"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/ppocrv4_rec-rk3588.rknn", "RKNN compiled model file")
	imgFile := flag.String("i", "../data/ppocr-rec-test.png", "Image file to run inference on")
	keysFile := flag.String("k", "../data/ppocr_keys_v1.txt", "Text file containing OCR character keys")
	flag.Parse()

	// create rknn runtime instance
	rt, err := rknnlite.NewRuntime(*modelFile, rknnlite.NPUCoreAuto)

	if err != nil {
		log.Fatal("Error initializing RKNN runtime: ", err)
	}

	// set runtime to pass input gocv.Mat's to Inference() function as float32
	// to RKNN backend
	rt.SetInputTypeFloat32(true)

	// optional querying of model file tensors and SDK version.  not necessary
	// for production inference code
	inputAttrs, outputAttrs := optionalQueries(rt)

	// load in Model character labels
	modelChars, err := rknnlite.LoadLabels(*keysFile)

	if err != nil {
		log.Fatal("Error loading model OCR character keys: ", err)
	}

	// check that we have as many modelChars as tensor outputs dimension
	if len(modelChars) != int(outputAttrs[0].Dims[2]) {
		log.Fatalf("OCR character keys text input has %d characters and does "+
			"not match the required number in the Model of %d",
			len(modelChars), outputAttrs[0].Dims[2])
	}

	// create PPOCR post processor
	ppocrProcessor := postprocess.NewPPOCRRecognise(postprocess.PPOCRRecogniseParams{
		ModelChars:   modelChars,
		OutputSeqLen: int(inputAttrs[0].Dims[2]) / 8, // modelWidth (320/8)
	})

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// resize image to 320x48 and keep aspect ratio, centered with black letterboxing
	resizedImg := gocv.NewMat()
	resizeKeepAspectRatio(img, &resizedImg, int(inputAttrs[0].Dims[2]), int(inputAttrs[0].Dims[1]))

	// convert image to float32 in 3 channels
	resizedImg.ConvertTo(&resizedImg, gocv.MatTypeCV32FC3)

	// normalize the image (img - 127.5) / 127.5
	resizedImg.AddFloat(-127.5)
	resizedImg.DivideFloat(127.5)

	defer img.Close()
	defer resizedImg.Close()

	start := time.Now()

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{resizedImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	endInference := time.Now()

	results := ppocrProcessor.Recognise(outputs)

	endRecognise := time.Now()

	log.Printf("Model first run speed: inference=%s, post processing=%s, total time=%s\n",
		endInference.Sub(start).String(),
		endRecognise.Sub(endInference).String(),
		endRecognise.Sub(start).String(),
	)

	for _, result := range results {
		log.Printf("Recognize result: %s, score=%.2f", result.Text, result.Score)
	}

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// optional code.  run benchmark to get average time of 10 runs
	runBenchmark(rt, ppocrProcessor, []gocv.Mat{resizedImg})

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, ppocrProcessor *postprocess.PPOCRRecognise,
	mats []gocv.Mat) {

	count := 100
	start := time.Now()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		// post process
		_ = ppocrProcessor.Recognise(outputs)

		err = outputs.Free()

		if err != nil {
			log.Fatal("Error freeing Outputs: ", err)
		}
	}

	end := time.Now()
	total := end.Sub(start)
	avg := total / time.Duration(count)

	log.Printf("Benchmark time=%s, count=%d, average total time=%s\n",
		total.String(), count, avg.String(),
	)
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
