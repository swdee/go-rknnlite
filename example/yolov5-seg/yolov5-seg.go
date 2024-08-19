package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"github.com/swdee/go-rknnlite/preprocess"
	"github.com/swdee/go-rknnlite/render"
	"gocv.io/x/gocv"
	"log"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/yolov5s-seg-640-640-rk3588.rknn", "RKNN compiled YOLO model file")
	imgFile := flag.String("i", "../data/bus.jpg", "Image file to run object detection on")
	labelFile := flag.String("l", "../data/coco_80_labels_list.txt", "Text file containing model labels")
	saveFile := flag.String("o", "../data/bus-yolov5-seg-out.jpg", "The output JPG file with object detection markers")
	renderFormat := flag.String("r", "outline", "The rendering format used for instance segmentation [outline|mask|dump]")

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

	// set runtime to leave output tensors as int8
	rt.SetWantFloat(false)

	// optional querying of model file tensors and SDK version.  not necessary
	// for production inference code
	inputAttrs := optionalQueries(rt)

	// create YOLOv5 post processor
	yoloProcesser := postprocess.NewYOLOv5Seg(postprocess.YOLOv5SegCOCOParams())

	// load in Model class names
	classNames, err := rknnlite.LoadLabels(*labelFile)

	if err != nil {
		log.Fatal("Error loading model labels: ", err)
	}

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// convert colorspace and resize image
	rgbImg := gocv.NewMat()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	resizer := preprocess.NewResizer(img.Cols(), img.Rows(),
		int(inputAttrs[0].Dims[1]), int(inputAttrs[0].Dims[2]))

	cropImg := gocv.NewMat()
	resizer.LetterBoxResize(rgbImg, &cropImg, render.Black)

	defer img.Close()
	defer rgbImg.Close()
	defer cropImg.Close()

	start := time.Now()

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{cropImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	endInference := time.Now()

	// detect objects
	detectResults, segMask := yoloProcesser.DetectObjects(outputs, resizer)

	endDetect := time.Now()

	switch *renderFormat {
	case "mask":
		// draw segmentation mask
		render.SegmentMask(&img, segMask.Mask, 0.5)

		render.DetectionBoxes(&img, detectResults, classNames,
			render.DefaultFont(), 2)

	case "dump":
		// dump only segmentation mask to file
		err = render.PaintSegmentToFile(*saveFile,
			img.Rows(), img.Cols(), segMask.Mask, 1)

		if err != nil {
			log.Fatal("Failed to dump segmentation mask to file: ", err)
		}

	case "outline":
		fallthrough
	default:
		// default outline
		render.SegmentOutline(&img, segMask.Mask, detectResults, 1000,
			classNames, render.DefaultFont(), 2)
	}

	endRendering := time.Now()

	// output detection boxes to stdout
	for _, detResult := range detectResults {
		fmt.Printf("%s @ (%d %d %d %d) %f\n", classNames[detResult.Class], detResult.Box.Left, detResult.Box.Top, detResult.Box.Right, detResult.Box.Bottom, detResult.Probability)
	}

	log.Printf("Model first run speed: inference=%s, post processing=%s, rendering=%s, total time=%s\n",
		endInference.Sub(start).String(),
		endDetect.Sub(endInference).String(),
		endRendering.Sub(endDetect).String(),
		endRendering.Sub(start).String(),
	)

	// save the result
	if *renderFormat != "dump" {
		if ok := gocv.IMWrite(*saveFile, img); !ok {
			log.Fatal("Failed to save the image")
		}
	}

	log.Printf("Saved object detection result to %s\n", *saveFile)

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

func optionalQueries(rt *rknnlite.Runtime) []rknnlite.TensorAttr {

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

	return inputAttrs
}
