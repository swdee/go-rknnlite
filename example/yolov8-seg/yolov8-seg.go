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
	"os"
	"strings"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/models/rk3588/yolov8s-seg-rk3588.rknn", "RKNN compiled YOLO model file")
	imgFile := flag.String("i", "../data/catdog.jpg", "Image file to run object detection on")
	labelFile := flag.String("l", "../data/coco_80_labels_list.txt", "Text file containing model labels")
	saveFile := flag.String("o", "../data/catdog-yolov8-seg-out.jpg", "The output JPG file with object detection markers")
	renderFormat := flag.String("r", "outline", "The rendering format used for instance segmentation [outline|mask|dump]")
	rkPlatform := flag.String("p", "rk3588", "Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588]")

	flag.Parse()

	err := rknnlite.SetCPUAffinityByPlatform(*rkPlatform, rknnlite.FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %v\n", err)
	}

	// check if user specified model file or if default is being used.  if default
	// then pick the default platform model to use.
	if f := flag.Lookup("m"); f != nil && f.Value.String() == f.DefValue && *rkPlatform != "rk3588" {
		*modelFile = strings.ReplaceAll(*modelFile, "rk3588", *rkPlatform)
	}

	// create rknn runtime instance
	rt, err := rknnlite.NewRuntimeByPlatform(*rkPlatform, *modelFile)

	if err != nil {
		log.Fatal("Error initializing RKNN runtime: ", err)
	}

	// set runtime to leave output tensors as int8
	rt.SetWantFloat(false)

	// optional querying of model file tensors and SDK version for printing
	// to stdout.  not necessary for production inference code
	err = rt.Query(os.Stdout)

	if err != nil {
		log.Fatal("Error querying runtime: ", err)
	}

	// create YOLOv8 post processor
	yoloProcesser := postprocess.NewYOLOv8Seg(postprocess.YOLOv8SegCOCOParams())

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
		int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))

	cropImg := rgbImg.Clone()
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
	detectObjs := yoloProcesser.DetectObjects(outputs, resizer)
	detectResults := detectObjs.GetDetectResults()
	segMask := yoloProcesser.SegmentMask(detectObjs, resizer)

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

	// optional code.  run benchmark to get average time
	runBenchmark(rt, yoloProcesser, []gocv.Mat{cropImg}, classNames,
		resizer, *renderFormat, img)

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, yoloProcesser *postprocess.YOLOv8Seg,
	mats []gocv.Mat, classNames []string, resizer *preprocess.Resizer,
	renderFormat string, srcImg gocv.Mat) {

	count := 100
	start := time.Now()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		// post process
		detectObjs := yoloProcesser.DetectObjects(outputs, resizer)
		detectResults := detectObjs.GetDetectResults()
		segMask := yoloProcesser.SegmentMask(detectObjs, resizer)

		switch renderFormat {
		case "mask":
			// draw segmentation mask
			render.SegmentMask(&srcImg, segMask.Mask, 0.5)

			render.DetectionBoxes(&srcImg, detectResults, classNames,
				render.DefaultFont(), 2)

		case "dump":
			// do nothing

		case "outline":
			fallthrough
		default:
			// default outline
			render.SegmentOutline(&srcImg, segMask.Mask, detectResults, 1000,
				classNames, render.DefaultFont(), 2)
		}

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
