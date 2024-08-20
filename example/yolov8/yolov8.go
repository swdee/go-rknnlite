/*
Example code showing how to perform object detection using a YOLOv8 model.
*/
package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"github.com/swdee/go-rknnlite/render"
	"gocv.io/x/gocv"
	"image"
	"log"
	"os"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/yolov8s-640-640-rk3588.rknn", "RKNN compiled YOLO model file")
	imgFile := flag.String("i", "../data/bus.jpg", "Image file to run object detection on")
	labelFile := flag.String("l", "../data/coco_80_labels_list.txt", "Text file containing model labels")
	saveFile := flag.String("o", "../data/bus-yolov8-out.jpg", "The output JPG file with object detection markers")

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

	// optional querying of model file tensors and SDK version for printing
	// to stdout.  not necessary for production inference code
	err = rt.Query(os.Stdout)

	if err != nil {
		log.Fatal("Error querying runtime: ", err)
	}

	// create YOLOv8 post processor
	yoloProcesser := postprocess.NewYOLOv8(postprocess.YOLOv8COCOParams())

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

	cropImg := rgbImg.Clone()
	scaleSize := image.Pt(int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))
	gocv.Resize(rgbImg, &cropImg, scaleSize, 0, 0, gocv.InterpolationArea)

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

	detectResults := yoloProcesser.DetectObjects(outputs)

	endDetect := time.Now()

	render.DetectionBoxes(&img, detectResults, classNames,
		render.DefaultFont(), 2)

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

	// Save the result
	if ok := gocv.IMWrite(*saveFile, img); !ok {
		log.Fatal("Failed to save the image")
	}

	log.Printf("Saved object detection result to %s\n", *saveFile)

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// optional code.  run benchmark to get average time
	runBenchmark(rt, yoloProcesser, []gocv.Mat{cropImg}, classNames, img)

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, yoloProcesser *postprocess.YOLOv8,
	mats []gocv.Mat, classNames []string, srcImg gocv.Mat) {

	count := 20
	start := time.Now()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		// post process
		detectResults := yoloProcesser.DetectObjects(outputs)

		render.DetectionBoxes(&srcImg, detectResults, classNames,
			render.DefaultFont(), 2)

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
