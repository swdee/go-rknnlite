/*
Example code showing how to perform inferencing using a Retina Face model.
*/
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
	modelFile := flag.String("m", "../data/models/rk3588/retinaface-320-rk3588.rknn", "RKNN compiled Retina Face model file")
	imgFile := flag.String("i", "../data/face.jpg", "Image file to run inference on")
	saveFile := flag.String("o", "../data/face-out.jpg", "The output JPG file with face detection markers")
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

	// optional querying of model file tensors and SDK version for printing
	// to stdout.  not necessary for production inference code
	err = rt.Query(os.Stdout)

	if err != nil {
		log.Fatal("Error querying runtime: ", err)
	}

	retinaProcessor := postprocess.NewRetinaFace(postprocess.WiderFaceParams())

	// retinaface does not use class names in its model, so define a single placeholder "face"
	classNames := []string{"face"}

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

	detectFaces := retinaProcessor.DetectFaces(outputs, resizer)
	detectResults := detectFaces.GetDetectResults()
	keyPoints := retinaProcessor.GetFaceLandmarks(detectFaces)

	endDetect := time.Now()

	render.FaceKeyPoints(&img, keyPoints)

	render.DetectionBoxes(&img, detectResults, classNames,
		render.DefaultFont(), 2)

	endRendering := time.Now()

	// output detection boxes to stdout
	for _, detResult := range detectResults {
		fmt.Printf("face @ (%d %d %d %d) %f\n", detResult.Box.Left, detResult.Box.Top, detResult.Box.Right, detResult.Box.Bottom, detResult.Probability)
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
	runBenchmark(rt, retinaProcessor, []gocv.Mat{cropImg}, classNames, resizer, img)

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, retinaProcessor *postprocess.RetinaFace,
	mats []gocv.Mat, classNames []string, resizer *preprocess.Resizer,
	srcImg gocv.Mat) {

	count := 100
	start := time.Now()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		// post process
		detectFaces := retinaProcessor.DetectFaces(outputs, resizer)
		detectResults := detectFaces.GetDetectResults()
		keyPoints := retinaProcessor.GetFaceLandmarks(detectFaces)

		render.FaceKeyPoints(&srcImg, keyPoints)

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
