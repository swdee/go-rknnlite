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
	"os"
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

	// optional querying of model file tensors and SDK version for printing
	// to stdout.  not necessary for production inference code
	err = rt.Query(os.Stdout)

	if err != nil {
		log.Fatal("Error querying runtime: ", err)
	}

	// create PPOCR post processor
	ppocrProcessor := postprocess.NewPPOCRDetect(postprocess.PPOCRDetectParams{
		Threshold:    0.3,
		BoxThreshold: 0.6,
		Dilation:     false,
		BoxType:      "poly",
		UnclipRatio:  1.5,
		ScoreMode:    "slow",
		ModelWidth:   int(rt.InputAttrs()[0].Dims[2]),
		ModelHeight:  int(rt.InputAttrs()[0].Dims[1]),
	})

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// resize image to 480x480 and keep aspect ratio, centered with black letterboxing
	resizedImg := gocv.NewMat()
	resizeKeepAspectRatio(img, &resizedImg, int(rt.InputAttrs()[0].Dims[2]), int(rt.InputAttrs()[0].Dims[1]))

	defer img.Close()
	defer resizedImg.Close()

	start := time.Now()

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{resizedImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	endInference := time.Now()

	// work out scale ratio between source image and resized image
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

	// optional code.  run benchmark to get average time of 10 runs
	runBenchmark(rt, ppocrProcessor, []gocv.Mat{resizedImg}, scaleW, scaleH)

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, ppocrProcessor *postprocess.PPOCRDetect,
	mats []gocv.Mat, scaleW, scaleH float32) {

	count := 100
	start := time.Now()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		// post process
		_ = ppocrProcessor.Detect(outputs, scaleW, scaleH)

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
