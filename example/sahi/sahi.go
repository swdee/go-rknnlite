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
	"strings"
	"sync"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/models/rk3588/yolov5s-rk3588.rknn", "RKNN compiled YOLO model file")
	imgFile := flag.String("i", "../data/protest.jpg", "Image file to run object detection on")
	poolSize := flag.Int("s", 1, "Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3")
	labelFile := flag.String("l", "../data/coco_80_labels_list.txt", "Text file containing model labels")
	saveFile := flag.String("o", "../data/protest-sahi-out.jpg", "The output JPG file with object detection markers")
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

	// create new pool
	pool, err := rknnlite.NewPoolByPlatform(*rkPlatform, *poolSize, *modelFile)

	if err != nil {
		log.Fatalf("Error creating RKNN pool: %v\n", err)
	}

	// set runtime to leave output tensors as int8
	pool.SetWantFloat(false)

	// create YOLOv5 post processor
	yoloProcesser := postprocess.NewYOLOv5(postprocess.YOLOv5COCOParams())

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

	// output dimensions of source image
	log.Printf("Source image dimensions %dx%d\n", img.Cols(), img.Rows())

	// get the tensor input dimensions
	rt := pool.Get()
	tensorWidth := int(rt.InputAttrs()[0].Dims[1])
	tensorHeight := int(rt.InputAttrs()[0].Dims[2])
	pool.Return(rt)

	start := time.Now()

	sahi := preprocess.NewSAHI(tensorWidth, tensorHeight, 0.2, 0.2)
	slices := sahi.Slice(img)

	// waitgroup used to wait for all go-routines to complete before closing
	// the pool
	var wg sync.WaitGroup
	// create mutex to ensure stdout results are in order
	var printMu sync.Mutex

	// run inference on all the slices
	for _, slice := range slices {
		// Pin the current slice in a new variable
		sl := slice

		wg.Add(1)

		// build one big string for this slice’s output
		var sb strings.Builder

		sb.WriteString(fmt.Sprintf("\nProcessing Slice (%d %d %d %d) with box size (%d %d)\n",
			sl.X, sl.Y, sl.X2, sl.Y2, sl.X2-sl.X, sl.Y2-sl.Y),
		)

		// pool.Get() blocks if no runtimes are available in the pool
		rt := pool.Get()

		go func(sl preprocess.Slice, rt *rknnlite.Runtime) {
			// perform inference on image file
			outputs, err := rt.Inference([]gocv.Mat{*sl.Mat()})

			if err != nil {
				log.Fatal("Runtime inferencing failed with error: ", err)
			}

			detectObjs := yoloProcesser.DetectObjects(outputs, sl.Resizer())
			detectResults := detectObjs.GetDetectResults()

			// output detection boxes to stdout
			for _, detResult := range detectResults {
				sb.WriteString(fmt.Sprintf(
					"%s @ (%d %d %d %d) %f\n",
					classNames[detResult.Class], detResult.Box.Left,
					detResult.Box.Top, detResult.Box.Right,
					detResult.Box.Bottom, detResult.Probability),
				)
			}

			sahi.AddResult(sl, detectResults)

			// free outputs allocated in C memory after you have finished post processing
			err = outputs.Free()
			sl.Free()

			// print slice object detection results
			printMu.Lock()
			fmt.Print(sb.String())
			printMu.Unlock()

			pool.Return(rt)
			wg.Done()
		}(sl, rt)
	}

	wg.Wait()

	// get the detection results from all slices combined into those which map
	// back onto the source image dimensions
	detectResults := sahi.GetDetectResults(postprocess.YOLOv5COCOParams().NMSThreshold, 0.7)

	fmt.Printf("\nCombined object detection results\n")

	for _, detResult := range detectResults {
		fmt.Printf("%s @ (%d %d %d %d) %f\n", classNames[detResult.Class], detResult.Box.Left, detResult.Box.Top, detResult.Box.Right, detResult.Box.Bottom, detResult.Probability)
	}

	render.DetectionBoxes(&img, detectResults, classNames,
		render.DefaultFont(), 2)

	log.Printf("SAHI Execution speed=%s, slices=%d, objects=%d\n",
		time.Now().Sub(start).String(),
		len(slices),
		len(detectResults),
	)

	// Save the result
	if ok := gocv.IMWrite(*saveFile, img); !ok {
		log.Fatal("Failed to save the image")
	}

	log.Printf("Saved object detection result to %s\n", *saveFile)

	// free results
	sahi.FreeResults()

	// close runtime and release resources
	pool.Close()

	log.Println("done")
}
