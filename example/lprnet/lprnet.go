/*
Example code showing how to perform inferencing using a LPRnet model
*/
package main

import (
	"flag"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
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
	modelFile := flag.String("m", "../data/lprnet-rk3588.rknn", "RKNN compiled model file")
	imgFile := flag.String("i", "../data/lplate.jpg", "Image file to run inference on")
	flag.Parse()

	err := rknnlite.SetCPUAffinity(rknnlite.RK3588FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %v\n", err)
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

	// create LPRNet post processor using parameters used during model training
	lprnetProcesser := postprocess.NewLPRNet(postprocess.LPRNetParams{
		PlatePositions: 18,
		PlateChars: []string{
			"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
			"苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
			"桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
			"新",
			"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			"A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
			"L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
			"W", "X", "Y", "Z", "I", "O", "-",
		},
	})

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// resize image to 94x24
	cropImg := gocv.NewMat()
	scaleSize := image.Pt(int(rt.InputAttrs()[0].Dims[2]), int(rt.InputAttrs()[0].Dims[1]))
	gocv.Resize(img, &cropImg, scaleSize, 0, 0, gocv.InterpolationArea)

	defer img.Close()
	defer cropImg.Close()

	start := time.Now()

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{cropImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	endInference := time.Now()

	// read number plates from outputs
	plates := lprnetProcesser.ReadPlates(outputs)

	endDetect := time.Now()

	log.Printf("Model first run speed: inference=%s, post processing=%s, total time=%s\n",
		endInference.Sub(start).String(),
		endDetect.Sub(endInference).String(),
		endDetect.Sub(start).String(),
	)

	for _, plate := range plates {
		log.Printf("License plate recognition result: %s\n", plate)
	}

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// optional code.  run benchmark to get average time of 10 runs
	runBenchmark(rt, lprnetProcesser, []gocv.Mat{cropImg})

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, lprnetProcesser *postprocess.LPRNet,
	mats []gocv.Mat) {

	count := 10
	start := time.Now()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		// post process
		_ = lprnetProcesser.ReadPlates(outputs)

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
