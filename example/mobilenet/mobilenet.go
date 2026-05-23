/*
Example code showing how to perform inferencing using a MobileNetv1 model.
*/
package main

import (
	"flag"
	"image"
	"log"
	"os"
	"strings"
	"time"

	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/bench"
	"gocv.io/x/gocv"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/models/rk3588/mobilenet_v1-rk3588.rknn", "RKNN compiled model file")
	imgFile := flag.String("i", "../data/cat_224x224.jpg", "Image file to run inference on")
	rkPlatform := flag.String("p", "rk3588", "Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588]")
	flag.Parse()

	err := rknnlite.SetCPUAffinityByPlatform(*rkPlatform, rknnlite.FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %v", err)
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

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{cropImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	// post process outputs and show top5 matches
	log.Println(" --- Top5 ---")

	for _, next := range rknnlite.GetTop5(outputs.Output) {
		log.Printf("%3d: %8.6f\n", next.LabelIndex, next.Probability)
	}

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// optional code.  run benchmark to get average time
	runBenchmark(rt, []gocv.Mat{cropImg})

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, mats []gocv.Mat) {

	report, err := bench.Run(bench.Config{
		Warmup: 5,
		Count:  100,
		Metrics: []string{
			"inference",
			"postprocess",
		},
	}, func() (map[string]time.Duration, error) {

		start := time.Now()

		// Perform inference.
		outputs, err := rt.Inference(mats)
		if err != nil {
			return nil, err
		}

		endInference := time.Now()

		// Post process classification output.
		_ = rknnlite.GetTop5(outputs.Output)

		endPost := time.Now()

		// Free RKNN output buffers.
		err = outputs.Free()
		if err != nil {
			return nil, err
		}

		return map[string]time.Duration{
			"inference":   endInference.Sub(start),
			"postprocess": endPost.Sub(endInference),
		}, nil
	})

	if err != nil {
		log.Fatal("Benchmark failed: ", err)
	}

	report.Print()
}
