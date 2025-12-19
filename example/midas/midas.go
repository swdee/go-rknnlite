/*
Example code showing how to perform depth estimation using a MiDaS model.
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
	"github.com/swdee/go-rknnlite/postprocess"
	"gocv.io/x/gocv"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/models/rk3588/dpt_swin2_tiny_256-rk3588.rknn", "RKNN compiled depth model file")
	imgFile := flag.String("i", "../data/bedroom.jpg", "Image file to run depth estimation on")
	saveFile := flag.String("o", "../data/bedroom-out.jpg", "Output JPG file (depth visualization)")
	rkPlatform := flag.String("p", "rk3588", "Rockchip platform [rk3562|rk3566|rk3568|rk3576|rk3582|rk3588]")

	flag.Parse()

	err := rknnlite.SetCPUAffinityByPlatform(*rkPlatform, rknnlite.FastCores)

	if err != nil {
		log.Printf("Failed to set CPU affinity: %v\n", err)
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

	// We want float32 outputs for easy depth visualization
	rt.SetWantFloat(true)

	// optional querying of model file tensors and SDK version for printing
	// to stdout.  not necessary for production inference code
	err = rt.Query(os.Stdout)

	if err != nil {
		log.Fatal("Error querying runtime: ", err)
	}

	// create midas post processor
	midasProcessor := postprocess.NewMiDaS(postprocess.MiDaSDefaultParams())

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// convert colorspace and resize image to input tensor size
	rgbImg := gocv.NewMat()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	cropImg := rgbImg.Clone()
	scaleSize := image.Pt(int(rt.InputAttrs()[0].Dims[2]), int(rt.InputAttrs()[0].Dims[1]))
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

	//  post process and create depth map
	depthMap := gocv.NewMat()
	defer depthMap.Close()
	err = midasProcessor.CreateDepthMap(outputs, depthMap)

	if err != nil {
		log.Fatal("Error creating depth map: ", err)
	}

	endCreateMap := time.Now()

	// resize the color map back to the original input image size
	resizedMap := gocv.NewMat()
	defer resizedMap.Close()
	gocv.Resize(depthMap, &resizedMap, image.Pt(img.Cols(), img.Rows()), 0, 0, gocv.InterpolationCubic)

	endRendering := time.Now()

	log.Printf("Model first run speed: inference=%s, post processing=%s, rendering=%s, total time=%s\n",
		endInference.Sub(start).String(),
		endCreateMap.Sub(endInference).String(),
		endRendering.Sub(endCreateMap).String(),
		endRendering.Sub(start).String(),
	)

	// Save the result
	if ok := gocv.IMWrite(*saveFile, resizedMap); !ok {
		log.Fatal("Failed to save the image")
	}

	log.Printf("Saved depth map result to %s\n", *saveFile)

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// optional code.  run benchmark to get average time
	runBenchmark(rt, midasProcessor, []gocv.Mat{cropImg}, img)

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, midasProcessor *postprocess.MiDaS,
	mats []gocv.Mat, srcImg gocv.Mat) {

	count := 20
	start := time.Now()

	depthMap := gocv.NewMat()
	defer depthMap.Close()
	resizedMap := gocv.NewMat()
	defer resizedMap.Close()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		// post process
		err = midasProcessor.CreateDepthMap(outputs, depthMap)

		if err != nil {
			log.Fatal("Error creating depth map: ", err)
		}

		// resize the color map back to the original input image size
		gocv.Resize(depthMap, &resizedMap, image.Pt(srcImg.Cols(), srcImg.Rows()), 0, 0, gocv.InterpolationCubic)

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
