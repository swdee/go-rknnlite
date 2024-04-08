package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"gocv.io/x/gocv"
	"image"
	"log"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/mobilenet_v1-rk3588.rknn", "RKNN compiled model file")
	imgFile := flag.String("i", "../data/cat_224x224.jpg", "Image file to run inference on")
	flag.Parse()

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// convert colorspace and resize image
	rgbImg := gocv.NewMat()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	cropImg := rgbImg.Clone()
	gocv.Resize(rgbImg, &cropImg, image.Pt(224, 224), 0, 0, gocv.InterpolationArea)

	defer img.Close()
	defer rgbImg.Close()
	defer cropImg.Close()

	// create rknn runtime instance
	rt, err := rknnlite.NewRuntime(*modelFile, rknnlite.NPUCoreAuto)

	if err != nil {
		log.Fatal("Error initialing RKNN runtime: ", err)
	}

	// optional querying of model file tensors and SDK version.  not necessary
	// for production inference code
	optionalQueries(rt)

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{cropImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	// post process outputs and show top5 matches
	log.Println(" --- Top5 ---")

	for _, next := range rknnlite.GetTop5(outputs) {
		log.Printf("%3d: %8.6f\n", next.LabelIndex, next.Probability)
	}

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func optionalQueries(rt *rknnlite.Runtime) {

	// get SDK version
	ver, err := rt.SDKVersion()

	if err != nil {
		log.Fatal("Error initialing RKNN runtime: ", err)
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
}
