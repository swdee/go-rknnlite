package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"log"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/yolov5s-640-640-rk3588.rknn", "RKNN compiled YOLO model file")
	imgFile := flag.String("i", "../data/bus.jpg", "Image file to run object detection on")

	flag.Parse()

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

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// convert colorspace and resize image
	rgbImg := gocv.NewMat()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	cropImg := rgbImg.Clone()
	scaleSize := image.Pt(int(inputAttrs[0].Dims[1]), int(inputAttrs[0].Dims[2]))
	gocv.Resize(rgbImg, &cropImg, scaleSize, 0, 0, gocv.InterpolationArea)

	defer img.Close()
	defer rgbImg.Close()
	defer cropImg.Close()

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{cropImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	log.Println("outputs=", len(outputs.Output))

	detectResGrp := rt.DetectObjects(outputs.Output, 1.0, 1.0)

	for _, detResult := range detectResGrp.Results {
		text := fmt.Sprintf("%s %.1f%%", detResult.Name, detResult.Prop*100)
		fmt.Printf("%s @ (%d %d %d %d) %f\n", detResult.Name, detResult.Box.Left, detResult.Box.Top, detResult.Box.Right, detResult.Box.Bottom, detResult.Prop)

		// Draw rectangle around detected object
		//rect := image.Rect(detResult.Box.Left, detResult.Box.Top, detResult.Box.Right-detResult.Box.Left, detResult.Box.Bottom-detResult.Box.Top)
		rect := image.Rect(detResult.Box.Left, detResult.Box.Top, detResult.Box.Right, detResult.Box.Bottom)
		gocv.Rectangle(&img, rect, color.RGBA{R: 0, G: 0, B: 255, A: 0}, 2)

		// Put text
		gocv.PutText(&img, text, image.Pt(detResult.Box.Left, detResult.Box.Top+12), gocv.FontHersheyPlain, 0.8, color.RGBA{R: 255, G: 255, B: 255, A: 0}, 1)
	}

	// Save the result
	if ok := gocv.IMWrite("./bus-go-out.jpg", img); !ok {
		log.Println("Failed to save the image")
	}

	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
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
