package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"github.com/swdee/go-rknnlite/preprocess"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"log"
	"time"
)

var (
	// classColors is a list of colors used to paint the object segment mask
	classColors = []color.RGBA{
		// Distinct colors
		{R: 255, G: 56, B: 56, A: 255},   // #FF3838
		{R: 255, G: 112, B: 31, A: 255},  // #FF701F
		{R: 255, G: 178, B: 29, A: 255},  // #FFB21D
		{R: 207, G: 210, B: 49, A: 255},  // #CFD231
		{R: 72, G: 249, B: 10, A: 255},   // #48F90A
		{R: 26, G: 147, B: 52, A: 255},   // #1A9334
		{R: 0, G: 212, B: 187, A: 255},   // #00D4BB
		{R: 0, G: 194, B: 255, A: 255},   // #00C2FF
		{R: 52, G: 69, B: 147, A: 255},   // #344593
		{R: 100, G: 115, B: 255, A: 255}, // #6473FF
		{R: 0, G: 24, B: 236, A: 255},    // #0018EC
		{R: 132, G: 56, B: 255, A: 255},  // #8438FF
		{R: 82, G: 0, B: 133, A: 255},    // #520085
		{R: 255, G: 149, B: 200, A: 255}, // #FF95C8
		{R: 255, G: 55, B: 199, A: 255},  // #FF37C7
		{R: 255, G: 157, B: 151, A: 255}, // #FF9D97
		{R: 44, G: 153, B: 168, A: 255},  // #2C99A8
		{R: 61, G: 219, B: 134, A: 255},  // #3DDB86
		{R: 203, G: 56, B: 255, A: 255},  // #CB38FF
		{R: 146, G: 204, B: 23, A: 255},  // #92CC17

		// Gradient colors
		{R: 250, G: 128, B: 114, A: 255}, // #FA8072
		{R: 64, G: 255, B: 0, A: 255},    // #40FF00
		{R: 255, G: 64, B: 0, A: 255},    // #FF4000
		{R: 64, G: 0, B: 255, A: 255},    // #4000FF
		{R: 0, G: 64, B: 255, A: 255},    // #0040FF
		{R: 0, G: 128, B: 255, A: 255},   // #0080FF
		{R: 0, G: 255, B: 0, A: 255},     // #00FF00
		{R: 128, G: 255, B: 0, A: 255},   // #80FF00
		{R: 255, G: 255, B: 128, A: 255}, // #FFFF80
		{R: 191, G: 255, B: 0, A: 255},   // #BFFF00
		{R: 191, G: 128, B: 255, A: 255}, // #BF80FF
		{R: 255, G: 128, B: 0, A: 255},   // #FF8000
		{R: 210, G: 105, B: 30, A: 255},  // #D2691E
		{R: 128, G: 255, B: 128, A: 255}, // #80FF80
		{R: 255, G: 128, B: 128, A: 255}, // #FF8080
		{R: 96, G: 96, B: 96, A: 255},    // #606060
		{R: 0, G: 0, B: 255, A: 255},     // #0000FF
		{R: 191, G: 0, B: 255, A: 255},   // #BF00FF
		{R: 255, G: 0, B: 0, A: 255},     // #FF0000
		{R: 192, G: 192, B: 192, A: 255}, // #C0C0C0
		{R: 128, G: 191, B: 255, A: 255}, // #80BFFF
		{R: 255, G: 0, B: 128, A: 255},   // #FF0080
		{R: 255, G: 0, B: 255, A: 255},   // #FF00FF
		{R: 255, G: 128, B: 128, A: 255}, // #FF8080
		{R: 0, G: 191, B: 255, A: 255},   // #00BFFF
		{R: 128, G: 128, B: 255, A: 255}, // #8080FF
		{R: 64, G: 0, B: 128, A: 255},    // #400080
		{R: 128, G: 0, B: 64, A: 255},    // #800040
		{R: 255, G: 128, B: 191, A: 255}, // #FF80BF
		{R: 0, G: 255, B: 255, A: 255},   // #00FFFF
		{R: 255, G: 0, B: 191, A: 255},   // #FF00BF
		{R: 128, G: 255, B: 255, A: 255}, // #80FFFF
		{R: 0, G: 255, B: 191, A: 255},   // #00FFBF
		{R: 255, G: 0, B: 64, A: 255},    // #FF0040
		{R: 255, G: 191, B: 128, A: 255}, // #FFBF80
		{R: 255, G: 255, B: 0, A: 255},   // #FFFF00
		{R: 255, G: 128, B: 255, A: 255}, // #FF80FF
		{R: 128, G: 255, B: 191, A: 255}, // #80FFBF
		{R: 128, G: 0, B: 255, A: 255},   // #8000FF
		{R: 255, G: 192, B: 203, A: 255}, // #FFC0CB
		{R: 191, G: 255, B: 128, A: 255}, // #BFFF80
		{R: 0, G: 255, B: 128, A: 255},   // #00FF80
		{R: 255, G: 191, B: 0, A: 255},   // #FFBF00
		{R: 0, G: 255, B: 64, A: 255},    // #00FF40
	}

	black = color.RGBA{R: 0, G: 0, B: 0, A: 255}
	white = color.RGBA{R: 255, G: 255, B: 255, A: 255}
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	/*
		// Start CPU profiling
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile() // Ensure the profile is stopped at the end
	*/

	// read in cli flags
	modelFile := flag.String("m", "../data/yolov5s-seg-640-640-rk3588.rknn", "RKNN compiled YOLO model file")
	imgFile := flag.String("i", "../data/bus.jpg", "Image file to run object detection on")
	labelFile := flag.String("l", "../data/coco_80_labels_list.txt", "Text file containing model labels")
	saveFile := flag.String("o", "../data/bus-yolov5-seg-out.jpg", "The output JPG file with object detection markers")

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

	// optional querying of model file tensors and SDK version.  not necessary
	// for production inference code
	inputAttrs := optionalQueries(rt)

	// create YOLOv5 post processor
	yoloProcesser := postprocess.NewYOLOv5Seg(postprocess.YOLOv5SegCOCOParams())

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
		int(inputAttrs[0].Dims[1]), int(inputAttrs[0].Dims[2]))

	cropImg := gocv.NewMat()
	resizer.LetterBoxResize(rgbImg, &cropImg, black)

	/*
		cropImg := rgbImg.Clone()
		scaleSize := image.Pt(int(inputAttrs[0].Dims[1]), int(inputAttrs[0].Dims[2]))
		gocv.Resize(rgbImg, &cropImg, scaleSize, 0, 0, gocv.InterpolationArea)
	*/

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

	log.Println("outputs=", len(outputs.Output))

	detectResults, segMask := yoloProcesser.DetectObjects(outputs, resizer)

	endDetect := time.Now()

	// draw segmentation mask
	/*	bench := time.Now()
		drawSegmentMask(&img, segMask.Mask, 0.5)
		log.Printf("draw mask time FAST: %dms\n", time.Since(bench).Milliseconds())
	*/

	/*
		// paint segment mask on black image to see mask only

		tmpImg := gocv.NewMatWithSize(img.Rows(), img.Cols(), gocv.MatTypeCV8UC3)
		drawSegmentMask(&tmpImg, segMask.Mask, 0.5)
		gocv.IMWrite("/tmp/yoloseg-alphachan.jpg", tmpImg)
		tmpImg.Close()
	*/

	drawBench := time.Now()
	drawSegmentOutline(&img, segMask.Mask, detectResults, 1000, classNames)
	log.Printf("draw outline time: %dms\n", time.Since(drawBench).Milliseconds())

	// draw detection boxes
	for _, detResult := range detectResults {

		//text := fmt.Sprintf("%s %.1f%%", classNames[detResult.Class], detResult.Probability*100)
		fmt.Printf("%s @ (%d %d %d %d) %f\n", classNames[detResult.Class], detResult.Box.Left, detResult.Box.Top, detResult.Box.Right, detResult.Box.Bottom, detResult.Probability)

		// Draw rectangle around detected object
		//		rect := image.Rect(detResult.Box.Left, detResult.Box.Top, detResult.Box.Right, detResult.Box.Bottom)
		//	gocv.Rectangle(&img, rect, color.RGBA{R: 0, G: 0, B: 255, A: 0}, 2)

		// Put text
		//	gocv.PutText(&img, text, image.Pt(detResult.Box.Left, detResult.Box.Top+12), gocv.FontHersheyDuplex, 0.4, color.RGBA{R: 255, G: 255, B: 255, A: 0}, 1)
	}

	endRendering := time.Now()

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

	// optional code.  run benchmark to get average time of 10 runs
	//runBenchmark(rt, yoloProcesser, []gocv.Mat{cropImg}, resizer)

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, yoloProcesser *postprocess.YOLOv5Seg,
	mats []gocv.Mat, resizer *preprocess.Resizer) {

	count := 100
	start := time.Now()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		// post process
		_, _ = yoloProcesser.DetectObjects(outputs, resizer)

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

// drawSegmentMask draws the segmentation mask on the image using the provide
// alpha/opacity level
func drawSegmentMask(img *gocv.Mat, segMask []uint8, alpha float32) {

	// get dimensions
	width := img.Cols()
	height := img.Rows()

	// it is too slow to manipulate pixel by pixel using GoCV due to slowness
	// over CGO.  So we copy the bytes from the source image and manipulate
	// the bytes directly before copying back to a Mat
	imgData := img.ToBytes()

	// iterate over each pixel in the segmentation mask
	for j := 0; j < height; j++ {
		for k := 0; k < width; k++ {

			idx := j*width + k

			if segMask[idx] != 0 {

				classIndex := segMask[idx] % uint8(len(classColors))
				color := classColors[classIndex]

				// calculate position in the byte slice
				pixelPos := j*width*3 + k*3

				// get original pixel colors directly from the byte slice
				b, g, r := imgData[pixelPos+0], imgData[pixelPos+1], imgData[pixelPos+2]

				// calculate blended colors based on alpha transparency
				imgData[pixelPos+0] = uint8(float32(b)*(1-alpha) + float32(color.B)*alpha)
				imgData[pixelPos+1] = uint8(float32(g)*(1-alpha) + float32(color.G)*alpha)
				imgData[pixelPos+2] = uint8(float32(r)*(1-alpha) + float32(color.R)*alpha)
			}
		}
	}

	// copy back to the original mat
	tmpImg, _ := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8UC3, imgData)
	defer tmpImg.Close()
	tmpImg.CopyTo(img)
}

type boxLabel struct {
	rect    image.Rectangle
	clr     color.RGBA
	text    string
	textPos image.Point
}

func drawSegmentOutline(img *gocv.Mat, segMask []uint8,
	detectResults []postprocess.DetectResult, minArea float64,
	classNames []string) error {

	width := img.Cols()
	height := img.Rows()
	boxesNum := len(detectResults)

	// create a Mat from the segMask
	maskMat, err := gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8U, segMask)

	if err != nil {
		return fmt.Errorf("error creating mask Mat: %w", err)
	}

	defer maskMat.Close()

	boxLabels := make([]boxLabel, 0)

	// iterate over each unique object ID to isolate the mask
	for objID := 1; objID < boxesNum+1; objID++ {

		// Create a binary mask for the current object (isolate the object by objID)
		objMask := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8U)
		lowerBound := gocv.Scalar{Val1: float64(objID)}
		upperBound := gocv.Scalar{Val1: float64(objID)}
		gocv.InRangeWithScalar(maskMat, lowerBound, upperBound, &objMask)
		defer objMask.Close()

		// Find contours for this object
		contours := gocv.FindContours(objMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
		defer contours.Close() // Ensure to free resources

		// Get the color for this object
		colorIndex := (objID - 1) % len(classColors)
		useClr := classColors[colorIndex]

		// Get the label from the detectResults
		label := classNames[detectResults[objID-1].Class]

		// Calculate the horizontal center of the bounding box
		boundingBox := detectResults[objID-1].Box
		centerX := (boundingBox.Left + boundingBox.Right) / 2

		// Draw contours
		for i := 0; i < contours.Size(); i++ {
			c := contours.At(i)

			// filter out small contours picked up from aliasing/noise in binary mask
			area := gocv.ContourArea(c)

			if area < minArea {
				continue
			}

			log.Printf("contour area=%.2f\n", area)

			approx := gocv.ApproxPolyDP(c, 3, true)

			// Create a PointsVector to hold our PointVector
			ptsVec := gocv.NewPointsVector()

			// Add our approximated PointVector to PointsVector
			ptsVec.Append(approx)

			// Draw polygon lines using PointsVector
			gocv.Polylines(img, ptsVec, true, useClr, 2)

			// Find the topmost point of the contour
			topPoint := findTopPoint(approx)

			// create text for label
			text := fmt.Sprintf("%s %.2f", label, detectResults[objID-1].Probability)
			textSize := gocv.GetTextSize(text, gocv.FontHersheySimplex, 0.4, 1)

			// Adjust the label position so the text is centered horizontally
			labelPosition := image.Pt(centerX-textSize.X/2, topPoint.Y-5) // Adjust Y for padding above the contour

			// create box for placing text on
			bRect := image.Rect(centerX-textSize.X/2-4, topPoint.Y-textSize.Y-8, centerX+textSize.X/2+4, topPoint.Y)
			/*			gocv.Rectangle(img, bRect, useClr, -1)

						// Draw the label at the topmost point of the contour
						gocv.PutTextWithParams(img, text, labelPosition,
							gocv.FontHersheySimplex, 0.4, white, 1,
							gocv.LineAA, false)
			*/
			nextLabel := boxLabel{
				rect:    bRect,
				clr:     useClr,
				text:    text,
				textPos: labelPosition,
			}
			boxLabels = append(boxLabels, nextLabel)

			approx.Close()
			ptsVec.Close()
		}
	}

	// draw all precalculated box labels so they the top most layer on the image
	// and don't get overlapped with segment contour lines
	for _, box := range boxLabels {
		// draw box text gets written on
		gocv.Rectangle(img, box.rect, box.clr, -1)

		// Draw the label over box
		gocv.PutTextWithParams(img, box.text, box.textPos,
			gocv.FontHersheySimplex, 0.4, white, 1,
			gocv.LineAA, false)
	}

	return nil
}

func findTopPoint(approx gocv.PointVector) image.Point {
	topPoint := approx.At(0)
	for i := 1; i < approx.Size(); i++ {
		pt := approx.At(i)
		if pt.Y < topPoint.Y {
			topPoint = pt
		}
	}
	return topPoint
}
