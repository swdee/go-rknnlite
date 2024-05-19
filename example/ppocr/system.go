package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"gocv.io/x/gocv"
	"image"
	"log"
	"math"
	"sort"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	detectModelFile := flag.String("d", "../data/ppocrv4_det-rk3588.rknn", "RKNN compiled model file for OCR Detection")
	recogniseModelFile := flag.String("r", "../data/ppocrv4_rec-rk3588.rknn", "RKNN compiled model file for OCR Recognition")
	keysFile := flag.String("k", "../data/ppocr_keys_v1.txt", "Text file containing OCR character keys")
	imgFile := flag.String("i", "../data/ppocr-det-test.png", "Image file to run inference on")
	flag.Parse()

	err := rknnlite.SetCPUAffinity(rknnlite.RK3588FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %w", err)
	}

	// create rknn runtime instance
	detectRt, err := rknnlite.NewRuntime(*detectModelFile, rknnlite.NPUCoreAuto)

	if err != nil {
		log.Fatal("Error initializing Detect RKNN runtime: ", err)
	}

	recogniseRt, err := rknnlite.NewRuntime(*recogniseModelFile, rknnlite.NPUCoreAuto)

	if err != nil {
		log.Fatal("Error initializing Recognise RKNN runtime: ", err)
	}

	// set runtime to pass input gocv.Mat's to Inference() function as float32
	// to RKNN backend
	recogniseRt.SetInputTypeFloat32(true)

	recogniseInputAttrs, recogniseOutputAttrs := optionalQueries(recogniseRt)
	detectInputAttrs, _ := optionalQueries(detectRt)

	// load in Model character labels
	modelChars, err := rknnlite.LoadLabels(*keysFile)

	if err != nil {
		log.Fatal("Error loading model OCR character keys: ", err)
	}

	// check that we have as many modelChars as tensor outputs dimension
	if len(modelChars) != int(recogniseOutputAttrs[0].Dims[2]) {
		log.Fatalf("OCR character keys text input has %d characters and does "+
			"not match the required number in the Model of %d",
			len(modelChars), recogniseOutputAttrs[0].Dims[2])
	}

	// create PPOCR post processor
	recogniseProcessor := postprocess.NewPPOCRRecognise(postprocess.PPOCRRecogniseParams{
		ModelChars:   modelChars,
		OutputSeqLen: int(recogniseInputAttrs[0].Dims[2]) / 8, // modelWidth (320/8)
	})

	detectProcessor := postprocess.NewPPOCRDetect(postprocess.PPOCRDetectParams{
		Threshold:    0.3,
		BoxThreshold: 0.6,
		Dilation:     false,
		BoxType:      "poly",
		UnclipRatio:  1.5,
		ScoreMode:    "slow",
		ModelWidth:   int(detectInputAttrs[0].Dims[2]),
		ModelHeight:  int(detectInputAttrs[0].Dims[1]),
	})

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// resize image to 480x480 and keep aspect ratio, centered with black letterboxing
	resizedImg := gocv.NewMat()
	resizeKeepAspectRatio(img, &resizedImg, int(detectInputAttrs[0].Dims[2]), int(detectInputAttrs[0].Dims[1]))

	defer img.Close()
	defer resizedImg.Close()

	start := time.Now()

	// perform inference on image file
	outputs, err := detectRt.Inference([]gocv.Mat{resizedImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	endInference := time.Now()

	// work out scale ratio between source image and resized image
	scaleW := float32(img.Cols()) / float32(resizedImg.Cols())
	scaleH := float32(img.Rows()) / float32(resizedImg.Rows())

	results := detectProcessor.Detect(outputs, scaleW, scaleH)

	endDetect := time.Now()

	log.Printf("Model first run speed: inference=%s, post processing=%s, total time=%s\n",
		endInference.Sub(start).String(),
		endDetect.Sub(endInference).String(),
		endDetect.Sub(start).String(),
	)

	// sort results in order from top to bottom and left to right
	SortBoxes(&results)

	for _, result := range results {
		for i, box := range result.Box {
			fmt.Printf("[%d]: [(%d, %d), (%d, %d), (%d, %d), (%d, %d)] %f\n",
				i,
				box.LeftTop.X, box.LeftTop.Y,
				box.RightTop.X, box.RightTop.Y,
				box.RightBottom.X, box.RightBottom.Y,
				box.LeftBottom.X, box.LeftBottom.Y,
				box.Score)

			// crop source image to get text area
			rect := image.Rect(box.LeftTop.X, box.LeftTop.Y, box.RightBottom.X, box.RightBottom.Y)
			region := img.Region(rect)

			// perform text recognition
			recogniseTextBlock(recogniseRt, recogniseProcessor, region,
				int(recogniseInputAttrs[0].Dims[2]), int(recogniseInputAttrs[0].Dims[1]))
		}
	}

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// close runtime and release resources
	err = detectRt.Close()

	if err != nil {
		log.Fatal("Error closing Detection RKNN runtime: ", err)
	}

	err = recogniseRt.Close()

	if err != nil {
		log.Fatal("Error closing Recognition RKNN runtime: ", err)
	}

	log.Println("done")
}

func recogniseTextBlock(recogniseRt *rknnlite.Runtime,
	recogniseProcessor *postprocess.PPOCRRecognise, img gocv.Mat,
	inWidth, inHeight int) {

	// resize image to 320x48 and keep aspect ratio, centered with black letterboxing
	resizedImg := gocv.NewMat()
	resizeKeepAspectRatio(img, &resizedImg, inWidth, inHeight)

	// convert image to float32 in 3 channels
	resizedImg.ConvertTo(&resizedImg, gocv.MatTypeCV32FC3)

	// normalize the image (img - 127.5) / 127.5
	resizedImg.AddFloat(-127.5)
	resizedImg.DivideFloat(127.5)

	defer resizedImg.Close()

	// perform inference on image file
	outputs, err := recogniseRt.Inference([]gocv.Mat{resizedImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	results := recogniseProcessor.Recognise(outputs)

	for _, result := range results {
		log.Printf("Recognize result: %s, score=%.2f", result.Text, result.Score)
	}

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}
}

// CompareBox compares two boxes
func CompareBox(box1, box2 postprocess.PPOCRBox) bool {

	if box1.LeftTop.Y < box2.LeftTop.Y {
		return true
	} else if box1.LeftTop.Y == box2.LeftTop.Y {
		return box1.LeftTop.X < box2.LeftTop.X
	} else {
		return false
	}
}

// SortBoxes sorts the boxes in PPOCRDetectResult and adjusts the order
func SortBoxes(detectResults *[]postprocess.PPOCRDetectResult) {

	for _, result := range *detectResults {

		boxes := result.Box
		sort.Slice(boxes, func(i, j int) bool {
			return CompareBox(boxes[i], boxes[j])
		})

		if len(boxes) == 0 {
			continue
		}

		for i := 0; i < len(boxes)-1; i++ {
			for j := i; j >= 0; j-- {
				if math.Abs(float64(boxes[j+1].LeftTop.Y-boxes[j].LeftTop.Y)) < 10 && (boxes[j+1].LeftTop.X < boxes[j].LeftTop.X) {
					boxes[j], boxes[j+1] = boxes[j+1], boxes[j]
				}
			}
		}
	}
}