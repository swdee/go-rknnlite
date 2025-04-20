/*
Example code showing how to perform OCR on an image using PaddleOCR recognition
*/
package main

import (
	"flag"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"github.com/swdee/go-rknnlite/preprocess"
	"github.com/swdee/go-rknnlite/render"
	"gocv.io/x/gocv"
	"log"
	"os"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../../data/ppocrv4_rec-rk3588.rknn", "RKNN compiled model file")
	imgFile := flag.String("i", "../../data/ppocr-rec-test.png", "Image file to run inference on")
	keysFile := flag.String("k", "../../data/ppocr_keys_v1.txt", "Text file containing OCR character keys")
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

	// set runtime to pass input gocv.Mat's to Inference() function as float32
	// to RKNN backend
	rt.SetInputTypeFloat32(true)

	// optional querying of model file tensors and SDK version for printing
	// to stdout.  not necessary for production inference code
	err = rt.Query(os.Stdout)

	if err != nil {
		log.Fatal("Error querying runtime: ", err)
	}

	// load in Model character labels
	modelChars, err := rknnlite.LoadLabels(*keysFile)

	if err != nil {
		log.Fatal("Error loading model OCR character keys: ", err)
	}

	// check that we have as many modelChars as tensor outputs dimension
	if len(modelChars) != int(rt.OutputAttrs()[0].Dims[2]) {
		log.Fatalf("OCR character keys text input has %d characters and does "+
			"not match the required number in the Model of %d",
			len(modelChars), rt.OutputAttrs()[0].Dims[2])
	}

	// create PPOCR post processor
	ppocrProcessor := postprocess.NewPPOCRRecognise(postprocess.PPOCRRecogniseParams{
		ModelChars:   modelChars,
		OutputSeqLen: int(rt.InputAttrs()[0].Dims[2]) / 8, // modelWidth (320/8)
	})

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// resize image to 320x48 and keep aspect ratio, centered with black letterboxing
	resizedImg := gocv.NewMat()

	resizer := preprocess.NewResizer(img.Cols(), img.Rows(),
		int(rt.InputAttrs()[0].Dims[2]), int(rt.InputAttrs()[0].Dims[1]),
	)

	resizer.LetterBoxResize(img, &resizedImg, render.Black)

	// convert image to float32 in 3 channels
	resizedImg.ConvertTo(&resizedImg, gocv.MatTypeCV32FC3)

	// normalize the image (img - 127.5) / 127.5
	resizedImg.AddFloat(-127.5)
	resizedImg.DivideFloat(127.5)

	defer img.Close()
	defer resizedImg.Close()
	defer resizer.Close()

	start := time.Now()

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{resizedImg})

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	endInference := time.Now()

	results := ppocrProcessor.Recognise(outputs)

	endRecognise := time.Now()

	log.Printf("Model first run speed: inference=%s, post processing=%s, total time=%s\n",
		endInference.Sub(start).String(),
		endRecognise.Sub(endInference).String(),
		endRecognise.Sub(start).String(),
	)

	for _, result := range results {
		log.Printf("Recognize result: %s, score=%.2f", result.Text, result.Score)
	}

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// optional code.  run benchmark to get average time of 10 runs
	runBenchmark(rt, ppocrProcessor, []gocv.Mat{resizedImg})

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")
}

func runBenchmark(rt *rknnlite.Runtime, ppocrProcessor *postprocess.PPOCRRecognise,
	mats []gocv.Mat) {

	count := 100
	start := time.Now()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		// post process
		_ = ppocrProcessor.Recognise(outputs)

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
