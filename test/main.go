package main

import (
	"fmt"
	"github.com/swdee/go-rknnlite/preprocess"
	"gocv.io/x/gocv"
	"log"
)

func main() {

	imgFile := "../example/data/palace.jpg"

	img := gocv.IMRead(imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", imgFile)
	}

	defer img.Close()

	// convert to BGRA
	srcBGRA := gocv.NewMat()
	defer srcBGRA.Close()
	gocv.CvtColor(img, &srcBGRA, gocv.ColorBGRToBGRA)

	// prepare destination Mat (e.g. half size)
	dstBGRA := gocv.NewMatWithSize(srcBGRA.Rows()/2, srcBGRA.Cols()/2, gocv.MatTypeCV8UC4)
	defer dstBGRA.Close()

	// do the RGA resize
	if err := preprocess.ResizeRGA(srcBGRA, dstBGRA); err != nil {
		log.Fatalf("resize error: %v", err)
	}

	// convert back to BGR for JPEG output
	dstBGR := gocv.NewMat()
	defer dstBGR.Close()
	gocv.CvtColor(dstBGRA, &dstBGR, gocv.ColorBGRAToBGR)

	gocv.IMWrite("/tmp/resized.jpg", dstBGR)

	fmt.Print("done\n")
}
