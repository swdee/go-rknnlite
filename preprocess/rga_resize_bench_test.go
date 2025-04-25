package preprocess

import (
	"image"
	"testing"

	"gocv.io/x/gocv"
)

func BenchmarkResizeRGA(b *testing.B) {

	// Load and prepare Mats *before* timing
	img := gocv.IMRead("../example/data/palace.jpg", gocv.IMReadColor)

	if img.Empty() {
		b.Fatal("benchmark: failed to load ../example/data/palace.jpg")
	}

	defer img.Close()

	// convert BGR to BGRA
	srcBGRA := gocv.NewMat()
	defer srcBGRA.Close()
	gocv.CvtColor(img, &srcBGRA, gocv.ColorBGRToBGRA)

	// half-size output
	dstBGRA := gocv.NewMatWithSize(srcBGRA.Rows()/2, srcBGRA.Cols()/2, gocv.MatTypeCV8UC4)
	defer dstBGRA.Close()

	if err := InitRGA(srcBGRA, dstBGRA); err != nil {
		b.Fatal(err)
	}
	defer CloseRGA()

	// Reset timer so we only measure ResizeRGA
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := ResizeRGAFrame(); err != nil {
			b.Fatalf("ResizeRGA failed: %v", err)
		}
	}
}

func BenchmarkGocvResize(b *testing.B) {

	// load & prep once
	img := gocv.IMRead("../example/data/palace.jpg", gocv.IMReadColor)
	if img.Empty() {
		b.Fatal("benchmark: failed to load ../example/data/palace.jpg")
	}
	defer img.Close()

	// convert BGR to BGRA
	srcBGRA := gocv.NewMat()
	defer srcBGRA.Close()
	gocv.CvtColor(img, &srcBGRA, gocv.ColorBGRToBGRA)

	// half-size output
	dstGocv := gocv.NewMatWithSize(srcBGRA.Rows()/2, srcBGRA.Cols()/2, srcBGRA.Type())
	defer dstGocv.Close()

	// target size for Resize
	size := image.Pt(dstGocv.Cols(), dstGocv.Rows())

	// only measure the GoCV Resize call
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gocv.Resize(srcBGRA, &dstGocv, size, 0, 0, gocv.InterpolationLinear)
	}
}
