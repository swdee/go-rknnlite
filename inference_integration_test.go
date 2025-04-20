//go:build integration
// +build integration

package rknnlite

import (
	"gocv.io/x/gocv"
	"image"
	"os"
	"testing"
)

func TestMobileNetTop5(t *testing.T) {

	modelFile := os.Getenv("RKNN_MODEL")

	if modelFile == "" {
		t.Fatalf("No Model file provided in RKNN_MODEL")
	}

	imgFile := os.Getenv("RKNN_IMAGE")

	if imgFile == "" {
		t.Fatalf("No Image file provided in RKNN_IMAGE")
	}

	// Initialize runtime
	rt, err := NewRuntime(modelFile, NPUCoreAuto)

	if err != nil {
		t.Fatalf("NewRuntime failed: %v", err)
	}

	defer rt.Close()

	// load image
	img := gocv.IMRead(imgFile, gocv.IMReadColor)

	if img.Empty() {
		t.Fatalf("Error reading image from: %s", imgFile)
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

	// run inference
	outputs, err := rt.Inference([]gocv.Mat{cropImg})

	if err != nil {
		t.Fatalf("Inference error: %v", err)
	}

	defer func() {
		if err := outputs.Free(); err != nil {
			t.Errorf("Free Outputs: %v", err)
		}
	}()

	// Extract Top5
	top5 := GetTop5(outputs.Output)

	if len(top5) != 5 {
		t.Fatalf("expected 5 results, got %d", len(top5))
	}

	// Probabilities must be in [0,1] and descending
	for i, p := range top5 {

		if p.Probability < 0 || p.Probability > 1 {
			t.Errorf("entry %d: probability %v out of [0,1]", i, p.Probability)
		}

		if i > 0 && p.Probability > top5[i-1].Probability {
			t.Errorf("probabilities not descending: index %d has %v > previous %v",
				i, p.Probability, top5[i-1].Probability)
		}
	}

	// Label indices must be in range [0, numClasses)
	numClasses := int(rt.OutputAttrs()[0].Dims[1])

	for i, p := range top5 {
		if int(p.LabelIndex) < 0 || int(p.LabelIndex) >= numClasses {
			t.Errorf("entry %d: label index %d out of range [0,%d)", i, p.LabelIndex, numClasses)
		}
	}

	// Sanity check: at least one probability above a tiny epsilon
	const eps = 1e-3
	var found bool

	for _, p := range top5 {
		if p.Probability > eps {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("all probabilities ≤ %v, something's wrong", eps)
	}
}
