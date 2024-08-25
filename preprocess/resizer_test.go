package preprocess

import (
	"gocv.io/x/gocv"
	"image/color"
	"testing"
)

var (
	black = color.RGBA{R: 0, G: 0, B: 0, A: 255}
)

func TestLetterBoxResize(t *testing.T) {

	tests := []struct {
		srcWidth      int
		srcHeight     int
		resizeWidth   int
		resizeHeight  int
		expectedXPad  int
		expectedYPad  int
		expectedScale float32
	}{
		{1280, 720, 640, 640, 0, 140, 0.50},
		{800, 1000, 640, 640, 64, 0, 0.64},
		{800, 800, 640, 640, 0, 0, 0.8},
	}

	for _, tc := range tests {
		img := gocv.NewMatWithSize(tc.srcHeight, tc.srcWidth, gocv.MatTypeCV8UC1)

		resizedImg := gocv.NewMat()

		resizer := NewResizer(tc.srcWidth, tc.srcHeight, tc.resizeWidth, tc.resizeHeight)

		resizer.LetterBoxResize(img, &resizedImg, black)

		if resizer.XPad() != tc.expectedXPad || resizer.YPad() != tc.expectedYPad {
			t.Errorf("Test failed for src (%d, %d): Padding values wrong, expected XPad=%d, YPad=%d, got xPad=%d, yPad=%d",
				tc.srcWidth, tc.srcHeight, tc.expectedXPad, tc.expectedYPad, resizer.XPad(), resizer.YPad())
		}

		if resizer.ScaleFactor() != tc.expectedScale {
			t.Errorf("Test failed for src (%d, %d): Scalefactor incorrect, expected %f, got %f",
				tc.srcWidth, tc.srcHeight, tc.expectedScale, resizer.ScaleFactor())
		}

		img.Close()
		resizedImg.Close()
		resizer.Close()
	}
}
