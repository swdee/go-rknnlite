/*
Example code showing how to perform Automatic License Plate Recognition (ALPR)
using a License Plate Detection YOLOv8n and LPRNet model
*/
package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"github.com/swdee/go-rknnlite/preprocess"
	"github.com/swdee/go-rknnlite/render"
	"gocv.io/x/gocv"
	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
	"image"
	"image/color"
	"image/draw"
	"log"
	"os"
	"strings"
	"time"
)

const (
	// Tensor input size for Yolo model
	YoloInputHeight = 640
	YoloInputWidth  = 640
	// Tensor input size for LPRNet model
	LPRInputWidth  = 94
	LPRInputHeight = 24
	// Size of chinese TTF font
	TTFFontSize = 20
	// Flag for specifying text mode
	Chinese = "cn"
	// the amount of pixels to pad the left side of the license plate text
	CnTextLeftPad = 4
	// the height in pixels of the license plate text label
	LPLabelHeight = 24
)

// ALPR defines the Automatic License Plate Recognition struct
type ALPR struct {
	// yoloRT is the runtime with loaded Yolov8n model used for license plate
	// detection
	yoloRT *rknnlite.Runtime
	// lprRT is the runtime with loaded LPRNet model used for license plate
	// recognition
	lprRT *rknnlite.Runtime
	// yoloProcessor is the postprocess used to detect objects in the Yolov8 results
	yoloProcesser *postprocess.YOLOv8
	// lprnetProcessor is the postprocess used to detect license plate results
	lprnetProcesser *postprocess.LPRNet
	// fontFace is the loaded TTF font face
	fontFace font.Face
	// textMode method used to render number plate text on image. default
	// english/latin characters only
	textMode string
}

// NewALPR returns an Automatic License Plate Recognition instance used for
// License plate detection using a YOLOv8n and LPRNet model
func NewALPR(yoloModelFile string, lprModelFile string, ttfFont string) (*ALPR, error) {

	var err error
	a := &ALPR{
		textMode: "en",
	}

	// create rknn runtimes
	a.yoloRT, err = rknnlite.NewRuntime(yoloModelFile, rknnlite.NPUCoreAuto)

	if err != nil {
		return nil, fmt.Errorf("error initializing YOLOv8n RKNN runtime: %w", err)
	}

	// set runtime to leave output tensors as int8
	a.yoloRT.SetWantFloat(false)

	// create rknn runtime instance
	a.lprRT, err = rknnlite.NewRuntime(lprModelFile, rknnlite.NPUCoreAuto)

	if err != nil {
		return nil, fmt.Errorf("error initializing LPRNet RKNN runtime: %w", err)
	}

	// create YOLOv8 post processor
	a.yoloProcesser = postprocess.NewYOLOv8(postprocess.YOLOv8Params{
		BoxThreshold: 0.25,
		NMSThreshold: 0.45,
		// model was trained with single class for plate detection only
		ObjectClassNum:  1,
		MaxObjectNumber: 64,
	})

	// create LPRNet post processor using parameters used during model training
	a.lprnetProcesser = postprocess.NewLPRNet(postprocess.LPRNetParams{
		PlatePositions: 18,
		PlateChars: []string{
			"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
			"苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
			"桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
			"新",
			"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			"A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
			"L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
			"W", "X", "Y", "Z", "I", "O", "-",
		},
	})

	// load ttf font
	err = a.initFont(ttfFont)

	if err != nil {
		return nil, fmt.Errorf("error initializing font face: %w", err)
	}

	return a, nil
}

// initFont loads the TTF font and sets up a new font face
func (a *ALPR) initFont(fontPath string) error {

	// load font data
	fontBytes, err := os.ReadFile(fontPath)

	if err != nil {
		return fmt.Errorf("failed to load font: %w", err)
	}

	// parse the font
	f, err := opentype.Parse(fontBytes)

	if err != nil {
		return fmt.Errorf("failed to parse font: %w", err)
	}

	// create a type face
	a.fontFace, err = opentype.NewFace(f, &opentype.FaceOptions{
		Size:    TTFFontSize,
		DPI:     72,
		Hinting: font.HintingFull,
	})

	if err != nil {
		return fmt.Errorf("failed to create type face: %w", err)
	}

	return nil
}

// Detect takes an image file and detects the license plate and returns an
// annotated image
func (a *ALPR) Detect(img gocv.Mat, resImg *gocv.Mat) (DetectTiming, error) {

	timings := DetectTiming{}

	if img.Empty() {
		return timings, fmt.Errorf("error source Mat is empty")
	}

	// copy image to our result Mat
	img.CopyTo(resImg)

	// convert colorspace and resize image
	rgbImg := gocv.NewMat()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	resizer := preprocess.NewResizer(img.Cols(), img.Rows(),
		YoloInputWidth, YoloInputHeight)

	cropImg := rgbImg.Clone()
	resizer.LetterBoxResize(rgbImg, &cropImg, render.Black)

	defer rgbImg.Close()
	defer cropImg.Close()

	timings.StartYoloInference = time.Now()

	// perform inference on image file
	outputs, err := a.yoloRT.Inference([]gocv.Mat{cropImg})

	if err != nil {
		return timings, fmt.Errorf("runtime inferencing failed with error: %w", err)
	}

	timings.EndYoloInference = time.Now()

	detectResults := a.yoloProcesser.DetectObjects(outputs, resizer)
	dResults := detectResults.GetDetectResults()

	timings.EndYoloDetect = time.Now()
	timings.StartPlateRecognition = time.Now()

	for _, detResult := range dResults {

		newLeft := detResult.Box.Left
		newTop := detResult.Box.Top
		newRight := detResult.Box.Right
		newBottom := detResult.Box.Bottom

		// rectangle of the license plate region
		rect := image.Rect(newLeft, newTop, newRight, newBottom)

		plateNumber, err := a.processPlate(resImg, rect)

		timings.EndPlateRecognition = time.Now()

		if err != nil {
			return timings, fmt.Errorf("error processing plate: %w", err)
		}

		text := fmt.Sprintf("%s (%.1f%%)", plateNumber, detResult.Probability*100)

		// draw box around license plate region
		gocv.Rectangle(resImg, rect, color.RGBA{R: 0, G: 0, B: 255, A: 0}, 2)

		// blank picture area to overlay text on
		textRect := image.Rect(newLeft-1, newTop-LPLabelHeight, newRight+1, newTop)
		gocv.Rectangle(resImg, textRect, color.RGBA{R: 0, G: 0, B: 255, A: 0}, -1)

		if a.textMode == Chinese {
			// Put text - Chinese character support
			a.putChineseText(resImg, text, newLeft-1, newTop-LPLabelHeight, LPLabelHeight,
				color.RGBA{R: 0, G: 0, B: 255, A: 255})
		} else {
			// Put text - Latin characters only
			gocv.PutText(resImg, strings.ToUpper(text), image.Pt(newLeft+4, newTop-6),
				gocv.FontHersheyDuplex, 0.6, color.RGBA{R: 255, G: 255, B: 255, A: 0}, 1)
		}
	}

	timings.EndPlateProcessing = time.Now()

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		return timings, fmt.Errorf("Error freeing Outputs: %w", err)
	}

	return timings, nil
}

// Close all RKNN runtime resources
func (a *ALPR) Close() error {

	// close runtimes and release resources
	err := a.yoloRT.Close()

	if err != nil {
		return fmt.Errorf("error closing YOLOv8n RKNN runtime: %w", err)
	}

	err = a.lprRT.Close()

	if err != nil {
		return fmt.Errorf("error closing LPRNet RKNN runtime: %w", err)
	}

	return nil
}

// processPlate takes a region of and image and extracts the license plate
// number text
func (a *ALPR) processPlate(img *gocv.Mat, rect image.Rectangle) (string, error) {

	// crop image to number plate detection region
	region := img.Region(rect)

	// resize image to 94x24
	cropImg := gocv.NewMat()
	scaleSize := image.Pt(LPRInputWidth, LPRInputHeight)
	gocv.Resize(region, &cropImg, scaleSize, 0, 0, gocv.InterpolationArea)

	defer cropImg.Close()

	// perform inference on image file
	outputs, err := a.lprRT.Inference([]gocv.Mat{cropImg})

	if err != nil {
		return "", fmt.Errorf("runtime inferencing failed with error: %w", err)
	}

	// read number plates from outputs
	plates := a.lprnetProcesser.ReadPlates(outputs)

	// we should only have one results, so return error if more than one.
	plateNumber := "ERROR"

	if len(plates) == 1 {
		plateNumber = plates[0]
	}

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		return "", fmt.Errorf("error freeing Outputs: %w", err)
	}

	return plateNumber, nil
}

// SetTextMode sets the drawing method used to render number plate of image.
// A value of "cn" supports chinese characters but is slow to render.  The
// default method "en" only support Latin characters and is fast.
func (a *ALPR) SetTextMode(mode string) {
	if strings.ToLower(mode) == Chinese {
		a.textMode = Chinese
	}
}

// TextBoundary defines the text boundaries calculated for some text in a
// specific font face
type TextBoundary struct {
	TotalWidth int
	Height     int
}

// textBoundary takes the license plate text and works out the total width
// and height of largest character
func (a *ALPR) textBoundary(text []rune) TextBoundary {

	tb := TextBoundary{}

	if len(text) == 0 {
		// text is empty
		return tb
	}

	// for each character measure it to get text font sizing
	for i := range text {
		bounds, advance, ok := a.fontFace.GlyphBounds(text[i])

		if !ok {
			// failed to get glyph bounds
			continue
		}

		textWidth := advance.Round()
		textHeight := (bounds.Max.Y - bounds.Min.Y).Round()

		// we record the character with largest height
		if textHeight > tb.Height {
			tb.Height = textHeight
		}

		tb.TotalWidth += textWidth
	}

	return tb
}

// putChineseText draws the license plate number on source Mat image using TTF
// chinese font.  x & y parameters are the coordinates on the image where the
// text region is to be drawn.  height is the height of the text label region
// to be drawn upon.
func (a *ALPR) putChineseText(img *gocv.Mat, text string, x, y, height int,
	bgColor color.RGBA) error {

	// convert text to rune
	runes := []rune(text)

	if len(runes) == 0 {
		return fmt.Errorf("text string is empty")
	}

	txtBounds := a.textBoundary(runes)
	textWidth := txtBounds.TotalWidth + CnTextLeftPad

	// create an image for the text
	rgba := image.NewRGBA(image.Rect(0, 0, textWidth, height))
	draw.Draw(rgba, rgba.Bounds(), &image.Uniform{C: bgColor}, image.Point{}, draw.Src)

	dr := &font.Drawer{
		Dst:  rgba,
		Src:  image.NewUniform(color.RGBA{255, 255, 255, 255}),
		Face: a.fontFace,
		Dot: fixed.Point26_6{
			// the multiply by 64 converts pixel sizing to fixed-point sizing
			X: CnTextLeftPad * 64,
			Y: fixed.Int26_6(txtBounds.Height * 64),
		},
	}
	dr.DrawString(text)

	// convert image.RGBA to gocv.Mat
	imgRGBA, err := gocv.NewMatFromBytes(rgba.Bounds().Dy(), rgba.Bounds().Dx(),
		gocv.MatTypeCV8UC4, rgba.Pix)

	if err != nil {
		return fmt.Errorf("error creating Mat from RGBA: %v", err)
	}

	defer imgRGBA.Close()

	// convert RGBA to BGR for overlay
	gocv.CvtColor(imgRGBA, &imgRGBA, gocv.ColorRGBAToBGR)

	// create a ROI from the destination image where to overlay
	roi := img.Region(image.Rect(x, y, x+textWidth, y+height))
	defer roi.Close()

	// Overlay text image onto the designated region of the main image
	imgRGBA.CopyTo(&roi)

	return nil
}

// DetectTiming is a struct of Times that occured during Detect() inferencing to be
// used for calculating inference times
type DetectTiming struct {
	StartYoloInference    time.Time
	EndYoloInference      time.Time
	EndYoloDetect         time.Time
	StartPlateRecognition time.Time
	EndPlateRecognition   time.Time
	EndPlateProcessing    time.Time
}

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	yoloModelFile := flag.String("m", "../data/lpd-yolov8n-640-640-rk3588.rknn", "RKNN compiled YOLO model file")
	lprModelFile := flag.String("l", "../data/lprnet-rk3588.rknn", "RKNN compiled LPRNet model file")
	imgFile := flag.String("i", "../data/car-cn.jpg", "Image file to run object detection on")
	saveFile := flag.String("o", "../data/car-cn-alpr-out.jpg", "The output JPG file with object detection markers")
	ttfFont := flag.String("f", "../data/fzhei-b01s-regular.ttf", "The TTF font to use")
	textMode := flag.String("t", "cn", "The text drawing mode [cn|en]")

	flag.Parse()

	err := rknnlite.SetCPUAffinity(rknnlite.RK3588FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %w", err)
	}

	alpr, err := NewALPR(*yoloModelFile, *lprModelFile, *ttfFont)

	if err != nil {
		log.Fatal("Error initializing ALPR: ", err)
	}

	alpr.SetTextMode(*textMode)

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	// create Mat for annotated image
	resImg := gocv.NewMat()

	defer img.Close()
	defer resImg.Close()

	// run image through ALPR detection
	timings, err := alpr.Detect(img, &resImg)

	if err != nil {
		log.Fatal("Error occurred on ALPR Detect: ", err)
	}

	log.Printf("Model first run speed: YOLO inference=%s, YOLO post processing=%s, Plate recognition=%s, Plate post processing=%s, Total time=%s\n",
		timings.EndYoloInference.Sub(timings.StartYoloInference).String(),
		timings.EndYoloDetect.Sub(timings.EndYoloInference).String(),
		timings.EndPlateRecognition.Sub(timings.StartPlateRecognition).String(),
		timings.EndPlateProcessing.Sub(timings.EndPlateRecognition).String(),
		timings.EndPlateProcessing.Sub(timings.StartYoloInference).String(),
	)

	// Save the result
	if ok := gocv.IMWrite(*saveFile, resImg); !ok {
		log.Fatal("Failed to save the image")
	}

	log.Printf("Saved object detection result to %s\n", *saveFile)

	// optional code.  run benchmark to get average time of 10 runs
	runBenchmark(alpr, img)

	err = alpr.Close()

	if err != nil {
		log.Fatal("Error closing ALPR: ", err)
	}

	log.Println("done")
}

func runBenchmark(alpr *ALPR, img gocv.Mat) {

	count := 50
	start := time.Now()

	// create Mat for annotated image
	resImg := gocv.NewMat()
	defer resImg.Close()

	for i := 0; i < count; i++ {

		// run image through ALPR detection
		_, err := alpr.Detect(img, &resImg)

		if err != nil {
			log.Fatal("Error occurred on ALPR Detect: ", err)
		}
	}

	end := time.Now()
	total := end.Sub(start)
	avg := total / time.Duration(count)

	log.Printf("Benchmark time=%s, count=%d, average total time=%s\n",
		total.String(), count, avg.String(),
	)
}
