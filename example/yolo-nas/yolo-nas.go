package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/preprocess"
	"github.com/swdee/go-rknnlite/render"
	"gocv.io/x/gocv"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	//modelFile := flag.String("m", "../data/models/rk3588/yolov8s-rk3588.rknn", "RKNN compiled YOLO model file")
	modelFile := flag.String("m", "/home/rock/devel/yolo_nas_s_manual.rknn", "RKNN compiled YOLO model file")
	imgFile := flag.String("i", "../data/bus.jpg", "Image file to run object detection on")
	//labelFile := flag.String("l", "../data/coco_80_labels_list.txt", "Text file containing model labels")
	//saveFile := flag.String("o", "../data/bus-yolo-nas-out.jpg", "The output JPG file with object detection markers")
	rkPlatform := flag.String("p", "rk3588", "Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588]")

	flag.Parse()

	err := rknnlite.SetCPUAffinityByPlatform(*rkPlatform, rknnlite.FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %v\n", err)
	}

	// check if user specified model file or if default is being used.  if default
	// then pick the default platform model to use.
	if f := flag.Lookup("m"); f != nil && f.Value.String() == f.DefValue && *rkPlatform != "rk3588" {
		*modelFile = strings.ReplaceAll(*modelFile, "rk3588", *rkPlatform)
	}

	// create rknn runtime instance
	rt, err := rknnlite.NewRuntimeByPlatform(*rkPlatform, *modelFile)

	if err != nil {
		log.Fatal("Error initializing RKNN runtime: ", err)
	}

	// set runtime to leave output tensors as int8
	rt.SetWantFloat(false)

	// optional querying of model file tensors and SDK version for printing
	// to stdout.  not necessary for production inference code
	err = rt.Query(os.Stdout)

	if err != nil {
		log.Fatal("Error querying runtime: ", err)
	}

	// create YOLOv8 post processor
	//yoloProcesser := postprocess.NewYOLOv8(postprocess.YOLOv8COCOParams())

	// load in Model class names
	//classNames, err := rknnlite.LoadLabels(*labelFile)

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
		int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))

	cropImg := rgbImg.Clone()
	resizer.LetterBoxResize(rgbImg, &cropImg, render.Black)

	defer img.Close()
	defer rgbImg.Close()
	defer cropImg.Close()

	start := time.Now()

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{cropImg})

	endInference := time.Now()

	dets := decodeRKNN(outputs.Output[0].BufInt, outputs.Output[1].BufInt,
		640, 640, 0.45, 0.5,
		resizer)

	endDetect := time.Now()

	for _, det := range dets {
		fmt.Println(det)
	}

	log.Printf("Model first run speed: inference=%s, post processing=%s\n",
		endInference.Sub(start).String(),
		endDetect.Sub(endInference).String(),
	)

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// optional code.  run benchmark to get average time
	runBenchmark(rt, []gocv.Mat{cropImg}, resizer, img)

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")

}

func runBenchmark(rt *rknnlite.Runtime,
	mats []gocv.Mat, resizer *preprocess.Resizer,
	srcImg gocv.Mat) {

	count := 100
	start := time.Now()

	for i := 0; i < count; i++ {
		// perform inference on image file
		outputs, err := rt.Inference(mats)

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

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

// detection holds one final box + score + class
type Detection struct {
	X1, Y1, X2, Y2 int
	Score          float32
	ClassID        int
}

// dequantParams for each tensor
const (
	boxZP    = -90
	boxScale = 3.387419
	clsZP    = -128
	clsScale = 0.003898
)

func decodeRKNN(boxTensor []int8, clsTensor []int8, imgW, imgH int,
	confThreshold float32, iouThreshold float32,
	resizer *preprocess.Resizer) []Detection {
	const numAnchors = 8400
	const numClasses = 80

	// 1) Dequantize & decode each anchor
	dets := make([]Detection, 0, 256)
	for a := 0; a < numAnchors; a++ {

		// -- dequantize raw box outputs
		// corner-based decode (x1,y1,x2,y2)
		x1p := (float32(boxTensor[a*4+0]) - boxZP) * boxScale
		y1p := (float32(boxTensor[a*4+1]) - boxZP) * boxScale
		x2p := (float32(boxTensor[a*4+2]) - boxZP) * boxScale
		y2p := (float32(boxTensor[a*4+3]) - boxZP) * boxScale

		// compute w,h if you need them later (e.g. for area)
		//w := x2p - x1p
		//h := y2p - y1p

		// undo letterbox
		x1 := int((x1p - float32(resizer.XPad())) / resizer.ScaleFactor())
		y1 := int((y1p - float32(resizer.YPad())) / resizer.ScaleFactor())
		x2 := int((x2p - float32(resizer.XPad())) / resizer.ScaleFactor())
		y2 := int((y2p - float32(resizer.YPad())) / resizer.ScaleFactor())

		// clamp to original image
		if x1 < 0 {
			x1 = 0
		}
		if y1 < 0 {
			y1 = 0
		}
		if x2 > resizer.SrcWidth() {
			x2 = resizer.SrcWidth()
		}
		if y2 > resizer.SrcHeight() {
			y2 = resizer.SrcHeight()
		}

		// -- find the class with highest probability (outputs are already quantized [0..1])
		bestProb := float32(0)
		bestClass := 0
		for c := 0; c < numClasses; c++ {
			// direct dequant → a probability in [0..1]
			p := (float32(clsTensor[a*numClasses+c]) - clsZP) * clsScale
			if p > bestProb {
				bestProb = p
				bestClass = c
			}
		}

		// 2) threshold
		if bestProb < confThreshold {
			continue
		}

		dets = append(dets, Detection{
			X1: x1, Y1: y1,
			X2: x2, Y2: y2,
			Score:   bestProb,
			ClassID: bestClass,
		})
	}

	// 3) Non-Max Suppression
	return nonMaxSuppression(dets, iouThreshold)
}

// simple NMS by score
func nonMaxSuppression(dets []Detection, iouThresh float32) []Detection {
	// sort descending by Score
	sort.Slice(dets, func(i, j int) bool {
		return dets[i].Score > dets[j].Score
	})
	keep := make([]Detection, 0, len(dets))
	used := make([]bool, len(dets))

	for i := range dets {
		if used[i] {
			continue
		}
		keep = append(keep, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if used[j] {
				continue
			}
			if iou(dets[i], dets[j]) > iouThresh {
				used[j] = true
			}
		}
	}
	return keep
}

// IoU of two boxes
func iou(a, b Detection) float32 {
	xx1 := float32(math.Max(float64(a.X1), float64(b.X1)))
	yy1 := float32(math.Max(float64(a.Y1), float64(b.Y1)))
	xx2 := float32(math.Min(float64(a.X2), float64(b.X2)))
	yy2 := float32(math.Min(float64(a.Y2), float64(b.Y2)))

	w := xx2 - xx1
	h := yy2 - yy1
	if w <= 0 || h <= 0 {
		return 0
	}
	inter := w * h
	areaA := float32((a.X2 - a.X1) * (a.Y2 - a.Y1))
	areaB := float32((b.X2 - b.X1) * (b.Y2 - b.Y1))
	return inter / (areaA + areaB - inter)
}
