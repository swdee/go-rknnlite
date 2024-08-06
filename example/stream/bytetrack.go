package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"github.com/swdee/go-rknnlite/tracker"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"log"
	"net/http"
	"strings"
	"time"
)

var (
	// FPS is the number of FPS to simulate
	FPS         = 30
	FPSinterval = time.Duration(float64(time.Second) / float64(FPS))

	clrPink   = color.RGBA{R: 255, G: 0, B: 255, A: 255}
	clrRed    = color.RGBA{R: 255, G: 0, B: 0, A: 255}
	clrBlack  = color.RGBA{R: 0, G: 0, B: 0, A: 255}
	clrWhite  = color.RGBA{R: 255, G: 255, B: 255, A: 255}
	clrYellow = color.RGBA{R: 255, G: 255, B: 50, A: 255}
)

// ImgScale holds the scale factor for images between their video source size
// and input tensor size
type ImgScale struct {
	Width  float32
	Height float32
}

// Timing is a struct to hold timers used for finding execution time
// for various parts of the process
type Timing struct {
	ProcessStart       time.Time
	DetObjStart        time.Time
	DetObjInferenceEnd time.Time
	DetObjEnd          time.Time
	TrackerStart       time.Time
	TrackerEnd         time.Time
	RenderingStart     time.Time
	ProcessEnd         time.Time
}

// ResultFrame is a struct to wrap the gocv byte buffer and error result
type ResultFrame struct {
	Buf *gocv.NativeByteBuffer
	Err error
}

// YOLOProcessor defines an interface for different versions of YOLO
// models used for object detection
type YOLOProcessor interface {
	DetectObjects(outputs *rknnlite.Outputs) []postprocess.DetectResult
}

// Processor is a struct that holds a YOLOProcessor.
type Processor struct {
	process YOLOProcessor
}

// NewProcessor creates a new Processor instance with the given YOLOProcessor.
func NewProcessor(process YOLOProcessor) *Processor {
	return &Processor{process: process}
}

// DetectObjects delegates the object detection task to the underlying YOLOProcessor.
func (p *Processor) DetectObjects(outputs *rknnlite.Outputs) []postprocess.DetectResult {
	return p.process.DetectObjects(outputs)
}

// Demo defines the struct for running the object tracking demo
type Demo struct {
	// vidBuffer buffers the video frames into memory
	vidBuffer []gocv.Mat
	// pool of rknnlite runtimes to perform inference in parallel
	pool *rknnlite.Pool
	// process is a YOLO object detection processor
	process *Processor
	// labels are the COCO labels the YOLO model was trained on
	labels []string
	// inputAttrs are the model tensor input attributes
	inputAttrs []rknnlite.TensorAttr
	// scale holds the scale factor between video frames resolution and
	// model tensor input size
	scale ImgScale
	// limitObjs restricts object detection results to be only those provided
	limitObjs []string
}

// NewDemo returns and instance of Demo, a streaming HTTP server showing
// video with object detection
func NewDemo(vidFile, modelFile, labelFile string, poolSize int,
	modelType string) (*Demo, error) {

	d := &Demo{
		limitObjs: make([]string, 0),
	}

	err := d.bufferVideo(vidFile)

	if err != nil {
		return nil, fmt.Errorf("Error buffering video: %w", err)
	}

	// create new pool
	d.pool, err = rknnlite.NewPool(poolSize, modelFile)

	if err != nil {
		log.Fatalf("Error creating RKNN pool: %v\n", err)
	}

	// set runtime to leave output tensors as int8
	d.pool.SetWantFloat(false)

	// create YOLOv5 post processor
	switch modelType {
	case "v8":
		d.process = NewProcessor(postprocess.NewYOLOv8(postprocess.YOLOv8COCOParams()))
	case "v5":
		d.process = NewProcessor(postprocess.NewYOLOv5(postprocess.YOLOv5COCOParams()))
	default:
		log.Fatal("Unknown model type, use 'v5' or 'v8'")
	}

	// load in Model class names
	d.labels, err = rknnlite.LoadLabels(labelFile)

	if err != nil {
		return nil, fmt.Errorf("Error loading model labels: %w", err)
	}

	// query Input tensors to get model  size
	rt := d.pool.Get()
	d.inputAttrs, err = rt.QueryInputTensors()
	d.pool.Return(rt)

	if err != nil {
		return nil, fmt.Errorf("Error querying Input Tensors: %w", err)
	}

	log.Printf("Model Input Tensor Dimensions %dx%d", int(d.inputAttrs[0].Dims[1]), int(d.inputAttrs[0].Dims[2]))

	// get image frame size and calculate scale factor
	d.calcScaleFactor()

	return d, nil
}

// LimitObjects limits the object detection kind to the labels provided, eg:
// limit to just "person".  Provide a comma delimited list of labels to
// restrict to.
func (d *Demo) LimitObjects(lim string) {

	words := strings.Split(lim, ",")

	for _, word := range words {
		trimmed := strings.TrimSpace(word)

		// check if word is an actual label in our labels file
		if containsStr(d.labels, trimmed) {
			d.limitObjs = append(d.limitObjs, trimmed)
		}
	}

	log.Printf("Limiting object detection class to: %s\n", strings.Join(d.limitObjs, ", "))
}

// containsStr is a function that takes a string slice and checks if a given
// string exists in the slice
func containsStr(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}

	return false
}

// calcScaleFactor calculates the scale factor between the video image frame
// size and that of the Model input tensor size
func (d *Demo) calcScaleFactor() {

	// get size of video image frame
	width := d.vidBuffer[0].Cols()
	height := d.vidBuffer[0].Rows()

	d.scale = ImgScale{
		Width:  float32(width) / float32(d.inputAttrs[0].Dims[1]),
		Height: float32(height) / float32(d.inputAttrs[0].Dims[2]),
	}

	log.Printf("Scale factor: Width=%.3f, Height=%.3f\n", d.scale.Width, d.scale.Height)
}

// bufferVideo reads in the video frames and saves them to a buffer
func (d *Demo) bufferVideo(vidFile string) error {

	// open handle to read frames of video file
	video, err := gocv.VideoCaptureFile(vidFile)

	if err != nil {
		return err
	}

	defer video.Close()

	d.vidBuffer = make([]gocv.Mat, 0)

	for {
		img := gocv.NewMat()

		// read the next frame from the video
		if ok := video.Read(&img); !ok {
			// reached last video frame
			break
		}

		// Check if the frame is empty
		if img.Empty() {
			continue
		}

		// push frame onto buffer
		d.vidBuffer = append(d.vidBuffer, img)
	}

	return nil
}

// Stream is the HTTP handler function used to stream video frames to browser
func (d *Demo) Stream(w http.ResponseWriter, r *http.Request) {

	log.Printf("New client connection established\n")

	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")

	// pointer to position in video buffer
	frameNum := -1

	// create a bytetracker for tracking detected objects
	// you must create a new instance of byteTrack per stream as it keeps a
	// record of past object detections for tracking
	byteTrack := tracker.NewBYTETracker(FPS, FPS*10, 0.5, 0.6, 0.8)

	// create a trails history
	trail := tracker.NewTrail(90, d.scale.Width, d.scale.Height)

	// create Mat for annotated image
	resImg := gocv.NewMat()
	defer resImg.Close()

	// used for calculating FPS
	frameCount := 0
	startTime := time.Now()
	fps := float64(0)

	ticker := time.NewTicker(FPSinterval)
	defer ticker.Stop()

	// chan to receive processed frames
	recvFrame := make(chan ResultFrame, 30)

loop:
	for {
		select {
		case <-r.Context().Done():
			log.Printf("Client disconnected\n")
			break loop

		// simulate reading 30FPS web camera
		case <-ticker.C:

			// increment pointer to next image in the video buffer
			frameNum++
			if frameNum > len(d.vidBuffer)-1 {
				// last frame reached so loop back to start of video
				frameNum = 0
				// clear tracker data
				byteTrack.Reset()
				// clear trail data
				trail.Reset()
			}

			go d.ProcessFrame(d.vidBuffer[frameNum], recvFrame, fps, frameNum,
				byteTrack, trail)

		case buf := <-recvFrame:

			if buf.Err != nil {
				log.Printf("Error occured during ProcessFrame: %v", buf.Err)

			} else {
				// Write the image to the response writer
				w.Write([]byte("--frame\r\n"))
				w.Write([]byte("Content-Type: image/jpeg\r\n\r\n"))
				w.Write(buf.Buf.GetBytes())
				w.Write([]byte("\r\n"))

				// Flush the buffer
				flusher, ok := w.(http.Flusher)
				if ok {
					flusher.Flush()
				}
			}

			buf.Buf.Close()

			// calculate FPS
			frameCount++
			elapsed := time.Since(startTime).Seconds()

			if elapsed >= 1.0 {
				fps = float64(frameCount) / elapsed
				frameCount = 0
				startTime = time.Now()
			}
		}
	}
}

// ProcessFrame takes an image from the video and runs inference/object
// detection on it, annotates the image and returns the result encoded
// as a JPG file
func (d *Demo) ProcessFrame(img gocv.Mat, retChan chan<- ResultFrame,
	fps float64, frameNum int, byteTrack *tracker.BYTETracker, trail *tracker.Trail) {

	timing := &Timing{
		ProcessStart: time.Now(),
	}

	resImg := gocv.NewMat()
	defer resImg.Close()

	// run object detection on frame
	detObjs, err := d.DetectObjects(img, frameNum, timing)

	if err != nil {
		log.Printf("Error detecting objects: %v", err)
	}

	// track detected objects
	timing.TrackerStart = time.Now()
	trackObjs, err := byteTrack.Update(tracker.DetectionsToObjects(detObjs))
	timing.TrackerEnd = time.Now()

	// add tracked objects to history trail
	for _, trackObj := range trackObjs {
		trail.Add(trackObj)
	}

	// copy the source image and annotate the copy
	img.CopyTo(&resImg)
	d.AnnotateImg(resImg, trackObjs, trail, fps, frameNum, timing)

	// Encode the image to JPEG format
	buf, err := gocv.IMEncode(".jpg", resImg)

	res := ResultFrame{
		Buf: buf,
		Err: err,
	}

	retChan <- res
}

// AnnotateImg draws the detection boxes and processing statistics on the given
// image Mat
func (d *Demo) AnnotateImg(img gocv.Mat, trackResults []*tracker.STrack,
	trail *tracker.Trail, fps float64, frameNum int, timing *Timing) {

	objCnt := 0
	timing.RenderingStart = time.Now()

	for _, tResult := range trackResults {

		// exclude objects detected that are not a given class/label
		if len(d.limitObjs) > 0 {
			if !containsStr(d.limitObjs, d.labels[tResult.GetLabel()]) {
				continue
			}
		}

		objCnt++

		text := fmt.Sprintf("%s %d",
			d.labels[tResult.GetLabel()], tResult.GetTrackID())

		// calculate the coordinates in the original image
		originalLeft := int(tResult.GetRect().TLX() * d.scale.Width)
		originalTop := int(tResult.GetRect().TLY() * d.scale.Height)
		originalRight := int(tResult.GetRect().BRX() * d.scale.Width)
		originalBottom := int(tResult.GetRect().BRY() * d.scale.Height)

		// Draw rectangle around detected object
		rect := image.Rect(originalLeft, originalTop, originalRight, originalBottom)
		gocv.Rectangle(&img, rect, clrPink, 1)

		// draw trail line showing tracking history
		points := trail.GetPoints(tResult.GetTrackID())

		if len(points) > 2 {
			// draw trail
			for i := 1; i < len(points); i++ {
				// draw line segment of trail
				gocv.Line(&img,
					image.Pt(points[i-1].X, points[i-1].Y),
					image.Pt(points[i].X, points[i].Y),
					clrYellow, 1,
				)

				if i == len(points)-1 {
					// draw center point circle on current rect/box
					gocv.Circle(&img, image.Pt(points[i].X, points[i].Y), 3, clrPink, -1)
				}
			}
		}

		// create box for placing text on
		textSize := gocv.GetTextSize(text, gocv.FontHersheySimplex, 0.4, 1)
		bRect := image.Rect(originalLeft, originalTop-textSize.Y-8, originalLeft+textSize.X+4, originalTop)
		gocv.Rectangle(&img, bRect, clrPink, -1)

		// put text with detection class/label and tracking id
		gocv.PutTextWithParams(&img, text, image.Pt(originalLeft+4, originalTop-5),
			gocv.FontHersheySimplex, 0.4, clrBlack,
			1, gocv.LineAA, false)
	}

	timing.ProcessEnd = time.Now()

	// calculate processing lag
	lag := time.Since(timing.ProcessStart).Milliseconds() - int64(FPS)

	// blank out background video
	rect := image.Rect(0, 0, img.Cols(), 36)
	gocv.Rectangle(&img, rect, clrBlack, -1) // -1 fills the rectangle

	// add FPS, object count, and frame number to top of image
	gocv.PutTextWithParams(&img, fmt.Sprintf("Frame: %d, FPS: %.2f, Lag: %dms, Objects: %d", frameNum, fps, lag, objCnt),
		image.Pt(4, 14), gocv.FontHersheySimplex, 0.5, clrPink, 1,
		gocv.LineAA, false)

	// add inference stats to top of image
	gocv.PutTextWithParams(&img, fmt.Sprintf("Inference: %.2fms, Post Processing: %.2fms, Tracking: %.2fms, Rendering: %.2fms, Total Time: %.2fms",
		float32(timing.DetObjInferenceEnd.Sub(timing.DetObjStart))/float32(time.Millisecond),
		float32(timing.DetObjEnd.Sub(timing.DetObjInferenceEnd))/float32(time.Millisecond),
		float32(timing.TrackerEnd.Sub(timing.TrackerStart))/float32(time.Millisecond),
		float32(timing.ProcessEnd.Sub(timing.RenderingStart))/float32(time.Millisecond),
		float32(timing.ProcessEnd.Sub(timing.ProcessStart))/float32(time.Millisecond),
	),
		image.Pt(4, 30), gocv.FontHersheySimplex, 0.5, clrPink, 1,
		gocv.LineAA, false)
}

// DetectObjects takes a raw video frame and runs YOLO inference on it to detect
// objects
func (d *Demo) DetectObjects(img gocv.Mat, frameNum int, timing *Timing) ([]postprocess.DetectResult, error) {

	timing.DetObjStart = time.Now()

	// convert colorspace and resize image
	rgbImg := gocv.NewMat()
	defer rgbImg.Close()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	cropImg := rgbImg.Clone()
	defer cropImg.Close()
	scaleSize := image.Pt(int(d.inputAttrs[0].Dims[1]), int(d.inputAttrs[0].Dims[2]))
	gocv.Resize(rgbImg, &cropImg, scaleSize, 0, 0, gocv.InterpolationArea)

	// perform inference on image file
	rt := d.pool.Get()
	outputs, err := rt.Inference([]gocv.Mat{cropImg})
	d.pool.Return(rt)

	if err != nil {
		return nil, fmt.Errorf("Runtime inferencing failed with error: %w", err)
	}

	timing.DetObjInferenceEnd = time.Now()

	detectResults := d.process.DetectObjects(outputs)

	timing.DetObjEnd = time.Now()

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	return detectResults, nil
}

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/yolov5s-640-640-rk3588.rknn", "RKNN compiled YOLO model file")
	modelType := flag.String("t", "v5", "Version of YOLO model [v5|v8]")
	vidFile := flag.String("v", "../data/palace.mp4", "Video file to run object detection and tracking on")
	labelFile := flag.String("l", "../data/coco_80_labels_list.txt", "Text file containing model labels")
	httpAddr := flag.String("a", "localhost:8080", "HTTP Address to run server on, format address:port")
	poolSize := flag.Int("s", 3, "Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3")
	limitLabels := flag.String("x", "", "Comma delimited list of labels (COCO) to restrict object tracking to")

	flag.Parse()

	err := rknnlite.SetCPUAffinity(rknnlite.RK3588FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %w", err)
	}

	demo, err := NewDemo(*vidFile, *modelFile, *labelFile, *poolSize, *modelType)

	if err != nil {
		log.Fatalf("Error creating demo: %v", err)
	}

	if *limitLabels != "" {
		demo.LimitObjects(*limitLabels)
	}

	http.HandleFunc("/stream", demo.Stream)

	// start http server
	log.Println(fmt.Sprintf("Open browser and view video at http://%s/stream",
		*httpAddr))
	log.Fatal(http.ListenAndServe(*httpAddr, nil))
}
