package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
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
	FPS         = int64(30)
	FPSinterval = time.Duration(float64(time.Second) / float64(FPS))
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
	ProcessEnd         time.Time
}

// ResultFrame is a struct to wrap the gocv byte buffer and error result
type ResultFrame struct {
	Buf *gocv.NativeByteBuffer
	Err error
}

// Demo defines the struct for running the object tracking demo
type Demo struct {
	// vidBuffer buffers the video frames into memory
	vidBuffer []gocv.Mat
	// pool of rknnlite runtimes to perform inference in parallel
	pool *rknnlite.Pool
	// process is a YOLO object detection processor
	process *postprocess.YOLOv5
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
func NewDemo(vidFile, modelFile, labelFile string, poolSize int) (*Demo, error) {

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
	d.process = postprocess.NewYOLOv5(postprocess.YOLOv5COCOParams())

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
				frameNum = 0
			}

			go d.ProcessFrame(d.vidBuffer[frameNum], recvFrame, fps, frameNum)

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
	fps float64, frameNum int) {

	timing := &Timing{
		ProcessStart: time.Now(),
	}

	resImg := gocv.NewMat()
	defer resImg.Close()

	// run object detection on frame
	objects, err := d.DetectObjects(img, frameNum, timing)

	if err != nil {
		log.Printf("Error detecting objects: %v", err)
	}

	// copy the source image and annotate the copy
	img.CopyTo(&resImg)
	d.AnnotateImg(resImg, objects, fps, frameNum, timing)

	// Encode the image to JPEG format
	buf, err := gocv.IMEncode(".jpg", resImg)

	res := ResultFrame{
		Buf: buf,
		Err: err,
	}

	retChan <- res
}

// AnnotateImg draws annotations on the image to provide statistics and render
// the bounding box around detected objects
func (d *Demo) AnnotateImg(img gocv.Mat, detectResults []postprocess.DetectResult,
	fps float64, frameNum int, timing *Timing) error {

	objCnt := 0

	for _, detResult := range detectResults {

		// exclude objects detected that are not a given class/label
		if !containsStr(d.limitObjs, d.labels[detResult.Class]) {
			continue
		}

		objCnt++

		text := fmt.Sprintf("%s %.1f%%", d.labels[detResult.Class], detResult.Probability*100)

		// calculate the coordinates in the original image
		originalLeft := int(float32(detResult.Box.Left) * d.scale.Width)
		originalTop := int(float32(detResult.Box.Top) * d.scale.Height)
		originalRight := int(float32(detResult.Box.Right) * d.scale.Width)
		originalBottom := int(float32(detResult.Box.Bottom) * d.scale.Height)

		// Draw rectangle around detected object
		rect := image.Rect(originalLeft, originalTop, originalRight, originalBottom)
		gocv.Rectangle(&img, rect, color.RGBA{R: 0, G: 0, B: 255, A: 0}, 2)

		// Put text
		gocv.PutText(&img, text, image.Pt(originalLeft, originalTop+12), gocv.FontHersheyDuplex, 0.4, color.RGBA{R: 255, G: 255, B: 255, A: 0}, 1)
	}

	timing.ProcessEnd = time.Now()

	// calculate processing lag
	lag := time.Since(timing.ProcessStart).Milliseconds() - FPS

	// add FPS, object count, and frame number to top of image
	gocv.PutText(&img, fmt.Sprintf("Frame: %d, FPS: %.2f, Lag: %dms, Objects: %d", frameNum, fps, lag, objCnt),
		image.Pt(4, 14), gocv.FontHersheyDuplex, 0.5, color.RGBA{R: 255, G: 0, B: 0, A: 255}, 1)

	// add inference stats to top of image
	gocv.PutText(&img, fmt.Sprintf("Inference: %.2fms, Post Processing: %.2fms, Rendering: %.2fms, Total Time: %.2fms",
		float32(timing.DetObjInferenceEnd.Sub(timing.DetObjStart))/float32(time.Millisecond),
		float32(timing.DetObjEnd.Sub(timing.DetObjInferenceEnd))/float32(time.Millisecond),
		float32(timing.ProcessEnd.Sub(timing.DetObjEnd))/float32(time.Millisecond),
		float32(timing.ProcessEnd.Sub(timing.ProcessStart))/float32(time.Millisecond),
	),
		image.Pt(4, 30), gocv.FontHersheyDuplex, 0.5, color.RGBA{R: 255, G: 0, B: 0, A: 255}, 1)

	return nil

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

	demo, err := NewDemo(*vidFile, *modelFile, *labelFile, *poolSize)

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
