package main

import (
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess"
	"github.com/swdee/go-rknnlite/preprocess"
	"github.com/swdee/go-rknnlite/render"
	"github.com/swdee/go-rknnlite/tracker"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"log"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var (
	// FPS is the number of FPS to simulate
	FPS         = 30
	FPSinterval = time.Duration(float64(time.Second) / float64(FPS))

	clrBlack = color.RGBA{R: 0, G: 0, B: 0, A: 255}
	clrWhite = color.RGBA{R: 255, G: 255, B: 255, A: 255}
)

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
	DetectObjects(outputs *rknnlite.Outputs,
		resizer *preprocess.Resizer) postprocess.DetectionResult
}

type VideoFormat string

const (
	VideoFile VideoFormat = "file"
	Webcam    VideoFormat = "webcam"
)

// VideoSource defines the video/media source to use for playback.
type VideoSource struct {
	Path     string
	Format   VideoFormat
	Settings string
	Codec    string
	// camera validated settings
	width  int
	height int
	fps    int
}

// Validate and extract the video source settings
func (v *VideoSource) Validate() error {
	// get camera settings
	pattern := `^(\d+)x(\d+)@(\d+)$`
	re := regexp.MustCompile(pattern)

	matches := re.FindStringSubmatch(v.Settings)

	if len(matches) == 0 {
		return fmt.Errorf("Camera settings does not match the pattern <width>x<height>@<fps>")
	}

	// ignore errors since it passed pattern matching above
	width, _ := strconv.Atoi(matches[1])
	height, _ := strconv.Atoi(matches[2])
	fps, _ := strconv.Atoi(matches[3])

	v.width = width
	v.height = height
	v.fps = fps

	// check Codec
	v.Codec = strings.ToUpper(v.Codec)

	if v.Codec != "YUYV" {
		v.Codec = "MJPG"
	}

	return nil
}

// Demo defines the struct for running the object tracking demo
type Demo struct {
	// vidSrc holds details on our video source for playback
	vidSrc *VideoSource
	// vidBuffer buffers the video frames into memory
	vidBuffer []gocv.Mat
	// pool of rknnlite runtimes to perform inference in parallel
	pool *rknnlite.Pool
	// process is a YOLO object detection processor
	process YOLOProcessor
	// labels are the COCO labels the YOLO model was trained on
	labels []string
	// limitObjs restricts object detection results to be only those provided
	limitObjs []string
	// resizer handles scaling of source image to input tensors
	resizer *preprocess.Resizer
	// modelType is the type of YOLO model to use as processor that was passed
	// as a command line flag
	modelType string
	// renderFormat indicates which rendering type to use with instance
	// segmentation, outline or mask
	renderFormat string
}

// NewDemo returns and instance of Demo, a streaming HTTP server showing
// video with object detection
func NewDemo(vidSrc *VideoSource, modelFile, labelFile string, poolSize int,
	modelType string, renderFormat string, cores []rknnlite.CoreMask) (*Demo, error) {

	var err error

	d := &Demo{
		vidSrc:    vidSrc,
		limitObjs: make([]string, 0),
	}

	if vidSrc.Format == VideoFile {
		// buffer video file
		err = d.bufferVideo(vidSrc.Path)

		if err != nil {
			return nil, fmt.Errorf("Error buffering video: %w", err)
		}
	}

	// create new pool
	d.pool, err = rknnlite.NewPool(poolSize, modelFile, cores)

	if err != nil {
		log.Fatalf("Error creating RKNN pool: %v\n", err)
	}

	// set runtime to leave output tensors as int8
	d.pool.SetWantFloat(false)

	// create resizer to handle scaling of input image to inference tensor
	// input size requirements
	rt := d.pool.Get()

	if vidSrc.Format == Webcam {
		d.resizer = preprocess.NewResizer(d.vidSrc.width, d.vidSrc.height,
			int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))
	} else {
		d.resizer = preprocess.NewResizer(d.vidBuffer[0].Cols(), d.vidBuffer[0].Rows(),
			int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))
	}

	d.pool.Return(rt)

	// create YOLOv5 post processor
	switch modelType {
	case "v8":
		d.process = postprocess.NewYOLOv8(postprocess.YOLOv8COCOParams())
	case "v5":
		d.process = postprocess.NewYOLOv5(postprocess.YOLOv5COCOParams())
	case "v10":
		d.process = postprocess.NewYOLOv10(postprocess.YOLOv10COCOParams())
	case "v11":
		d.process = postprocess.NewYOLOv11(postprocess.YOLOv11COCOParams())
	case "x":
		d.process = postprocess.NewYOLOX(postprocess.YOLOXCOCOParams())

	case "v5seg":
		d.process = postprocess.NewYOLOv5Seg(postprocess.YOLOv5SegCOCOParams())
		// force FPS to 10, as we don't have enough CPU power to do 30 FPS
		FPS = 10
		FPSinterval = time.Duration(float64(time.Second) / float64(FPS))
		log.Println("***WARNING*** Instance Segmentation requires a lot of CPU, downgraded to 10 FPS")
	case "v8seg":
		d.process = postprocess.NewYOLOv8Seg(postprocess.YOLOv8SegCOCOParams())
		// force FPS to 10, as we don't have enough CPU power to do 30 FPS
		FPS = 10
		FPSinterval = time.Duration(float64(time.Second) / float64(FPS))
		log.Println("***WARNING*** Instance Segmentation requires a lot of CPU, downgraded to 10 FPS")

	case "v8pose":
		d.process = postprocess.NewYOLOv8Pose(postprocess.YOLOv8PoseCOCOParams())

	case "v8obb":
		d.process = postprocess.NewYOLOv8obb(postprocess.YOLOv8obbDOTAv1Params())

	default:
		log.Fatal("Unknown model type, use 'v5', 'v8', 'v10', 'v11', 'x', 'v5seg', 'v8seg', 'v8pose', or 'v8obb'")
	}

	d.modelType = modelType
	d.renderFormat = renderFormat

	// load in Model class names
	d.labels, err = rknnlite.LoadLabels(labelFile)

	if err != nil {
		return nil, fmt.Errorf("Error loading model labels: %w", err)
	}

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

// startWebcam starts the web camera and copies frames to a channel.   function
// is to be called from a goroutine as its blocking
func (d *Demo) startWebcam(framesCh chan gocv.Mat, exitCh chan struct{}) {

	var err error
	var webcam *gocv.VideoCapture

	devNum, _ := strconv.Atoi(d.vidSrc.Path)
	webcam, err = gocv.VideoCaptureDevice(devNum)

	if err != nil {
		log.Printf("Error opening web camera: %v", err)
		return
	}

	defer webcam.Close()

	webcam.Set(gocv.VideoCaptureFOURCC, webcam.ToCodec(d.vidSrc.Codec))
	webcam.Set(gocv.VideoCaptureFrameWidth, float64(d.vidSrc.width))
	webcam.Set(gocv.VideoCaptureFrameHeight, float64(d.vidSrc.height))
	webcam.Set(gocv.VideoCaptureFPS, float64(d.vidSrc.fps))

	camImg := gocv.NewMat()
	defer camImg.Close()

loop:
	for {
		select {
		case <-exitCh:
			log.Printf("Closing webcamera")
			break loop

		default:

			if ok := webcam.Read(&camImg); !ok {
				// error reading webcamera frame
				continue
			}
			if camImg.Empty() {
				continue
			}

			// send frame to channel, copy to avoid race conditions
			frameCopy := camImg.Clone()
			framesCh <- frameCopy
		}
	}
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
	trail := tracker.NewTrail(90)

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

	// create channel to receive frames from the webcam
	cameraFrames := make(chan gocv.Mat, 8)
	closeCamera := make(chan struct{})

	if d.vidSrc.Format == Webcam {
		go d.startWebcam(cameraFrames, closeCamera)
	}

loop:
	for {
		select {
		case <-r.Context().Done():
			log.Printf("Client disconnected\n")
			closeCamera <- struct{}{}
			break loop

		// receive web camera frames
		case frame := <-cameraFrames:
			frameNum++

			go d.ProcessFrame(frame, recvFrame, fps, frameNum,
				byteTrack, trail, true)

		// simulate reading 30FPS web camera
		case <-ticker.C:
			if d.vidSrc.Format == Webcam {
				// skip this routine if running webcamera video source
				continue
			}

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
				byteTrack, trail, false)

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
	fps float64, frameNum int, byteTrack *tracker.BYTETracker,
	trail *tracker.Trail, closeImg bool) {

	timing := &Timing{
		ProcessStart: time.Now(),
	}

	resImg := gocv.NewMat()
	defer resImg.Close()

	// copy source image
	img.CopyTo(&resImg)

	// run object detection on frame
	detectObjs, err := d.DetectObjects(resImg, frameNum, timing)

	if err != nil {
		log.Printf("Error detecting objects: %v", err)
		return
	}

	if detectObjs == nil {
		// no objects detected
		return
	}

	detectResults := detectObjs.GetDetectResults()

	// track detected objects
	timing.TrackerStart = time.Now()

	trackObjs, err := byteTrack.Update(
		postprocess.DetectionsToObjects(detectResults),
	)

	timing.TrackerEnd = time.Now()

	// add tracked objects to history trail
	for _, trackObj := range trackObjs {
		trail.Add(trackObj)
	}

	// segment mask creation must be done after object tracking, as the tracked
	// objects can be different to the object detection results so need to
	// strip those objects from the mask
	var segMask postprocess.SegMask
	var keyPoints [][]postprocess.KeyPoint

	if d.modelType == "v5seg" {
		segMask = d.process.(*postprocess.YOLOv5Seg).TrackMask(detectObjs,
			trackObjs, d.resizer)

	} else if d.modelType == "v8seg" {
		segMask = d.process.(*postprocess.YOLOv8Seg).TrackMask(detectObjs,
			trackObjs, d.resizer)

	} else if d.modelType == "v8pose" {
		keyPoints = d.process.(*postprocess.YOLOv8Pose).GetPoseEstimation(detectObjs)
	}

	timing.DetObjEnd = time.Now()

	// annotate the image
	d.AnnotateImg(resImg, detectResults, trackObjs, segMask, keyPoints,
		trail, fps, frameNum, timing)

	// Encode the image to JPEG format
	buf, err := gocv.IMEncode(".jpg", resImg)

	res := ResultFrame{
		Buf: buf,
		Err: err,
	}

	if closeImg {
		// close copied web camera frame
		img.Close()
	}

	retChan <- res
}

// LimitResults takes the tracked results and strips out any results that
// we don't want to track
func (d *Demo) LimitResults(trackResults []*tracker.STrack) []*tracker.STrack {

	if len(d.limitObjs) == 0 {
		return trackResults

	}

	// strip out and detected objects we don't want to track
	var newTrackResults []*tracker.STrack

	for _, tResult := range trackResults {

		// exclude objects detected that are not a given class/label
		if len(d.limitObjs) > 0 {
			if !containsStr(d.limitObjs, d.labels[tResult.GetLabel()]) {
				continue
			}
		}

		newTrackResults = append(newTrackResults, tResult)
	}

	return newTrackResults
}

// AnnotateImg draws the detection boxes and processing statistics on the given
// image Mat
func (d *Demo) AnnotateImg(img gocv.Mat, detectResults []postprocess.DetectResult,
	trackResults []*tracker.STrack,
	segMask postprocess.SegMask, keyPoints [][]postprocess.KeyPoint,
	trail *tracker.Trail, fps float64,
	frameNum int, timing *Timing) {

	timing.RenderingStart = time.Now()

	// strip out tracking results for classes of objects we don't want
	trackResults = d.LimitResults(trackResults)
	objCnt := len(trackResults)

	if d.modelType == "v5seg" || d.modelType == "v8seg" {

		if d.renderFormat == "mask" {
			render.TrackerMask(&img, segMask.Mask, trackResults, detectResults, 0.5)

			render.TrackerBoxes(&img, trackResults, d.labels,
				render.DefaultFont(), 1)
		} else {
			render.TrackerOutlines(&img, segMask.Mask, trackResults, detectResults,
				1000, d.labels, render.DefaultFont(), 2, 5)
		}

	} else if d.modelType == "v8pose" {

		render.PoseKeyPoints(&img, keyPoints, 2)

		render.TrackerBoxes(&img, trackResults, d.labels,
			render.DefaultFont(), 1)

	} else if d.modelType == "v8obb" {

		render.TrackerOrientedBoundingBoxes(&img, trackResults, detectResults,
			d.labels, render.DefaultFontAlign(render.Center), 1)

	} else {
		// draw detection boxes
		render.TrackerBoxes(&img, trackResults, d.labels,
			render.DefaultFont(), 1)
	}

	// draw object trail lines
	if d.modelType != "v8pose" {
		render.Trail(&img, trackResults, trail, render.DefaultTrailStyle())
	}

	timing.ProcessEnd = time.Now()

	// calculate processing lag
	lag := time.Since(timing.ProcessStart).Milliseconds() - int64(FPS)

	// blank out background video
	rect := image.Rect(0, 0, img.Cols(), 36)
	gocv.Rectangle(&img, rect, clrBlack, -1) // -1 fills the rectangle

	// add FPS, object count, and frame number to top of image
	gocv.PutTextWithParams(&img, fmt.Sprintf("Frame: %d, FPS: %.2f, Lag: %dms, Objects: %d", frameNum, fps, lag, objCnt),
		image.Pt(4, 14), gocv.FontHersheySimplex, 0.5, clrWhite, 1,
		gocv.LineAA, false)

	// add inference stats to top of image
	gocv.PutTextWithParams(&img, fmt.Sprintf("Inference: %.2fms, Post Processing: %.2fms, Tracking: %.2fms, Rendering: %.2fms, Total Time: %.2fms",
		float32(timing.DetObjInferenceEnd.Sub(timing.DetObjStart))/float32(time.Millisecond),
		float32(timing.DetObjEnd.Sub(timing.DetObjInferenceEnd))/float32(time.Millisecond),
		float32(timing.TrackerEnd.Sub(timing.TrackerStart))/float32(time.Millisecond),
		float32(timing.ProcessEnd.Sub(timing.RenderingStart))/float32(time.Millisecond),
		float32(timing.ProcessEnd.Sub(timing.ProcessStart))/float32(time.Millisecond),
	),
		image.Pt(4, 30), gocv.FontHersheySimplex, 0.5, clrWhite, 1,
		gocv.LineAA, false)
}

// DetectObjects takes a raw video frame and runs YOLO inference on it to detect
// objects
func (d *Demo) DetectObjects(img gocv.Mat, frameNum int,
	timing *Timing) (postprocess.DetectionResult, error) {

	timing.DetObjStart = time.Now()

	// convert colorspace and resize image
	rgbImg := gocv.NewMat()
	defer rgbImg.Close()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	cropImg := rgbImg.Clone()
	defer cropImg.Close()

	d.resizer.LetterBoxResize(rgbImg, &cropImg, render.Black)

	// perform inference on image file
	rt := d.pool.Get()
	outputs, err := rt.Inference([]gocv.Mat{cropImg})
	d.pool.Return(rt)

	if err != nil {
		return nil, fmt.Errorf("Runtime inferencing failed with error: %w", err)
	}

	timing.DetObjInferenceEnd = time.Now()

	detectObjs := d.process.DetectObjects(outputs, d.resizer)

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	return detectObjs, nil
}

// cameraResFlag is a custom type that tracks whether the CLI flag was explicitly set
type cameraResFlag struct {
	value string
	isSet bool
}

// String implement's the flag.Value interface for cameraResFlag
func (c *cameraResFlag) String() string {
	return c.value
}

// Set
func (c *cameraResFlag) Set(val string) error {
	c.value = val
	c.isSet = true
	return nil
}

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/yolov5s-640-640-rk3588.rknn", "RKNN compiled YOLO model file")
	modelType := flag.String("t", "v5", "Version of YOLO model [v5|v8|v10|v11|x|v5seg|v8seg|v8pose]")
	vidFile := flag.String("v", "../data/palace.mp4", "Video file to run object detection and tracking on or device of web camera when used with -c flag")
	labelFile := flag.String("l", "../data/coco_80_labels_list.txt", "Text file containing model labels")
	httpAddr := flag.String("a", "localhost:8080", "HTTP Address to run server on, format address:port")
	poolSize := flag.Int("s", 3, "Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3")
	limitLabels := flag.String("x", "", "Comma delimited list of labels (COCO) to restrict object tracking to")
	renderFormat := flag.String("r", "outline", "The rendering format used for instance segmentation [outline|mask]")
	codecFormat := flag.String("codec", "mjpg", "Web Camera codec The rendering format [mjpg|yuyv]")

	// Initialize the custom camera resolution flag with a default value
	cameraRes := &cameraResFlag{value: "1280x720@30"}
	flag.Var(cameraRes, "c", "Web Camera resolution in format <width>x<height>@<fps>, eg: 1280x720@30")

	flag.Parse()

	if *poolSize > 33 {
		log.Fatalf("RKNN runtime pool size (flag -s) is to large, a value of 3, 6, 9, or 12 works best")
	}

	// check which video source to use
	var vidSrc *VideoSource

	if cameraRes.isSet {
		vidSrc = &VideoSource{
			Path:     *vidFile,
			Format:   Webcam,
			Settings: cameraRes.value,
			Codec:    *codecFormat,
		}

		err := vidSrc.Validate()

		if err != nil {
			log.Fatalf("Error in video source settings: %v", err)
		}

	} else {
		vidSrc = &VideoSource{
			Path:   *vidFile,
			Format: VideoFile,
		}
	}

	err := rknnlite.SetCPUAffinity(rknnlite.RK3588FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %v\n", err)
	}

	demo, err := NewDemo(vidSrc, *modelFile, *labelFile, *poolSize,
		*modelType, *renderFormat, rknnlite.RK3588)

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
