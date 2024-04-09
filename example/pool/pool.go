package main

import (
	"flag"
	"github.com/swdee/go-rknnlite"
	"gocv.io/x/gocv"
	"image"
	"log"
	"os"
	"path/filepath"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/mobilenet_v1-rk3588.rknn", "RKNN compiled model file")
	imgDir := flag.String("d", "../data/imagenet/", "A directory of images to run inference on")
	poolSize := flag.Int("s", 1, "Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3")
	repeat := flag.Int("r", 1, "Repeat processing image directory the specified number of times, use this if you don't have enough images")

	flag.Parse()

	// check dir exists
	info, err := os.Stat(*imgDir)

	if err != nil {
		log.Fatalf("No such image directory %s, error: %v\n", *imgDir, err)
	}

	if !info.IsDir() {
		log.Fatal("Image path is not a directory")
	}

	// create new pool
	pool, err := rknnlite.NewPool(*poolSize, *modelFile)

	if err != nil {
		log.Fatalf("Error creating RKNN pool: %v\n", err)
	}

	// get list of all files in the directory
	files, err := os.ReadDir(*imgDir)

	if err != nil {
		log.Fatalf("Error reading image directory: %v\n", err)
	}

	start := time.Now()

	// repeat processing the specified number of times to increase the number
	// of images processed
	for i := 0; i < *repeat; i++ {
		// process each image
		for _, file := range files {
			// skip directories
			if file.IsDir() {
				continue
			}

			// pool.Get() blocks if no runtimes are available in the pool
			rt := pool.Get()

			go func(pool *rknnlite.Pool, rt *rknnlite.Runtime, file os.DirEntry) {
				processFile(rt, filepath.Join(*imgDir, file.Name()))
				pool.Return(rt)
			}(pool, rt, file)
		}
	}

	log.Printf("Completed in %s\n", time.Since(start).String())

	pool.Close()
}

func processFile(rt *rknnlite.Runtime, file string) {

	// load image
	img := gocv.IMRead(file, gocv.IMReadColor)

	if img.Empty() {
		log.Printf("Error reading image from: ", file)
		return
	}

	start := time.Now()

	// convert colorspace and resize image
	rgbImg := gocv.NewMat()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	cropImg := rgbImg.Clone()
	gocv.Resize(rgbImg, &cropImg, image.Pt(224, 224), 0, 0, gocv.InterpolationArea)

	defer img.Close()
	defer rgbImg.Close()
	defer cropImg.Close()

	// perform inference on image file
	outputs, err := rt.Inference([]gocv.Mat{cropImg})

	exe := time.Since(start)

	if err != nil {
		log.Printf("Runtime inferencing failed with error: ", err)
	}

	for _, next := range rknnlite.GetTop5(outputs) {
		log.Printf("%dms - File[%s] is %3d: %8.6f\n", exe.Milliseconds(),
			file, next.LabelIndex, next.Probability)
		break
	}
}
