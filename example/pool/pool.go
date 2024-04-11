/*
Running multiple Runtimes in a Pool allows you to take advantage of all three
NPU cores to significantly reduce average inferencing time.
*/
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
	quiet := flag.Bool("q", false, "Run in quiet mode, don't display individual inference results")

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

	log.Println("Running...")

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
				processFile(rt, filepath.Join(*imgDir, file.Name()), *quiet)
				pool.Return(rt)
			}(pool, rt, file)
		}
	}

	// calculate average inference
	numFiles := (*repeat * len(files))
	end := time.Since(start)
	avg := (end.Seconds() / float64(numFiles)) * 1000

	log.Printf("Processed %d images in %s, average inference per image is %.2fms\n",
		numFiles, end.String(), avg)

	pool.Close()
}

func processFile(rt *rknnlite.Runtime, file string, quiet bool) {

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
	_, err := rt.Inference([]gocv.Mat{cropImg})

	end := time.Since(start)

	if err != nil {
		log.Printf("Runtime inferencing failed with error: ", err)
	}

	if !quiet {
		log.Printf("File %s, inference time %dms\n", file, end.Milliseconds())
	}
}
