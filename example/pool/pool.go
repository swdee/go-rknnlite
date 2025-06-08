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
	"strings"
	"sync"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/models/rk3588/mobilenet_v1-rk3588.rknn", "RKNN compiled model file")
	imgDir := flag.String("d", "../data/imagenet/", "A directory of images to run inference on")
	poolSize := flag.Int("s", 1, "Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3")
	repeat := flag.Int("r", 1, "Repeat processing image directory the specified number of times, use this if you don't have enough images")
	quiet := flag.Bool("q", false, "Run in quiet mode, don't display individual inference results")
	cpuaff := flag.String("c", "fast", "CPU Affinity, run on [fast|slow] CPU cores")
	rkPlatform := flag.String("p", "rk3588", "Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588]")

	flag.Parse()

	// set cpu affinity to run on specific CPU cores
	cpumask := rknnlite.FastCores

	if strings.ToLower(*cpuaff) == "slow" {
		cpumask = rknnlite.SlowCores
	}

	err := rknnlite.SetCPUAffinityByPlatform(*rkPlatform, cpumask)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %v\n", err)
	}

	// check dir exists
	info, err := os.Stat(*imgDir)

	if err != nil {
		log.Fatalf("No such image directory %s, error: %v\n", *imgDir, err)
	}

	if !info.IsDir() {
		log.Fatal("Image path is not a directory")
	}

	// check if user specified model file or if default is being used.  if default
	// then pick the default platform model to use.
	if f := flag.Lookup("m"); f != nil && f.Value.String() == f.DefValue && *rkPlatform != "rk3588" {
		*modelFile = strings.ReplaceAll(*modelFile, "rk3588", *rkPlatform)
	}

	// create new pool
	pool, err := rknnlite.NewPoolByPlatform(*rkPlatform, *poolSize, *modelFile)

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

	// waitgroup used to wait for all go-routines to complete before closing
	// the pool
	var wg sync.WaitGroup

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

			wg.Add(1)
			go func(pool *rknnlite.Pool, rt *rknnlite.Runtime, file os.DirEntry) {
				processFile(rt, filepath.Join(*imgDir, file.Name()), *quiet)
				pool.Return(rt)
				wg.Done()
			}(pool, rt, file)
		}
	}

	wg.Wait()

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
		log.Printf("Error reading image from: %s\n", file)
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

	end := time.Since(start)

	if err != nil {
		log.Printf("Runtime inferencing failed with error: %v\n", err)
	}

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Printf("Error freeing Outputs: %v\n", err)
	}

	if !quiet {
		log.Printf("File %s, inference time %dms\n", file, end.Milliseconds())
	}
}
