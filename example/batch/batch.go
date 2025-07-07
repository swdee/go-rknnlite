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

var (
	// model input tensor dimensions, these values will be set
	// when runtime queries the modelFile being loaded
	height, width, channels, batchSize int
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/models/rk3588/mobilenetv2-batch8-rk3588.rknn", "RKNN compiled model file")
	imgDir := flag.String("d", "../data/imagenet/", "A directory of images to run inference on")
	poolSize := flag.Int("s", 1, "Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3")
	repeat := flag.Int("r", 1, "Repeat processing image directory the specified number of times, use this if you don't have enough images")
	quiet := flag.Bool("q", false, "Run in quiet mode, don't display individual inference results")
	rkPlatform := flag.String("p", "rk3588", "Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588]")

	flag.Parse()

	// set cpu affinity to run on specific CPU cores
	err := rknnlite.SetCPUAffinityByPlatform(*rkPlatform, rknnlite.FastCores)

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

	// create new pool, we pass NPUCoreAuto as RKNN does not allow batch Models
	// to be pinned to specific NPU cores
	useCore := rknnlite.NPUCoreAuto

	if strings.HasPrefix(strings.ToLower(*rkPlatform), "rk356") {
		useCore = rknnlite.NPUSkipSetCore
	}

	pool, err := rknnlite.NewPool(*poolSize, *modelFile,
		[]rknnlite.CoreMask{useCore})

	if err != nil {
		log.Fatalf("Error creating RKNN pool: %v\n", err)
	}

	// set runtime to leave output tensors as int8
	pool.SetWantFloat(false)

	// get a runtime and query the input tensor dimensions of the model
	rt := pool.Get()

	// optional querying of model file tensors and SDK version for printing
	// to stdout.  not necessary for production inference code
	err = rt.Query(os.Stdout)

	if err != nil {
		log.Fatal("Error querying runtime: ", err)
	}

	batchSize = int(rt.InputAttrs()[0].Dims[0])
	width = int(rt.InputAttrs()[0].Dims[1])
	height = int(rt.InputAttrs()[0].Dims[2])
	channels = int(rt.InputAttrs()[0].Dims[3])

	pool.Return(rt)

	// get list of all files in the directory
	entries, err := os.ReadDir(*imgDir)

	if err != nil {
		log.Fatalf("Error reading image directory: %v\n", err)
	}

	var files []string

	for _, e := range entries {
		if e.IsDir() {
			continue
		}

		files = append(files, filepath.Join(*imgDir, e.Name()))
	}

	log.Println("Running...")

	// waitgroup used to wait for all go-routines to complete before closing
	// the pool
	const batchSize = 8
	var wg sync.WaitGroup

	start := time.Now()

	// repeat processing image set the specified number of times
	for i := 0; i < *repeat; i++ {
		// process image files in groups of batchSize
		for offset := 0; offset < len(files); offset += batchSize {

			end := offset + batchSize

			if end > len(files) {
				end = len(files)
			}

			subset := files[offset:end]

			// pool.Get() blocks if no runtimes are available in the pool
			rt := pool.Get()
			wg.Add(1)

			go func(rt *rknnlite.Runtime, batchPaths []string) {
				defer wg.Done()
				processBatch(rt, batchPaths, *quiet)
				pool.Return(rt)
			}(rt, subset)
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

func processBatch(rt *rknnlite.Runtime, paths []string, quiet bool) {

	// create batch
	batch := rknnlite.NewBatch(batchSize, height, width, channels,
		rt.GetInputTypeFloat32())
	defer batch.Close()

	// for each image path, load & preprocess, then Add to batch
	for idx, file := range paths {

		img := gocv.IMRead(file, gocv.IMReadColor)

		if img.Empty() {
			log.Printf("Error reading %s\n", file)
			continue
		}

		defer img.Close()

		// rgb + resize
		rgbImg := gocv.NewMat()
		gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)
		defer rgbImg.Close()

		cropImg := gocv.NewMat()
		gocv.Resize(rgbImg, &cropImg, image.Pt(width, height), 0, 0, gocv.InterpolationArea)
		defer cropImg.Close()

		if err := batch.AddAt(idx, cropImg); err != nil {
			log.Printf("Batch.Add error: %v\n", err)
		}
	}

	// run inference on the entire batch at once
	start := time.Now()
	outputs, err := rt.Inference([]gocv.Mat{batch.Mat()})
	spent := time.Since(start)

	if err != nil {
		log.Printf("Inference error: %v\n", err)
		return
	}

	defer outputs.Free()

	// unpack per image results
	for idx := 0; idx < len(paths); idx++ {

		if quiet {
			continue
		}

		// get int8 output tensor for image at idx
		_, err := batch.GetOutputInt(idx, outputs.Output[0], int(outputs.OutputAttributes().DimForDFL))

		if err != nil {
			log.Printf("GetOutputInt[%d] error: %v\n", idx, err)
			continue
		}

		log.Printf("File %s, inference time %dms\n", paths[idx], spent.Milliseconds())
	}
}
