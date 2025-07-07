package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess/reid"
	"gocv.io/x/gocv"
	"image"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/models/rk3588/osnet-market1501-batch8-rk3588.rknn", "RKNN compiled model file")
	imgFile := flag.String("i", "../data/reid-walking.jpg", "Image file to run inference on")
	objsFile := flag.String("d", "../data/reid-objects.dat", "Data file containing object co-ordinates")
	rkPlatform := flag.String("p", "rk3588", "Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588]")
	euDist := flag.Float64("e", 0.51, "The Euclidean distance [0.0-1.0], a value less than defines a match")
	flag.Parse()

	err := rknnlite.SetCPUAffinityByPlatform(*rkPlatform, rknnlite.FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %v", err)
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

	// load objects file
	objs, err := ParseObjects(*objsFile)

	if err != nil {
		log.Fatal("Error parsing objects: ", err)
	}

	// load image
	img := gocv.IMRead(*imgFile, gocv.IMReadColor)

	if img.Empty() {
		log.Fatal("Error reading image from: ", *imgFile)
	}

	// convert colorspace
	srcImg := gocv.NewMat()
	gocv.CvtColor(img, &srcImg, gocv.ColorBGRToRGB)

	defer img.Close()
	defer srcImg.Close()

	start := time.Now()

	// create a batch to process all images in the compare and dataset's
	// in a single forward pass
	batch := rknnlite.NewBatch(
		int(rt.InputAttrs()[0].Dims[0]),
		int(rt.InputAttrs()[0].Dims[2]),
		int(rt.InputAttrs()[0].Dims[1]),
		int(rt.InputAttrs()[0].Dims[3]),
		rt.GetInputTypeFloat32(),
	)

	// scale size is the size of the input tensor dimensions to scale the object too
	scaleSize := image.Pt(int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))

	// add the compare images to the batch
	for _, cmpObj := range objs.Compare {
		err := AddObjectToBatch(batch, srcImg, cmpObj, scaleSize)

		if err != nil {
			log.Fatal("Error creating batch: ", err)
		}
	}

	// add the dataset images to the batch
	for _, dtObj := range objs.Dataset {
		err := AddObjectToBatch(batch, srcImg, dtObj, scaleSize)

		if err != nil {
			log.Fatal("Error creating batch: ", err)
		}
	}

	defer batch.Close()

	endBatch := time.Now()

	// run inference on the batch
	outputs, err := rt.Inference([]gocv.Mat{batch.Mat()})

	endInference := time.Now()

	if err != nil {
		log.Fatal("Runtime inferencing failed with error: ", err)
	}

	// get total number of compare objects
	totalCmp := len(objs.Compare)

	// compare each object to those objects in the dataset for similarity
	for i, cmpObj := range objs.Compare {
		// get the compare objects output
		cmpOutput, err := batch.GetOutputInt(i, outputs.Output[0], int(outputs.OutputAttributes().DimForDFL))

		if err != nil {
			log.Fatal("Getting output tensor failed with error: ", err)
		}

		log.Printf("Comparing object %d at (%d,%d,%d,%d)\n", i,
			cmpObj.X1, cmpObj.Y1, cmpObj.X2, cmpObj.Y2)

		for j, dtObj := range objs.Dataset {
			// get each objects outputs
			nextOutput, err := batch.GetOutputInt(totalCmp+j, outputs.Output[0], int(outputs.OutputAttributes().DimForDFL))

			if err != nil {
				log.Fatal("Getting output tensor failed with error: ", err)
			}

			dist := CompareObjects(
				cmpOutput,
				nextOutput,
				outputs.OutputAttributes().Scales[0],
				outputs.OutputAttributes().ZPs[0],
			)

			// check euclidean distance to determine match of same person or not
			objRes := "different person"

			if dist < float32(*euDist) {
				objRes = "same person"
			}

			log.Printf("  Object %d at (%d,%d,%d,%d) has euclidean distance: %f (%s)\n",
				j,
				dtObj.X1, dtObj.Y1, dtObj.X2, dtObj.Y2,
				dist, objRes)
		}
	}

	endCompare := time.Now()

	log.Printf("Model first run speed: batch preparation=%s, inference=%s, post processing=%s, total time=%s\n",
		endBatch.Sub(start).String(),
		endInference.Sub(endBatch).String(),
		endCompare.Sub(endInference).String(),
		endCompare.Sub(start).String(),
	)

	// free outputs allocated in C memory after you have finished post processing
	err = outputs.Free()

	if err != nil {
		log.Fatal("Error freeing Outputs: ", err)
	}

	// close runtime and release resources
	err = rt.Close()

	if err != nil {
		log.Fatal("Error closing RKNN runtime: ", err)
	}

	log.Println("done")

	/*
		//CompareObject(rt, srcImg, cmpObj, objs.Dataset)

		//rgbImg := img.Clone()



		frameWidth := 67
		frameHeight := 177

		roiRect1 := image.Rect(497, 195, 497+frameWidth, 195+frameHeight)

		// cklady
		//roiRect1 := image.Rect(0, 0, 134, 361)

		roiImg1 := rgbImg.Region(roiRect1)

		cropImg1 := rgbImg.Clone()
		scaleSize1 := image.Pt(int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))
		gocv.Resize(roiImg1, &cropImg1, scaleSize1, 0, 0, gocv.InterpolationArea)

		defer img.Close()
		defer rgbImg.Close()
		defer cropImg1.Close()
		defer roiImg1.Close()

		gocv.IMWrite("/tmp/frame-master.jpg", cropImg1)

		batch := rt.NewBatch(
			int(rt.InputAttrs()[0].Dims[0]),
			int(rt.InputAttrs()[0].Dims[2]),
			int(rt.InputAttrs()[0].Dims[1]),
			int(rt.InputAttrs()[0].Dims[3]),
		)
		err = batch.Add(cropImg1)

		if err != nil {
			log.Fatal("Error creating batch: ", err)
		}
		defer batch.Close()

		// perform inference on image file
		outputs, err := rt.Inference([]gocv.Mat{batch.Mat()})

		if err != nil {
			log.Fatal("Runtime inferencing failed with error: ", err)
		}

		output, err := batch.GetOutputInt(0, outputs.Output[0], int(outputs.OutputAttributes().DimForDFL))

		if err != nil {
			log.Fatal("Getting output tensor failed with error: ", err)
		}

		fingerPrint := DequantizeAndL2Normalize(
			output,
			outputs.OutputAttributes().Scales[0],
			outputs.OutputAttributes().ZPs[0],
		)

		// seed the EMA fingerprint to the master
		emaFP := make([]float32, len(fingerPrint))
		copy(emaFP, fingerPrint)
		const alpha = 0.9 // smoothing factor

		hash, err := FingerprintHash(fingerPrint)

		if err != nil {
			log.Fatalf("hashing failed: %v", err)
		}

		log.Println("object fingerprint:", hash)

		// free outputs allocated in C memory after you have finished post processing
		err = outputs.Free()

		if err != nil {
			log.Fatal("Error freeing Outputs: ", err)
		}


		// sample 2 images

		yOffsets := []int{1, 195, 388}
		xOffsets := []int{497, 565, 633, 701, 769, 836, 904}

		images := [][]int{}

		for _, ny := range yOffsets {
			for _, nx := range xOffsets {
				images = append(images, []int{nx, ny})
			}
		}

		// ck lady

		//	images := [][]int{
		//		{134, 0, 117, 325},
		//		{251, 0, 75, 208},
		//		{326, 0, 68, 187},
		//	}


		// Image 2
		for frame, next := range images {

			roiRect2 := image.Rect(next[0], next[1], next[0]+frameWidth, next[1]+frameHeight)
			// ck lady
			//roiRect2 := image.Rect(next[0], next[1], next[0]+next[2], next[1]+next[3])
			roiImg2 := rgbImg.Region(roiRect2)

			cropImg2 := rgbImg.Clone()
			scaleSize2 := image.Pt(int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))
			gocv.Resize(roiImg2, &cropImg2, scaleSize2, 0, 0, gocv.InterpolationArea)

			defer cropImg2.Close()
			defer roiImg2.Close()

			gocv.IMWrite(fmt.Sprintf("/tmp/frame-%d.jpg", frame), cropImg2)

			start := time.Now()

			batch.Clear()
			err = batch.Add(cropImg2)

			if err != nil {
				log.Fatal("Error creating batch: ", err)
			}

			outputs, err = rt.Inference([]gocv.Mat{batch.Mat()})

			if err != nil {
				log.Fatal("Runtime inferencing failed with error: ", err)
			}

			endInference := time.Now()

			output, err := batch.GetOutputInt(0, outputs.Output[0], int(outputs.OutputAttributes().DimForDFL))

			if err != nil {
				log.Fatal("Getting output tensor failed with error: ", err)
			}

			fingerPrint2 := DequantizeAndL2Normalize(
				output,
				outputs.OutputAttributes().Scales[0],
				outputs.OutputAttributes().ZPs[0],
			)


			//	sim := CosineSimilarity(fingerPrint, fingerPrint2)
			//	dist := CosineDistance(fingerPrint, fingerPrint2)
			//	fmt.Printf("Frame %d, cosine similarity: %f,  distance=%f\n", frame, sim, dist)


			// compute Euclidean (L2) distance directly
			dist := EuclideanDistance(fingerPrint, fingerPrint2)

			// 3) compute vs EMA
			emaDist := EuclideanDistance(emaFP, fingerPrint2)

			endDetect := time.Now()

			objRes := "different person"
			if emaDist < 0.51 {
				objRes = "same person"
			}

			fmt.Printf("Frame %d, euclidean distance: %f, ema=%f (%s)\n", frame, dist, emaDist, objRes)

			log.Printf(" Inference=%s, detect=%s, total time=%s\n",
				endInference.Sub(start).String(),
				endDetect.Sub(endInference).String(),
				endDetect.Sub(start).String(),
			)

			// free outputs allocated in C memory after you have finished post processing
			err = outputs.Free()

			if err != nil {
				log.Fatal("Error freeing Outputs: ", err)
			}

			// 4) update the EMA fingerprint
			if frame >= 7 && frame <= 13 {

				//    emaFP = α*emaFP + (1-α)*fp2
				for i := range emaFP {
					emaFP[i] = alpha*emaFP[i] + (1-alpha)*fingerPrint2[i]
				}
				// 5) re‐normalize emaFP back to unit length
				var sum float32
				for _, v := range emaFP {
					sum += v * v
				}
				norm := float32(math.Sqrt(float64(sum)))
				if norm > 0 {
					for i := range emaFP {
						emaFP[i] /= norm
					}
				}
			}

		}

		// close runtime and release resources
		err = rt.Close()

		if err != nil {
			log.Fatal("Error closing RKNN runtime: ", err)
		}

		log.Println("done")
	*/
}

// Box holds object bounding box coordinates (x1, y1, x2, y2)
type Box struct {
	X1, Y1, X2, Y2 int
}

// Objects is a struct to represent the compare and dataset objects parsed
// from the objects data file
type Objects struct {
	Compare []Box
	Dataset []Box
}

// ParseObjects reads the TOML-like objects data file returns the two lists
// of objects and their bounding box coordinates
func ParseObjects(path string) (*Objects, error) {

	f, err := os.Open(path)

	if err != nil {
		return nil, err
	}

	defer f.Close()

	objs := &Objects{}
	section := "" // either "compare" or "dataset"
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// skip blank or comment
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// section header
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.ToLower(line[1 : len(line)-1])
			continue
		}

		// data line, expect four ints separated by commas
		fields := strings.Split(line, ",")

		if len(fields) != 4 {
			return nil, fmt.Errorf("invalid data line %q", line)
		}

		nums := make([]int, 4)

		for i, fstr := range fields {
			v, err := strconv.Atoi(strings.TrimSpace(fstr))

			if err != nil {
				return nil, fmt.Errorf("parsing %q: %w", fstr, err)
			}

			nums[i] = v
		}

		// define box
		box := Box{nums[0], nums[1], nums[2], nums[3]}

		switch section {

		case "compare":
			objs.Compare = append(objs.Compare, box)

		case "dataset":
			objs.Dataset = append(objs.Dataset, box)

		default:
			return nil, fmt.Errorf("line %q outside of a known section", line)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return objs, nil
}

// AddObjectToBatch adds the cropped object from source image to the batch for
// running inference on
func AddObjectToBatch(batch *rknnlite.Batch, srcImg gocv.Mat, obj Box,
	scaleSize image.Point) error {

	// get the objects region of interest from source Mat
	objRect := image.Rect(obj.X1, obj.Y1, obj.X2, obj.Y2)
	objRoi := srcImg.Region(objRect)

	objImg := objRoi.Clone()
	gocv.Resize(objRoi, &objImg, scaleSize, 0, 0, gocv.InterpolationArea)

	defer objRoi.Close()
	defer objImg.Close()

	return batch.Add(objImg)
}

// CompareObjects compares the outputs of two objects
func CompareObjects(objA []int8, objB []int8, scales float32,
	ZPs int32) float32 {

	// get the fingerprint of both objects
	fpA := reid.DequantizeAndL2Normalize(objA, scales, ZPs)
	fpB := reid.DequantizeAndL2Normalize(objB, scales, ZPs)

	// compute Euclidean (L2) distance directly
	return reid.EuclideanDistance(fpA, fpB)
}
