package rknnlite

import (
	"errors"
	"flag"
	"fmt"
	"gocv.io/x/gocv"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

var modelFiles = flag.String("m", "osnet_x1_0_market_256x128-rk3588-batch{1,4,8,16}.rknn",
	"RKNN compiled model files in format <name>-batch{N1,N2,...,Nk}.rknn")
var rkPlatform = flag.String("p", "rk3588",
	"Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3588]")

// ExpandModelPattern takes a pattern like
//
//	"/some/dir/osnet_x1_0_market_256x128-rk3588-batch{1,4,8,16}.rknn"
//
// and returns:
//
//	[]string{
//	  "/some/dir/osnet_x1_0_market_256x128-rk3588-batch1.rknn",
//	  "/some/dir/osnet_x1_0_market_256x128-rk3588-batch4.rknn",
//	  "/some/dir/osnet_x1_0_market_256x128-rk3588-batch8.rknn",
//	  "/some/dir/osnet_x1_0_market_256x128-rk3588-batch16.rknn",
//	}
func expandModelPattern(pattern string) ([]modelBatches, error) {

	// split off the directory and file
	dir, file := filepath.Split(pattern)

	// match exactly "<prefix>-batch{n1,n2,...}.rknn"
	re := regexp.MustCompile(`^(.+)-batch\{([\d,]+)\}\.rknn$`)
	m := re.FindStringSubmatch(file)

	if m == nil {
		return nil, errors.New("invalid pattern: must be name-batch{n1,n2,...}.rknn")
	}

	prefix := m[1]  // e.g. "osnet_x1_0_market_256x128-rk3588"
	numsCSV := m[2] // e.g. "1,4,8,16"
	nums := strings.Split(numsCSV, ",")
	out := make([]modelBatches, 0, len(nums))

	for _, strNum := range nums {

		num, err := strconv.Atoi(strNum)

		if err != nil {
			return nil, fmt.Errorf("invalid batch size %q: %w", strNum, err)
		}

		name := fmt.Sprintf("%s-batch%d.rknn", prefix, num)

		out = append(out, modelBatches{
			batchSize: num,
			modelFile: filepath.Join(dir, name),
		})
	}

	return out, nil
}

type modelBatches struct {
	batchSize int
	modelFile string
}

// BenchmarkBatchSize runs benchmarks against multiple models to work out per
// image inference time.
func BenchmarkBatchSize(b *testing.B) {

	flag.Parse()

	// from the modelFiles argument create a table of model files and corresponding
	// batch sizes
	cases, err := expandModelPattern(*modelFiles)

	if err != nil {
		b.Fatalf("Invalid modelFile syntax: %v", err)
	}

	const (
		height   = 256
		width    = 128
		channels = 3
	)

	for _, tc := range cases {
		tc := tc // capture

		b.Run(fmt.Sprintf("Batch%02d", tc.batchSize), func(b *testing.B) {

			// load the RKNN model for this batch size
			err := SetCPUAffinityByPlatform(*rkPlatform, FastCores)

			if err != nil {
				b.Fatalf("Failed to set CPU Affinity: %v", err)
			}

			// check if user specified model file or if default is being used.  if default
			// then pick the default platform model to use.
			modelFile := tc.modelFile

			if *rkPlatform != "rk3588" {
				modelFile = strings.ReplaceAll(modelFile, "rk3588", *rkPlatform)
			}

			// create rknn runtime instance
			rt, err := NewRuntimeByPlatform(*rkPlatform, modelFile)

			if err != nil {
				b.Fatalf("Error initializing RKNN runtime: %v", err)
			}

			defer rt.Close()

			// set runtime to leave output tensors as int8
			rt.SetWantFloat(false)

			// prepare zero images
			imgs := make([]gocv.Mat, tc.batchSize)

			for i := range imgs {
				m := gocv.Zeros(height, width, gocv.MatTypeCV8UC3)
				defer m.Close()
				imgs[i] = m
			}

			// pre-allocate the batch container
			batch := rt.NewBatch(tc.batchSize, height, width, channels)
			defer batch.Close()

			b.ResetTimer()
			var totalInf time.Duration

			for i := 0; i < b.N; i++ {
				batch.Clear()
				start := time.Now()

				for _, img := range imgs {
					if err := batch.Add(img); err != nil {
						b.Fatalf("Add() error: %v", err)
					}
				}

				if _, err := rt.Inference([]gocv.Mat{batch.Mat()}); err != nil {
					b.Fatalf("Inference() error: %v", err)
				}

				totalInf += time.Since(start)
			}

			b.StopTimer()

			// milliseconds per batch
			msBatch := float64(totalInf.Nanoseconds()) / 1e6 / float64(b.N)
			b.ReportMetric(msBatch, "ms/batch")

			// milliseconds per image
			msImg := msBatch / float64(tc.batchSize)
			b.ReportMetric(msImg, "ms/img")

		})
	}
}

func TestBatchAddAndOverflow(t *testing.T) {

	r := &Runtime{inputTypeFloat32: false}

	batch := r.NewBatch(2, 2, 3, 1)
	defer batch.Close()

	// create Mats with known data
	m1 := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV8U)
	defer m1.Close()

	buf1, _ := m1.DataPtrUint8()

	for i := range buf1 {
		buf1[i] = uint8(i + 1) // 1,2,3...6
	}

	m2 := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV8U)
	defer m2.Close()

	buf2, _ := m2.DataPtrUint8()

	for i := range buf2 {
		buf2[i] = uint8((i + 1) * 10) // 10,20,...60
	}

	// Add two images
	if err := batch.Add(m1); err != nil {
		t.Fatalf("Add(m1) failed: %v", err)
	}

	if err := batch.Add(m2); err != nil {
		t.Fatalf("Add(m2) failed: %v", err)
	}

	// Underlying batch mat should contain both
	bMat := batch.Mat()
	allData, err := bMat.DataPtrUint8()

	if err != nil {
		t.Fatalf("DataPtrUint8 on batch failed: %v", err)
	}

	// first 6 from buf1, next 6 from buf2
	for i := 0; i < 6; i++ {
		if allData[i] != buf1[i] {
			t.Errorf("element %d = %d; want %d from img1", i, allData[i], buf1[i])
		}
	}

	for i := 0; i < 6; i++ {
		if allData[6+i] != buf2[i] {
			t.Errorf("element %d = %d; want %d from img2", 6+i, allData[6+i], buf2[i])
		}
	}

	// third Add should overflow
	m3 := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV8U)
	err3 := batch.Add(m3)

	if err3 == nil {
		t.Fatal("expected overflow error on third Add, got nil")
	}
}

func TestBatchAddAtAndClear(t *testing.T) {

	r := &Runtime{inputTypeFloat32: false}

	batch := r.NewBatch(3, 2, 2, 1)
	defer batch.Close()

	m := gocv.NewMatWithSize(2, 2, gocv.MatTypeCV8U)
	defer m.Close()

	dat, _ := m.DataPtrUint8()

	for i := range dat {
		dat[i] = uint8(i + 5)
	}

	// AddAt index 1
	if err := batch.AddAt(1, m); err != nil {
		t.Fatalf("AddAt failed: %v", err)
	}

	// matCnt should still be zero
	if batch.matCnt != 0 {
		t.Errorf("matCnt = %d; want 0 after AddAt", batch.matCnt)
	}

	// Clear resets matCnt
	batch.Clear()

	if batch.matCnt != 0 {
		t.Errorf("matCnt = %d; want 0 after Clear", batch.matCnt)
	}

	// Add at invalid index
	err := batch.AddAt(5, m)

	if err == nil {
		t.Error("expected error for AddAt out of range, got nil")
	}
}

func TestGetOutputIntAndF32(t *testing.T) {

	r := &Runtime{inputTypeFloat32: false}

	batch := r.NewBatch(2, 2, 2, 1)
	defer batch.Close()

	// Test GetOutputInt bounds
	dOut := Output{BufInt: []int8{1, 2, 3, 4}, Size: 4}

	if _, err := batch.GetOutputInt(-1, dOut, 2); err == nil {
		t.Error("expected error for GetOutputInt idx<0")
	}

	if _, err := batch.GetOutputInt(2, dOut, 2); err == nil {
		t.Error("expected error for GetOutputInt idx>=size")
	}

	// valid slice
	slice, err := batch.GetOutputInt(1, dOut, 2)

	if err != nil {
		t.Errorf("GetOutputInt failed: %v", err)
	}

	if len(slice) != 2 {
		t.Errorf("len(slice) = %d; want 2", len(slice))
	}

	// Test GetOutputF32 bounds
	dOutF := Output{BufFloat: []float32{1, 2, 3, 4}, Size: 4}

	if _, err := batch.GetOutputF32(-1, dOutF, 2); err == nil {
		t.Error("expected error for GetOutputF32 idx<0")
	}

	if _, err := batch.GetOutputF32(2, dOutF, 2); err == nil {
		t.Error("expected error for GetOutputF32 idx>=size")
	}

	sliceF, err := batch.GetOutputF32(0, dOutF, 2)

	if err != nil {
		t.Errorf("GetOutputF32 failed: %v", err)
	}

	if len(sliceF) != 2 {
		t.Errorf("len(sliceF) = %d; want 2", len(sliceF))
	}
}
