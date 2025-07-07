package tracker

import (
	"fmt"
	"github.com/swdee/go-rknnlite"
	"github.com/swdee/go-rknnlite/postprocess/reid"
	"gocv.io/x/gocv"
	"image"
	"sync"
)

// DistanceMethod defines ReID distance calculation methods
type DistanceMethod int

const (
	Euclidean DistanceMethod = 1
	Cosine    DistanceMethod = 2
)

// reID struct holds all ReIdentification processing features
type reID struct {
	// pool is the rknnlike runtime pool to run inference on
	pool *rknnlite.Pool
	// dist is the distance method to apply to calculations to determine similarity
	dist DistanceMethod
	// threshold is the distance cutoff to determine similar or different objects
	threshold float32
	// batchSize store model input tensor batch size
	batchSize int
	width     int
	height    int
	channels  int
	// batchPools holds a pool of batches
	batchPool *rknnlite.BatchPool
	// scaleSize is the size of the input tensor dimensions to scale the object too
	scaleSize image.Point
}

// UseReID sets up Re-Identification processing on the BYTETracker instance
func (bt *BYTETracker) UseReID(pool *rknnlite.Pool, dist DistanceMethod,
	threshold float32) {

	// query runtime and get tensor dimensions
	rt := pool.Get()

	batchSize := int(rt.InputAttrs()[0].Dims[0])
	width := int(rt.InputAttrs()[0].Dims[1])
	height := int(rt.InputAttrs()[0].Dims[2])
	channels := int(rt.InputAttrs()[0].Dims[3])

	bt.reid = &reID{
		pool:      pool,
		dist:      dist,
		threshold: threshold,
		batchSize: batchSize,
		width:     width,
		height:    height,
		channels:  channels,
		scaleSize: image.Pt(width, height),
		batchPool: rknnlite.NewBatchPool(pool.Size(), rt),
	}

	pool.Return(rt)

	bt.useReid = true
}

// UpdateWithFrame updates the tracker with new detections and passes the
// image frame so ReID inference can be conducted
func (bt *BYTETracker) UpdateWithFrame(objects []Object, frame gocv.Mat) ([]*STrack, error) {

	// check if ReID is enabled and get embedding features for all objects
	if bt.useReid {

		bufFrame := frame.Clone()
		defer bufFrame.Close()

		features, err := bt.reid.processObjects(objects, bufFrame)

		if err != nil {
			return nil, fmt.Errorf("failed to process objects: %w", err)
		}

		for i := range objects {
			objects[i].Feature = features[i]
		}
	}

	// run track update
	tracks, err := bt.Update(objects)

	if err != nil {
		return nil, fmt.Errorf("error updating objects: %w", err)
	}

	return tracks, nil
}

// Close frees memory from reid instance
func (r *reID) Close() {
	r.batchPool.Close()
}

// processObjects takes the detected objects and runs inference on them to get
// their embedded feature fingerprint.  Function should be called from a
// Goroutine.
func (r *reID) processObjects(objects []Object, frame gocv.Mat) ([][]float32, error) {

	var wg sync.WaitGroup
	total := len(objects)

	// collect per objects feature embeddings
	allEmbeddings := make([][]float32, total)
	errCh := make(chan error, (total+r.batchSize-1)/r.batchSize)

	for offset := 0; offset < total; offset += r.batchSize {

		end := offset + r.batchSize

		if end > total {
			end = total
		}

		batchObjs := objects[offset:end]

		// capture range variables for closure
		capOffset := offset
		capCnt := end - offset

		wg.Add(1)
		batch := r.batchPool.Get()
		rt := r.pool.Get()

		go func(rt *rknnlite.Runtime, batch *rknnlite.Batch, bobjs []Object, off, cnt int) {
			defer wg.Done()
			fps, err := r.processBatch(rt, batch, bobjs, frame)
			r.pool.Return(rt)
			r.batchPool.Return(batch)

			if err != nil {
				errCh <- err
				return
			}

			// copy this batch’s fingerprints into correct offset place for
			// all fingerprint results
			for i := 0; i < cnt; i++ {
				allEmbeddings[off+i] = fps[i]
			}

			errCh <- nil
		}(rt, batch, batchObjs, capOffset, capCnt)
	}

	wg.Wait()
	close(errCh)

	// if any error, just bail
	for e := range errCh {
		if e != nil {
			return nil, fmt.Errorf("ReID error: %w", e)
		}
	}

	return allEmbeddings, nil
}

// processBatch adds the objects to a batch and runs inference on them
func (r *reID) processBatch(rt *rknnlite.Runtime, batch *rknnlite.Batch,
	bobjs []Object, frame gocv.Mat) ([][]float32, error) {

	height := frame.Rows()
	width := frame.Cols()

	for _, obj := range bobjs {

		// clamp and get bounding box coordinates
		x1 := clamp(int(obj.Rect.TLX()), 0, width)
		y1 := clamp(int(obj.Rect.TLY()), 0, height)
		x2 := clamp(int(obj.Rect.BRX()), 0, width)
		y2 := clamp(int(obj.Rect.BRY()), 0, height)

		objRect := image.Rect(x1, y1, x2, y2)

		// get the objects region of interest from source Mat
		objRoi := frame.Region(objRect)
		objImg := gocv.NewMat()

		// resize to input tensor size
		gocv.Resize(objRoi, &objImg, r.scaleSize, 0, 0, gocv.InterpolationArea)

		objRoi.Close()

		err := batch.Add(objImg)
		objImg.Close()

		if err != nil {
			return nil, fmt.Errorf("error adding image to batch")
		}
	}

	// run inference on the batch
	outputs, err := rt.Inference([]gocv.Mat{batch.Mat()})

	if err != nil {
		return nil, fmt.Errorf("inference failed: %v", err)
	}

	defer outputs.Free()

	// unpack per object results
	fingerprints := make([][]float32, len(bobjs))

	for idx := 0; idx < len(bobjs); idx++ {

		output, err := batch.GetOutputInt(idx, outputs.Output[0], int(outputs.OutputAttributes().DimForDFL))

		if err != nil {
			return nil, fmt.Errorf("error getting output %d: %v", idx, err)
		}

		// get object fingerprint
		fingerprints[idx] = reid.DequantizeAndL2Normalize(
			output,
			outputs.OutputAttributes().Scales[0],
			outputs.OutputAttributes().ZPs[0],
		)
	}

	return fingerprints, nil
}

// clamp restricts the value x to be within the range min and max
func clamp(val, min, max int) int {

	if val > min {

		if val < max {
			return val // casting the float to int after the comparison
		}

		return max
	}

	return min
}
