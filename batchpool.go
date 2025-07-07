package rknnlite

import (
	"sync"
)

// BatchPool is a pool of batches
type BatchPool struct {
	// pool of batches
	batches chan *Batch
	// size of pool
	size  int
	close sync.Once
}

// NewBatchPool returns a pool of Batches
func NewBatchPool(size int, rt *Runtime) *BatchPool {

	p := &BatchPool{
		batches: make(chan *Batch, size),
		size:    size,
	}

	batchSize := int(rt.InputAttrs()[0].Dims[0])
	width := int(rt.InputAttrs()[0].Dims[1])
	height := int(rt.InputAttrs()[0].Dims[2])
	channels := int(rt.InputAttrs()[0].Dims[3])
	inputType := rt.GetInputTypeFloat32()

	// create batch pool to be the same size as the runtime pool
	for i := 0; i < size; i++ {
		batch := NewBatch(
			batchSize,
			height,
			width,
			channels,
			inputType,
		)

		// attach to pool
		p.Return(batch)
	}

	return p
}

// Gets a batch from the pool
func (p *BatchPool) Get() *Batch {
	return <-p.batches
}

// Return a batch to the pool
func (p *BatchPool) Return(batch *Batch) {

	batch.Clear()

	select {
	case p.batches <- batch:
	default:
		// pool is full or closed
	}
}

// Close the pool and all batches in it
func (p *BatchPool) Close() {
	p.close.Do(func() {
		// close channel
		close(p.batches)

		// close all runtimes
		for next := range p.batches {
			_ = next.Close()
		}
	})
}
