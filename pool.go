package rknnlite

import (
	"sync"
)

// Pool is a simple runtime pool to open multiple of the same Model across
// all NPU cores
type Pool struct {
	// pool of runtimes
	runtimes chan *Runtime
	// size of pool
	size  int
	close sync.Once
}

// NewPool creates a new runtime pool
func NewPool(size int, modelFile string) (*Pool, error) {
	p := &Pool{
		runtimes: make(chan *Runtime, size),
		size:     size,
	}

	for i := 0; i < size; i++ {
		rt, err := NewRuntime(modelFile, getRuntimeCore(i))

		if err != nil {
			// close any instances that may have been created before receiving
			// the error
			p.Close()
			return nil, err
		}

		// attach to pool
		p.Return(rt)
	}

	return p, nil
}

// Gets a runtime from the pool
func (p *Pool) Get() *Runtime {
	return <-p.runtimes
}

// Return a runtime to the pool
func (p *Pool) Return(runtime *Runtime) {
	select {
	case p.runtimes <- runtime:
	default:
		// pool is full or closed
	}
}

// Close the pool and all runtimes in it
func (p *Pool) Close() {
	p.close.Do(func() {
		// close channel
		close(p.runtimes)

		// close all runtimes
		for next := range p.runtimes {
			_ = next.Close()
		}
	})
}

// getRuntimeCore takes an integer and returns the core mask value to use
func getRuntimeCore(i int) CoreMask {

	switch i % 3 {
	case 0:
		return NPUCore0
	case 1:
		return NPUCore1
	case 2:
		return NPUCore2
	}

	// impossible to reach here
	return NPUCoreAuto
}
