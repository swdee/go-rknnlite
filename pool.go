package rknnlite

import (
	"fmt"
	"strings"
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

// NewPool creates a new runtime pool that pins the runtimes to the
// specified NPU cores.  You can use the variables RK3588, RK3582, RK3576,
// RK3568, RK3566, RK3562 for the CoreMask array, or create your own, eg:
// []CoreMask{NPUCore0, NPUCore1}
func NewPool(size int, modelFile string, cores []CoreMask) (*Pool, error) {
	p := &Pool{
		runtimes: make(chan *Runtime, size),
		size:     size,
	}

	for i := 0; i < size; i++ {
		rt, err := NewRuntime(modelFile, getRuntimeCore(i, cores))

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

// NewPoolByPlatform creates a new runtime pool that pins the runtimes to the
// NPU cores for the given rockchip platform string of
// rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588
func NewPoolByPlatform(platform string, size int, modelFile string) (*Pool, error) {

	platform = strings.TrimSpace(platform)
	platform = strings.ToLower(platform)

	// default to NPUSkipSetCore
	var useCores []CoreMask

	switch platform {
	case "rk3562", "rk3566", "rk3568":
		useCores = RK3562

	case "rk3576":
		useCores = RK3576

	case "rk3582", "rk3588":
		useCores = RK3588

	default:
		return nil, fmt.Errorf("unknown platform: %s", platform)
	}

	return NewPool(size, modelFile, useCores)
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

// SetWantFloat defines if the Model load requires Output tensors to be converted
// to float32 for post processing, or left as quantitized int8
func (p *Pool) SetWantFloat(val bool) {
	// set value for each runtime in the pool
	for i := 0; i < p.size; i++ {
		rt := p.Get()
		rt.SetWantFloat(val)
		p.Return(rt)
	}
}

// getRuntimeCore takes an integer and returns the core mask value to use from
// the coremask list
func getRuntimeCore(i int, cores []CoreMask) CoreMask {
	return cores[i%len(cores)]
}
