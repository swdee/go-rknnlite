package postprocess

import (
	"fmt"
	"sync"
)

// bufferPool holds a set of named buffer pools
type bufferPool struct {
	mu    sync.Mutex
	pools map[string]*bufferEntry
}

// bufferEntry defines a single buffer
type bufferEntry struct {
	pool    sync.Pool
	maxSize int
}

// NewBufferPool returns an empty segBufferPool.
func NewBufferPool() *bufferPool {
	return &bufferPool{
		pools: make(map[string]*bufferEntry),
	}
}

// Create registers a new pool under 'name' that will produce buffers
// up to maxSize. Calling it twice with the same name returns an error.
func (b *bufferPool) Create(name string, maxSize int) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.pools[name]; exists {
		return fmt.Errorf("buffer pool %q already exists", name)
	}

	entry := &bufferEntry{maxSize: maxSize}

	entry.pool.New = func() any {
		return make([]uint8, maxSize)
	}

	b.pools[name] = entry
	return nil
}

// Get returns a []uint8 slice of length 'size' from the named pool.
// If size > maxSize, it allocates a new slice of exactly size.
// Panics if the pool name is unknown.
func (b *bufferPool) Get(name string, size int) []uint8 {
	b.mu.Lock()
	entry, ok := b.pools[name]
	b.mu.Unlock()

	if !ok {
		panic(fmt.Sprintf("buffer pool %q not registered", name))
	}

	buf := entry.pool.Get().([]uint8)

	if cap(buf) < size {
		return make([]uint8, size)
	}

	// get buffer of required size
	buf = buf[:size]

	// zero out the buffer
	for i := range buf {
		buf[i] = 0
	}

	return buf
}

// Put returns a buffer back into it's named pool.
// You must only call Put on a buffer you previously got via Get
// with the same name.
func (b *bufferPool) Put(name string, buf []uint8) {
	b.mu.Lock()
	entry, ok := b.pools[name]
	b.mu.Unlock()

	if !ok {
		panic(fmt.Sprintf("buffer pool %q not registered", name))
	}

	// restore to full capacity so it matches entry.New next time
	entry.pool.Put(buf[:entry.maxSize])
}
