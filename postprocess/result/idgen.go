package result

import "sync"

// idGenerator is a struct to hold a counter for generating the next incremental
// ID number
type IDGenerator struct {
	id int64
	sync.Mutex
}

func NewIDGenerator() *IDGenerator {
	return &IDGenerator{}
}

// Getnext next incremental number
func (id *IDGenerator) GetNext() int64 {
	id.Lock()
	defer id.Unlock()
	id.id++
	return id.id
}
