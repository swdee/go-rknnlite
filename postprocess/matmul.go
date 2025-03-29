package postprocess

import (
	"runtime"
	"sync"
)

// matmulUint8 performs matrix multiplication using the CPU
func matmulUint8(data *strideData, boxesNum int, protoChannel int,
	protoHeight int, protoWeight int) []uint8 {

	A := data.filterSegmentsByNMS
	B := data.proto
	// C is matmulOut
	C := make([]uint8, boxesNum*protoHeight*protoWeight)

	rowsA := boxesNum
	colsA := protoChannel
	colsB := protoHeight * protoWeight

	var temp float32

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			temp = 0
			for k := 0; k < colsA; k++ {
				temp += A[i*colsA+k] * B[k*colsB+j]
			}
			if temp > 0 {
				C[i*colsB+j] = 4 // an object
			} else {
				C[i*colsB+j] = 0 // background
			}
		}
	}

	return C
}

// matmulUint8Parallel performs matrix multiplication using the CPU with
// optimization to process in parallel across goroutines
func matmulUint8Parallel(data *strideData, boxesNum int, protoChannel int,
	protoHeight int, protoWeight int) []uint8 {

	A := data.filterSegmentsByNMS
	B := data.proto
	C := make([]uint8, boxesNum*protoHeight*protoWeight)

	rowsA := boxesNum
	colsA := protoChannel
	colsB := protoHeight * protoWeight

	// use a worker pool based on available CPU cores
	numWorkers := runtime.NumCPU()
	rowCh := make(chan int, rowsA)

	// worker function for performing the matrix multiplication on a row
	worker := func() {
		for i := range rowCh {
			for j := 0; j < colsB; j++ {
				var temp float32
				for k := 0; k < colsA; k++ {
					temp += A[i*colsA+k] * B[k*colsB+j]
				}
				if temp > 0 {
					C[i*colsB+j] = 4 // an object
				} else {
					C[i*colsB+j] = 0 // background
				}
			}
		}
	}

	// start the workers
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		go func() {
			defer wg.Done()
			worker()
		}()
	}

	// distribute rows to workers
	for i := 0; i < rowsA; i++ {
		rowCh <- i
	}
	close(rowCh)

	// wait for all workers to complete
	wg.Wait()

	return C
}
