package postprocess

import (
	"runtime"
	"sync"
)

// matmulUint8 is the straight‑line version: no channels or goroutines.
func matmulUint8(data *strideData, boxesNum, protoChannel,
	protoHeight, protoWeight int, C []uint8) {

	A := data.filterSegmentsByNMS
	B := data.proto

	rows := boxesNum
	colsA := protoChannel
	colsB := protoHeight * protoWeight

	// one big flat loop per row, per column, per channel
	for i := 0; i < rows; i++ {

		baseA := i * colsA
		baseC := i * colsB

		for j := 0; j < colsB; j++ {
			var sum float32

			// accumulate A[i,k] * B[k,j]
			for k := 0; k < colsA; k++ {
				sum += A[baseA+k] * B[k*colsB+j]
			}

			if sum > 0 {
				C[baseC+j] = 4 // object
			}

			// else leave zero, which is the background
		}
	}
}

// matmulUint8Parallel splits the rows across NumCPU workers,
// avoids a channel‐per‐row, and writes disjoint regions of C.
func matmulUint8Parallel(data *strideData, boxesNum, protoChannel,
	protoHeight, protoWeight int, C []uint8) {

	A := data.filterSegmentsByNMS
	B := data.proto

	rows := boxesNum
	colsA := protoChannel
	colsB := protoHeight * protoWeight

	numWorkers := runtime.NumCPU()
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	// each worker handles rows i = w, w+numWorkers, w+2*numWorkers
	for w := 0; w < numWorkers; w++ {
		go func(w int) {
			defer wg.Done()

			for i := w; i < rows; i += numWorkers {
				baseA := i * colsA
				baseC := i * colsB

				for j := 0; j < colsB; j++ {
					var sum float32

					for k := 0; k < colsA; k++ {
						sum += A[baseA+k] * B[k*colsB+j]
					}

					if sum > 0 {
						C[baseC+j] = 4
					}
				}
			}
		}(w)
	}

	wg.Wait()
}
