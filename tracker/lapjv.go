package tracker

import (
	"errors"
)

const (
	LARGE = 1000000.0
)

// lapjvInternal is the main function to solve the dense sparse LAP
// LAPJV (Linear Assignment Problem, Jonker-Volgenant algorithm)
func lapjvInternal(n int, cost [][]float64, x, y []int) (int, error) {

	freeRows := make([]int, n)
	v := make([]float64, n)

	ret := ccrrtDense(n, cost, freeRows, x, y, v)

	i := 0

	for ret > 0 && i < 2 {
		ret = carrDense(n, cost, ret, freeRows, x, y, v)
		i++
	}

	if ret > 0 {
		err := caDense(n, cost, ret, freeRows, x, y, v)

		if err != nil {
			return 0, nil
		}

		ret = 0
	}

	return ret, nil
}

// ccrrtDense performs column-reduction and reduction transfer for a dense cost matrix
func ccrrtDense(n int, cost [][]float64, freeRows, x, y []int, v []float64) int {

	unique := make([]bool, n)

	for i := 0; i < n; i++ {
		x[i] = -1
		v[i] = LARGE
		y[i] = 0
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			c := cost[i][j]
			if c < v[j] {
				v[j] = c
				y[j] = i
			}
		}
	}

	for i := 0; i < n; i++ {
		unique[i] = true
	}

	j := n

	for j > 0 {
		j--
		i := y[j]
		if x[i] < 0 {
			x[i] = j
		} else {
			unique[i] = false
			y[j] = -1
		}
	}

	nFreeRows := 0

	for i := 0; i < n; i++ {

		if x[i] < 0 {
			freeRows[nFreeRows] = i
			nFreeRows++

		} else if unique[i] {

			j := x[i]
			minVal := LARGE

			for j2 := 0; j2 < n; j2++ {
				if j2 == j {
					continue
				}

				c := cost[i][j2] - v[j2]

				if c < minVal {
					minVal = c
				}
			}

			v[j] -= minVal
		}
	}

	return nFreeRows
}

// carrDense performs augmenting row reduction for a dense cost matrix
func carrDense(n int, cost [][]float64, nFreeRows int, freeRows,
	x, y []int, v []float64) int {

	current := 0
	newFreeRows := 0
	rrCnt := 0

	for current < nFreeRows {

		rrCnt++
		freeI := freeRows[current]
		current++

		j1 := 0
		v1 := cost[freeI][0] - v[0]
		j2 := -1
		v2 := LARGE

		for j := 1; j < n; j++ {
			c := cost[freeI][j] - v[j]
			if c < v2 {
				if c >= v1 {
					v2 = c
					j2 = j
				} else {
					v2 = v1
					v1 = c
					j2 = j1
					j1 = j
				}
			}
		}

		i0 := y[j1]
		v1New := v[j1] - (v2 - v1)
		v1Lowers := v1New < v[j1]

		if rrCnt < current*n {
			if v1Lowers {
				v[j1] = v1New
			} else if i0 >= 0 && j2 >= 0 {
				j1 = j2
				i0 = y[j2]
			}

			if i0 >= 0 {
				if v1Lowers {
					current--
					freeRows[current] = i0
				} else {
					freeRows[newFreeRows] = i0
					newFreeRows++
				}
			}
		} else {
			if i0 >= 0 {
				freeRows[newFreeRows] = i0
				newFreeRows++
			}
		}

		x[freeI] = j1
		y[j1] = freeI
	}

	return newFreeRows
}

// findDense finds columns with minimum d[j] and put them on the SCAN list
func findDense(n int, lo int, d []float64, cols, y []int) int {

	hi := lo + 1
	mind := d[cols[lo]]

	for k := hi; k < n; k++ {

		j := cols[k]

		if d[j] <= mind {
			if d[j] < mind {
				hi = lo
				mind = d[j]
			}

			cols[k] = cols[hi]
			cols[hi] = j
			hi++
		}
	}

	return hi
}

// scanDense scans all columns in TODO starting from arbitrary column in SCAN
// and try to decrease d of the TODO columns using the SCAN column
func scanDense(n int, cost [][]float64, lo, hi *int, d []float64,
	cols, pred, y []int, v []float64) int {

	for *lo != *hi {

		j := cols[*lo]
		*lo++
		i := y[j]
		mind := d[j]
		h := cost[i][j] - v[j] - mind

		for k := *hi; k < n; k++ {
			j = cols[k]
			credIJ := cost[i][j] - v[j] - h

			if credIJ < d[j] {
				d[j] = credIJ
				pred[j] = i

				if credIJ == mind {
					if y[j] < 0 {
						return j
					}

					cols[k] = cols[*hi]
					cols[*hi] = j
					(*hi)++
				}
			}
		}
	}

	return -1
}

// findPathDense performs a single iteration of modified Dijkstra shortest path
// algorithm as explained in the JV paper.  This is a dense matrix version.
func findPathDense(n int, cost [][]float64, startI int, y []int, v []float64,
	pred []int) int {

	lo := 0
	hi := 0
	finalJ := -1
	nReady := 0
	cols := make([]int, n)
	d := make([]float64, n)

	for i := 0; i < n; i++ {
		cols[i] = i
		pred[i] = startI
		d[i] = cost[startI][i] - v[i]
	}

	for finalJ == -1 {
		// No columns left on the SCAN list
		if lo == hi {
			nReady = lo
			hi = findDense(n, lo, d, cols, y)

			for k := lo; k < hi; k++ {
				j := cols[k]

				if y[j] < 0 {
					finalJ = j
				}
			}
		}

		if finalJ == -1 {
			finalJ = scanDense(n, cost, &lo, &hi, d, cols, pred, y, v)
		}
	}

	mind := d[cols[lo]]

	for k := 0; k < nReady; k++ {
		j := cols[k]
		v[j] += d[j] - mind
	}

	return finalJ
}

// caDense performs augmenting for a dense cost matrix
func caDense(n int, cost [][]float64, nFreeRows int, freeRows,
	x, y []int, v []float64) error {

	pred := make([]int, n)

	for _, freeI := range freeRows[:nFreeRows] {

		i := -1
		k := 0

		j := findPathDense(n, cost, freeI, y, v, pred)

		if j < 0 {
			return errors.New("Error occurred in caDense(): j < 0")
		}

		if j >= n {
			return errors.New("Error occurred in caDense(): j >= n")
		}

		for i != freeI {

			i = pred[j]
			y[j] = i
			j, x[i] = x[i], j
			k++

			if k >= n {
				return errors.New("Error occurred in caDense(): k >= n")
			}
		}
	}

	return nil
}
