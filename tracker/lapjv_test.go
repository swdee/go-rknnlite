package tracker

import (
	"fmt"
	"testing"
)

func runLapjvTest(t *testing.T, costMatrix [][]float64, expectedX, expectedY []int) {

	n := len(costMatrix)
	x := make([]int, n)
	y := make([]int, n)

	ret, err := lapjvInternal(n, costMatrix, x, y)
	if err != nil {
		t.Errorf("lapjvInternal returned an error: %v", err)
	}

	if ret != 0 {
		t.Errorf("lapjvInternal returned a non-zero value: %d", ret)
	}

	for i := 0; i < n; i++ {
		if x[i] != expectedX[i] {
			t.Errorf("Expected x[%d] = %d, but got %d", i, expectedX[i], x[i])
		}
		if y[i] != expectedY[i] {
			t.Errorf("Expected y[%d] = %d, but got %d", i, expectedY[i], y[i])
		}
	}

	fmt.Printf("Solution found:\n")
	for i := 0; i < n; i++ {
		fmt.Printf("Row %d assigned to column %d\n", i, x[i])
	}
}

func TestLapjvInternal(t *testing.T) {
	costMatrix1 := [][]float64{
		{4, 1, 3, 2},
		{2, 0, 5, 3},
		{3, 2, 2, 3},
		{2, 3, 3, 2},
	}

	expectedX1 := []int{3, 1, 2, 0}
	expectedY1 := []int{3, 1, 2, 0}

	costMatrix2 := [][]float64{
		{10, 19, 8, 15},
		{10, 18, 7, 17},
		{13, 16, 9, 14},
		{12, 19, 8, 18},
	}

	expectedX2 := []int{3, 0, 1, 2}
	expectedY2 := []int{1, 2, 3, 0}

	t.Run("Test Case 1", func(t *testing.T) {
		runLapjvTest(t, costMatrix1, expectedX1, expectedY1)
	})

	t.Run("Test Case 2", func(t *testing.T) {
		runLapjvTest(t, costMatrix2, expectedX2, expectedY2)
	})
}
