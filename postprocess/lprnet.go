package postprocess

import (
	"github.com/swdee/go-rknnlite"
)

// LPRNet defines the struct for LPRNet model inference post processing
type LPRNet struct {
	Params LPRNetParams
}

// LPRNetParams defines the struct containing the LPRNet parameters to use for
// post processing operations
type LPRNetParams struct {
	// PlatePositions is the number of license plate positions to traverse
	PlatePositions int
	// PlateChars are the characters on the number plate used to train the model
	PlateChars []string
	// numChars is the number of characters in PlateChars
	numChar int
}

// NewLPRNet return an instance of the LPRNet post processor
func NewLPRNet(p LPRNetParams) *LPRNet {
	l := &LPRNet{
		Params: p,
	}

	l.Params.numChar = len(p.PlateChars)

	return l
}

// ReadPlates takes the RKNN outputs and reads out the license plate numbers
func (l *LPRNet) ReadPlates(outputs *rknnlite.Outputs) []string {

	results := make([]string, len(outputs.Output))

	for idx, output := range outputs.Output {
		results[idx] = l.processPlate(output)
	}

	return results
}

// processPlate takes a single RKNN Output and returns the number plate as string
func (l *LPRNet) processPlate(output rknnlite.Output) string {

	// prebs holds the position of the maximum probabilty of matching the
	// indexed character
	prebs := make([]int, l.Params.PlatePositions)

	// traverse license plate positions
	for x := 0; x < l.Params.PlatePositions; x++ {
		preb := make([]int, l.Params.numChar)

		for y := 0; y < l.Params.numChar; y++ {
			// get next column
			val := output.BufFloat[x+y*l.Params.PlatePositions]
			preb[y] = int(val)
		}

		prebs[x] = l.argMax(preb)
	}

	// remove duplicates and blanks
	noRepeatBlankLabel := []int{}
	preC := prebs[0]

	if prebs[0] != l.Params.numChar-1 {
		noRepeatBlankLabel = append(noRepeatBlankLabel, prebs[0])
	}

	for _, val := range prebs {
		if val == l.Params.numChar-1 || val == preC {
			preC = val
			continue
		}
		noRepeatBlankLabel = append(noRepeatBlankLabel, val)
		preC = val
	}

	// convert number plate to string
	plate := ""

	for _, char := range noRepeatBlankLabel {
		plate += l.Params.PlateChars[char]
	}

	return plate
}

// argMax returns the index of the maximum value in the array.
func (l *LPRNet) argMax(arr []int) int {

	maxIndex := 0

	for i, value := range arr {
		if value > arr[maxIndex] {
			maxIndex = i
		}
	}

	return maxIndex
}
