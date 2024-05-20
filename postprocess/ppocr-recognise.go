package postprocess

import (
	"fmt"
	"github.com/swdee/go-rknnlite"
	"math"
)

// PPOCRRecognise defines the struct for the PPOCR model inference post processing
type PPOCRRecognise struct {
	Params PPOCRRecogniseParams
}

// PPOCRParams defines the struct containing the PPOCR parameters to use for
// post processing operations
type PPOCRRecogniseParams struct {
	// ModelChars is the list of characters used to train the PPOCR model
	ModelChars []string
	// numChars is the number of characters in ModelChars
	numChar int
	// OutputSeqLen is the length of sequence output data from the OCR model
	OutputSeqLen int
}

// PPOCRRecogniseResult is a text result recognised by OCR
type PPOCRRecogniseResult struct {
	// Text is the recognised text
	Text string
	// Score is the confidence score of the text recognised
	Score float32
}

// NewPPOCRRecognise returns an instance of the PPOCR post processor
func NewPPOCRRecognise(param PPOCRRecogniseParams) *PPOCRRecognise {
	p := &PPOCRRecognise{
		Params: param,
	}

	p.Params.numChar = len(param.ModelChars)

	return p
}

// Recognise takes the RKNN outputs and converts them to text
func (p *PPOCRRecognise) Recognise(outputs *rknnlite.Outputs) []PPOCRRecogniseResult {

	results := make([]PPOCRRecogniseResult, len(outputs.Output))

	for idx, output := range outputs.Output {
		rec, err := p.recogniseText(output)

		if err != nil {
			results[idx] = PPOCRRecogniseResult{
				Text:  "ERROR ModelChars",
				Score: 0,
			}
		} else {
			results[idx] = rec
		}
	}

	return results
}

// recogniseText takes a single RKNN Output and returns the OCR'd text as string
func (p *PPOCRRecognise) recogniseText(output rknnlite.Output) (PPOCRRecogniseResult, error) {

	res := PPOCRRecogniseResult{}

	var argmaxVal float32
	var argmaxIdx, lastIdx, count int

	for n := 0; n < p.Params.OutputSeqLen; n++ {

		offset := n * p.Params.numChar
		argmaxIdx, argmaxVal = p.argMax(output.BufFloat[offset : offset+p.Params.numChar])

		if argmaxIdx > 0 && !(n > 0 && argmaxIdx == lastIdx) {
			// add to score max value
			res.Score += argmaxVal
			count++

			if argmaxIdx > p.Params.numChar {
				return PPOCRRecogniseResult{}, fmt.Errorf("output index is larger than size of ModelChars list")
			}

			res.Text += p.Params.ModelChars[argmaxIdx]
		}

		lastIdx = argmaxIdx
	}

	res.Score /= float32(count) + 1e-6

	if count == 0 || math.IsNaN(float64(res.Score)) {
		res.Score = 0.0
	}

	return res, nil
}

// argMax returns the index of the maximum element in a slice
func (p *PPOCRRecognise) argMax(slice []float32) (int, float32) {

	if len(slice) == 0 {
		return 0, 0
	}

	maxIdx := 0
	maxValue := slice[0]

	for i, value := range slice {
		if value > maxValue {
			maxValue = value
			maxIdx = i
		}
	}

	return maxIdx, maxValue
}
