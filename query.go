package rknnlite

import (
	"fmt"
	"io"
)

// Query the runtime and loaded model to get input and output tensor information
// as well as SDK version in text/human readable format
func (r *Runtime) Query(w io.Writer) error {

	// get SDK version
	ver, err := r.SDKVersion()

	if err != nil {
		return fmt.Errorf("Error initializing RKNN runtime: %w", err)
	}

	fmt.Fprintf(w, "Driver Version: %s, API Version: %s\n", ver.DriverVersion, ver.APIVersion)

	// get model input and output numbers
	num, err := r.QueryModelIONumber()

	if err != nil {
		return fmt.Errorf("Error querying IO Numbers: %w", err)
	}

	fmt.Fprintf(w, "Model Input Number: %d, Output Number: %d\n", num.NumberInput, num.NumberOutput)

	// query Input tensors
	inputAttrs, err := r.QueryInputTensors()

	if err != nil {
		return fmt.Errorf("Error querying Input Tensors: %w", err)
	}

	fmt.Fprintf(w, "Input tensors:\n")

	for _, attr := range inputAttrs {
		fmt.Fprintf(w, "  %s\n", attr.String())
	}

	// query Output tensors
	outputAttrs, err := r.QueryOutputTensors()

	if err != nil {
		return fmt.Errorf("Error querying Output Tensors: %w", err)
	}

	fmt.Fprintf(w, "Output tensors:\n")

	for _, attr := range outputAttrs {
		fmt.Fprintf(w, "  %s\n", attr.String())
	}

	return nil
}
