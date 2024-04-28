package rknnlite

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// LoadLabels reads the labels used to train the Model from the given text file.
// It should contain one label per line.
func LoadLabels(file string) ([]string, error) {

	// open the file
	f, err := os.Open(file)

	if err != nil {
		return nil, fmt.Errorf("error opening file: %w", err)
	}

	defer f.Close()

	// create a scanner to read the file.
	scanner := bufio.NewScanner(f)

	var labels []string

	// read and trim each line
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		labels = append(labels, line)
	}

	// check for errors during scanning
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return labels, nil
}
