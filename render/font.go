package render

import (
	"gocv.io/x/gocv"
	"image/color"
)

type Alignment int

const (
	Left   Alignment = 1
	Center Alignment = 2
	Right  Alignment = 3
)

// Font defines the parameters for rendering text on an image using GoCV
type Font struct {
	Face      gocv.HersheyFont
	Scale     float64
	Color     color.RGBA
	Thickness int
	LineType  gocv.LineType
	// Padding to place around text
	LeftPad   int
	RightPad  int
	TopPad    int
	BottomPad int
	// Alignment of the text label to the bounding box
	Alignment Alignment
}

// DefaultFont returns default font settings
func DefaultFont() Font {
	return Font{
		Face:      gocv.FontHersheySimplex,
		Scale:     0.5,
		Color:     White,
		Thickness: 1,
		LineType:  gocv.LineAA,
		LeftPad:   4,
		RightPad:  4,
		TopPad:    4,
		BottomPad: 6,
		Alignment: Left,
	}
}
