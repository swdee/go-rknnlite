package render

import "image/color"

var (
	// classColors is a list of colors used to paint the object segment mask
	classColors = []color.RGBA{
		// Distinct colors
		{R: 255, G: 56, B: 56, A: 255},   // #FF3838
		{R: 255, G: 112, B: 31, A: 255},  // #FF701F
		{R: 255, G: 178, B: 29, A: 255},  // #FFB21D
		{R: 207, G: 210, B: 49, A: 255},  // #CFD231
		{R: 72, G: 249, B: 10, A: 255},   // #48F90A
		{R: 26, G: 147, B: 52, A: 255},   // #1A9334
		{R: 0, G: 212, B: 187, A: 255},   // #00D4BB
		{R: 0, G: 194, B: 255, A: 255},   // #00C2FF
		{R: 52, G: 69, B: 147, A: 255},   // #344593
		{R: 100, G: 115, B: 255, A: 255}, // #6473FF
		{R: 0, G: 24, B: 236, A: 255},    // #0018EC
		{R: 132, G: 56, B: 255, A: 255},  // #8438FF
		{R: 82, G: 0, B: 133, A: 255},    // #520085
		{R: 255, G: 149, B: 200, A: 255}, // #FF95C8
		{R: 255, G: 55, B: 199, A: 255},  // #FF37C7
		{R: 255, G: 157, B: 151, A: 255}, // #FF9D97
		{R: 44, G: 153, B: 168, A: 255},  // #2C99A8
		{R: 61, G: 219, B: 134, A: 255},  // #3DDB86
		{R: 203, G: 56, B: 255, A: 255},  // #CB38FF
		{R: 146, G: 204, B: 23, A: 255},  // #92CC17

		// Gradient colors
		{R: 250, G: 128, B: 114, A: 255}, // #FA8072
		{R: 64, G: 255, B: 0, A: 255},    // #40FF00
		{R: 255, G: 64, B: 0, A: 255},    // #FF4000
		{R: 64, G: 0, B: 255, A: 255},    // #4000FF
		{R: 0, G: 64, B: 255, A: 255},    // #0040FF
		{R: 0, G: 128, B: 255, A: 255},   // #0080FF
		{R: 0, G: 255, B: 0, A: 255},     // #00FF00
		{R: 128, G: 255, B: 0, A: 255},   // #80FF00
		{R: 255, G: 255, B: 128, A: 255}, // #FFFF80
		{R: 191, G: 255, B: 0, A: 255},   // #BFFF00
		{R: 191, G: 128, B: 255, A: 255}, // #BF80FF
		{R: 255, G: 128, B: 0, A: 255},   // #FF8000
		{R: 210, G: 105, B: 30, A: 255},  // #D2691E
		{R: 128, G: 255, B: 128, A: 255}, // #80FF80
		{R: 255, G: 128, B: 128, A: 255}, // #FF8080
		{R: 96, G: 96, B: 96, A: 255},    // #606060
		{R: 0, G: 0, B: 255, A: 255},     // #0000FF
		{R: 191, G: 0, B: 255, A: 255},   // #BF00FF
		{R: 255, G: 0, B: 0, A: 255},     // #FF0000
		{R: 192, G: 192, B: 192, A: 255}, // #C0C0C0
		{R: 128, G: 191, B: 255, A: 255}, // #80BFFF
		{R: 255, G: 0, B: 128, A: 255},   // #FF0080
		{R: 255, G: 0, B: 255, A: 255},   // #FF00FF
		{R: 255, G: 128, B: 128, A: 255}, // #FF8080
		{R: 0, G: 191, B: 255, A: 255},   // #00BFFF
		{R: 128, G: 128, B: 255, A: 255}, // #8080FF
		{R: 64, G: 0, B: 128, A: 255},    // #400080
		{R: 128, G: 0, B: 64, A: 255},    // #800040
		{R: 255, G: 128, B: 191, A: 255}, // #FF80BF
		{R: 0, G: 255, B: 255, A: 255},   // #00FFFF
		{R: 255, G: 0, B: 191, A: 255},   // #FF00BF
		{R: 128, G: 255, B: 255, A: 255}, // #80FFFF
		{R: 0, G: 255, B: 191, A: 255},   // #00FFBF
		{R: 255, G: 0, B: 64, A: 255},    // #FF0040
		{R: 255, G: 191, B: 128, A: 255}, // #FFBF80
		{R: 255, G: 255, B: 0, A: 255},   // #FFFF00
		{R: 255, G: 128, B: 255, A: 255}, // #FF80FF
		{R: 128, G: 255, B: 191, A: 255}, // #80FFBF
		{R: 128, G: 0, B: 255, A: 255},   // #8000FF
		{R: 255, G: 192, B: 203, A: 255}, // #FFC0CB
		{R: 191, G: 255, B: 128, A: 255}, // #BFFF80
		{R: 0, G: 255, B: 128, A: 255},   // #00FF80
		{R: 255, G: 191, B: 0, A: 255},   // #FFBF00
		{R: 0, G: 255, B: 64, A: 255},    // #00FF40
	}

	Black  = color.RGBA{R: 0, G: 0, B: 0, A: 255}
	White  = color.RGBA{R: 255, G: 255, B: 255, A: 255}
	Yellow = color.RGBA{R: 255, G: 255, B: 50, A: 255}
	Pink   = color.RGBA{R: 255, G: 0, B: 255, A: 255}

	// postPalette are the colors used for the skeleton/pose
	posePalette = []color.RGBA{
		{R: 255, G: 128, B: 0, A: 255},
		{R: 255, G: 153, B: 51, A: 255},
		{R: 255, G: 178, B: 102, A: 255},
		{R: 230, G: 230, B: 0, A: 255},
		{R: 255, G: 153, B: 255, A: 255},
		{R: 153, G: 204, B: 255, A: 255},
		{R: 255, G: 102, B: 255, A: 255},
		{R: 255, G: 51, B: 255, A: 255},
		{R: 102, G: 178, B: 255, A: 255},
		{R: 51, G: 153, B: 255, A: 255},
		{R: 255, G: 153, B: 153, A: 255},
		{R: 255, G: 102, B: 102, A: 255},
		{R: 255, G: 51, B: 51, A: 255},
		{R: 153, G: 255, B: 153, A: 255},
		{R: 102, G: 255, B: 102, A: 255},
		{R: 51, G: 255, B: 51, A: 255},
		{R: 0, G: 255, B: 0, A: 255},
		{R: 0, G: 0, B: 255, A: 255},
		{R: 255, G: 0, B: 0, A: 255},
		{R: 255, G: 255, B: 255, A: 255},
	}

	// keyPointColors correspond to the skeleton/pose key points
	// and colors to use to render for the joints (circles).
	// require 17 colors
	keyPointColors = []color.RGBA{
		posePalette[16], posePalette[16], posePalette[16], posePalette[16], posePalette[16],
		posePalette[9], posePalette[9], posePalette[9], posePalette[9], posePalette[9],
		posePalette[9], posePalette[0], posePalette[0], posePalette[0], posePalette[0],
		posePalette[0], posePalette[0],
	}

	// limbColors correspond to the lines drawn between the key points
	// on the skeleton/pose.  require 19 colors
	limbColors = []color.RGBA{
		posePalette[0], posePalette[0], posePalette[0], posePalette[0], posePalette[7],
		posePalette[7], posePalette[7], posePalette[9], posePalette[9], posePalette[9],
		posePalette[9], posePalette[9], posePalette[16], posePalette[16], posePalette[16],
		posePalette[16], posePalette[16], posePalette[16], posePalette[16],
	}

	// faceLandmarkColors correspond to the face landmark feature keypoints
	// used in RetinaFace models
	faceLandmarkColors = []color.RGBA{
		{R: 51, G: 153, B: 255, A: 255}, // left eye
		{R: 51, G: 153, B: 255, A: 255}, // right eye
		{R: 255, G: 0, B: 0, A: 255},    // nose
		{R: 0, G: 255, B: 0, A: 255},    // left mouth corner
		{R: 0, G: 255, B: 0, A: 255},    // right mouth corner
	}
)

// 0 is arms
// 9 is legs
