package preprocess

import (
	"gocv.io/x/gocv"
	"image"
	"image/color"
)

// Resizer defines the struct used for handling image resizing
type Resizer struct {
	// srcWidth is the width of the source image
	srcWidth int
	// srcHeight is the height of the source image
	srcHeight int
	// destWidth is the width to scale to
	destWidth int
	// destHeight is the height to scale to
	destHeight int
	// tempMat is a Mat used during the resize process
	tempMat gocv.Mat
	// letterbox parameters used in scaling
	xPad  int
	yPad  int
	scale float32
	// resize dimensions
	resizeW int
	resizeH int
}

// NewResizer returns a resizer used for scaling an image to the needed
// dimensions for input tensor size
func NewResizer(srcWidth, srcHeight, destWidth, destHeight int) *Resizer {
	r := &Resizer{
		srcWidth:   srcWidth,
		srcHeight:  srcHeight,
		destWidth:  destWidth,
		destHeight: destHeight,
		tempMat:    gocv.NewMat(),
	}

	// precalculate scaling dimensions
	r.preCalc()

	return r
}

// Close frees memory allocated during resize process
func (r *Resizer) Close() error {
	return r.tempMat.Close()
}

// preCalc the scaling factors for source and destination Mats
func (r *Resizer) preCalc() {

	r.resizeW = r.destWidth
	r.resizeH = r.destHeight

	scaleW := float32(r.destWidth) / float32(r.srcWidth)
	scaleH := float32(r.destHeight) / float32(r.srcHeight)
	r.scale = scaleH

	if scaleW < scaleH {
		r.scale = scaleW
		r.resizeH = int(float32(r.srcHeight) * r.scale)
	} else {
		r.resizeW = int(float32(r.srcWidth) * r.scale)
	}

	r.yPad = (r.destHeight - r.resizeH) / 2 // padding height / 2
	r.xPad = (r.destWidth - r.resizeW) / 2  // padding width / 2
}

// LetterBoxResize resizes the input image to the dimensions needed for the input
// tensor size whilst maintaining image aspect.  Color is that used for letter
// box padding.
// this code is the conversion of C code function convert_image_with_letterbox()
func (r *Resizer) LetterBoxResize(src gocv.Mat, dest *gocv.Mat, color color.RGBA) {

	gocv.Resize(src, &r.tempMat, image.Pt(r.resizeW, r.resizeH),
		0, 0, gocv.InterpolationArea)

	gocv.CopyMakeBorder(r.tempMat, dest, r.yPad, r.destHeight-r.resizeH-r.yPad,
		r.xPad, r.destWidth-r.resizeW-r.xPad, gocv.BorderConstant, color)
}

// ScaleFactor returns the scale factor used in letterbox resize
func (r *Resizer) ScaleFactor() float32 {
	return r.scale
}

// XPad returns the x padding used in letterbox resize
func (r *Resizer) XPad() int {
	return r.xPad
}

// YPad returns the y padding used in letterbox resize
func (r *Resizer) YPad() int {
	return r.yPad
}

// SrcWidth returns the width of the source image
func (r *Resizer) SrcWidth() int {
	return r.srcWidth
}

// SrcHeight returns the height of the source image
func (r *Resizer) SrcHeight() int {
	return r.srcHeight
}
