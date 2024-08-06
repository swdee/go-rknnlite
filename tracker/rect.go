package tracker

import (
	"math"
)

// Tlwh (top, left, width, height) represents a 1x4 matrix
type Tlwh []float32

// Tlbr (top, left, bottom, right) represents a 1x4 matrix
type Tlbr []float32

// Xyah (center x, center y, aspect ratio, height) represents a 1x4 matrix
type Xyah []float32

// Rect represents a rectangle with Tlwh (top, left, width, height) format
type Rect struct {
	Tlwh Tlwh
}

// NewRect creates a new Rect with given coordinates
func NewRect(x, y, width, height float32) Rect {
	return Rect{
		Tlwh: Tlwh{x, y, width, height},
	}
}

// X returns the x coordinate of the rectangle
func (r *Rect) X() float32 {
	return r.Tlwh[0]
}

// Y returns the y coordinate of the rectangle
func (r *Rect) Y() float32 {
	return r.Tlwh[1]
}

// Width returns the width of the rectangle
func (r *Rect) Width() float32 {
	return r.Tlwh[2]
}

// Height returns the height of the rectangle
func (r *Rect) Height() float32 {
	return r.Tlwh[3]
}

// SetX sets the x coordinate of the rectangle
func (r *Rect) SetX(x float32) {
	r.Tlwh[0] = x
}

// SetY sets the y coordinate of the rectangle
func (r *Rect) SetY(y float32) {
	r.Tlwh[1] = y
}

// SetWidth sets the width of the rectangle
func (r *Rect) SetWidth(width float32) {
	r.Tlwh[2] = width
}

// SetHeight sets the height of the rectangle
func (r *Rect) SetHeight(height float32) {
	r.Tlwh[3] = height
}

// TLX returns the top-left x coordinate of the rectangle
func (r *Rect) TLX() float32 {
	return r.Tlwh[0]
}

// TLY returns the top-left y coordinate of the rectangle
func (r *Rect) TLY() float32 {
	return r.Tlwh[1]
}

// BRX returns the bottom-right x coordinate of the rectangle
func (r *Rect) BRX() float32 {
	return r.Tlwh[0] + r.Tlwh[2]
}

// BRY returns the bottom-right y coordinate of the rectangle
func (r *Rect) BRY() float32 {
	return r.Tlwh[1] + r.Tlwh[3]
}

// GetTlbr converts the rectangle to Tlbr (top, left, bottom, right) format
func (r *Rect) GetTlbr() Tlbr {
	return Tlbr{
		r.Tlwh[0],
		r.Tlwh[1],
		r.Tlwh[0] + r.Tlwh[2],
		r.Tlwh[1] + r.Tlwh[3],
	}
}

// GetXyah converts the rectangle to Xyah (center x, center y, aspect ratio,
// height) format
func (r *Rect) GetXyah() Xyah {
	return Xyah{
		r.Tlwh[0] + r.Tlwh[2]/2,
		r.Tlwh[1] + r.Tlwh[3]/2,
		r.Tlwh[2] / r.Tlwh[3],
		r.Tlwh[3],
	}
}

// CalcIoU calculates the Intersection over Union (IoU) with another rectangle
func (r *Rect) CalcIoU(other Rect) float32 {

	boxArea := (other.Tlwh[2] + 1) * (other.Tlwh[3] + 1)
	iw := float32(math.Min(float64(r.Tlwh[0]+r.Tlwh[2]), float64(other.Tlwh[0]+other.Tlwh[2])) - math.Max(float64(r.Tlwh[0]), float64(other.Tlwh[0])) + 1)
	iou := float32(0)

	if iw > 0 {
		ih := float32(math.Min(float64(r.Tlwh[1]+r.Tlwh[3]), float64(other.Tlwh[1]+other.Tlwh[3])) - math.Max(float64(r.Tlwh[1]), float64(other.Tlwh[1])) + 1)

		if ih > 0 {
			ua := (r.Tlwh[2]+1)*(r.Tlwh[3]+1) + boxArea - iw*ih
			iou = iw * ih / ua
		}
	}

	return iou
}

// GenerateRectByTlbr creates a Rect from Tlbr (top, left, bottom, right) format
func GenerateRectByTlbr(tlbr Tlbr) Rect {
	return NewRect(tlbr[0], tlbr[1], tlbr[2]-tlbr[0], tlbr[3]-tlbr[1])
}

// GenerateRectByXyah creates a Rect from Xyah (center x, center y,
// aspect ratio, height) format
func GenerateRectByXyah(xyah Xyah) Rect {
	width := xyah[2] * xyah[3]
	return NewRect(xyah[0]-width/2, xyah[1]-xyah[3]/2, width, xyah[3])
}
