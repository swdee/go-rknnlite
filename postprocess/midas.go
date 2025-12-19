package postprocess

import (
	"fmt"
	"math"

	"github.com/swdee/go-rknnlite"
	"gocv.io/x/gocv"
)

// MiDaS defines the struct for a MiDaS depth estimation inference post processing
type MiDaS struct {
	// Params are the depth map configuration parameters
	Params MiDaSParams
}

// GrayscaleMap is used to not apply coloring to output depthmap, but to leave as grayscale
const GrayscaleMap = gocv.ColormapTypes(9999)

type MiDaSParams struct {
	// Invert the depth map
	Invert bool
	// Colormap to apply to depth map, if you want it left as grayscale then
	// pass postprocess.GrayscaleMap
	Colormap gocv.ColormapTypes
}

// MiDaSDefaultParams sets output depth map to non-inverting and use Hot color scheme
func MiDaSDefaultParams() MiDaSParams {
	return MiDaSParams{
		Invert:   false,
		Colormap: gocv.ColormapHot,
	}
}

// NewMiDaS returns and instance of the MiDaS post processor
func NewMiDaS(p MiDaSParams) *MiDaS {
	return &MiDaS{
		Params: p,
	}
}

// CreateDepthMap converts the tensor output data into a depth estimation map image
func (m *MiDaS) CreateDepthMap(outputs *rknnlite.Outputs, depthMat gocv.Mat) error {

	// output tensor is in NCHW format
	// get output tensor width/height
	outH := int(outputs.OutputAttributes().DimHeights[0])
	outW := int(outputs.OutputAttributes().DimWidths[0])

	// Convert float depth to uint8 visualization
	depthU8 := m.depthToU8(outputs.Output[0].BufFloat, outH, outW)

	// Make a Mat from bytes
	u8Mat, err := gocv.NewMatFromBytes(outH, outW, gocv.MatTypeCV8U, depthU8)

	if err != nil {
		return fmt.Errorf("Failed to create depth mat: %v", err)
	}

	defer u8Mat.Close()

	if m.Params.Colormap == GrayscaleMap {
		// no coloring
		u8Mat.CopyTo(&depthMat)

	} else {
		// apply colormap
		gocv.ApplyColorMap(u8Mat, &depthMat, m.Params.Colormap)
	}

	return nil
}

// depthToU8 converts a float32 depth map into an 8-bit visualization image.
//
// MiDaS outputs “relative depth” values that are not bounded to [0,1] and
// can vary per image. To visualize, we normalize the depth values to [0,255]
// using the min/max over the whole output map.
//
// Output layout is row-major grayscale: out[y*w + x]
func (m *MiDaS) depthToU8(depth []float32, h, w int) []byte {

	total := h * w
	out := make([]byte, total)

	// First pass: find min/max depth ignoring NaN/Inf values
	minV := float32(math.Inf(1))
	maxV := float32(math.Inf(-1))

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// Read the depth value at (y,x) from the model output buffer
			v := m.getDepthAt(depth, y, x, h, w)

			// Skip invalid floating-point values so they don't poison min/max
			if !m.isFinite32(v) {
				continue
			}

			if v < minV {
				minV = v
			}

			if v > maxV {
				maxV = v
			}
		}
	}

	// Guard against all-invalid outputs or a constant output (max==min)
	den := maxV - minV
	if !m.isFinite32(minV) || !m.isFinite32(maxV) || den <= 0 {
		// Fallback: return all zeros (black image)
		return out
	}

	// Second pass: normalize each pixel to [0,1], optionally invert, clamp, then scale to [0,255]
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := m.getDepthAt(depth, y, x, h, w)

			// If this pixel is invalid, pin it to minV so it becomes black after normalization
			if !m.isFinite32(v) {
				v = minV
			}

			// Normalize to 0..1 based on the image's min/max range
			n := (v - minV) / den

			// Optional inversion for visualization (swap near/far appearance)
			if m.Params.Invert {
				n = 1.0 - n
			}

			// Clamp to [0,1] to avoid overflow/underflow due to outliers or rounding
			if n < 0 {
				n = 0
			}
			if n > 1 {
				n = 1
			}

			// Convert to uint8 grayscale
			out[y*w+x] = byte(n * 255.0)
		}
	}

	return out
}

// getDepthAt returns the depth value at pixel coordinate (y,x) from the raw output buffer.
// This function assumes the output tensor is laid out as NCHW
func (m *MiDaS) getDepthAt(buf []float32, y, x, h, w int) float32 {

	// index = ((n*C + ch)*H + y)*W + x ; n=0, ch=0
	idx := (0*h+y)*w + x
	if idx >= 0 && idx < len(buf) {
		return buf[idx]
	}

	// Out-of-range access should never happen if h/w match the tensor dimensions
	// Returning 0 is a safe fallback to avoid panics
	return 0
}

// isFinite32 returns True if v is neither NaN nor +/-Inf
func (m *MiDaS) isFinite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}
