package reid

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

// DequantizeAndL2Normalize converts a quantized int8 vector "q" into a float32 vector,
// applies dequantization using the provided scale "s" and zero-point "z",
// and then normalizes the result to unit length using L2 normalization.
//
// This is commonly used to convert quantized embedding vectors back to a
// normalized float form for comparison or similarity calculations.
//
// If the resulting vector has zero magnitude, the function returns the
// unnormalized dequantized vector.
func DequantizeAndL2Normalize(q []int8, s float32, z int32) []float32 {

	N := len(q)
	x := make([]float32, N)

	// dequantize
	for i := 0; i < N; i++ {
		x[i] = float32(int32(q[i])-z) * s
	}

	// compute L2 norm
	var sumSquares float32

	for _, v := range x {
		sumSquares += v * v
	}

	norm := float32(math.Sqrt(float64(sumSquares)))

	if norm == 0 {
		// avoid /0
		return x
	}

	// normalize
	for i := 0; i < N; i++ {
		x[i] /= norm
	}

	return x
}

// FingerprintHash takes an L2-normalized []float32 and returns
// a hex-encoded SHA-256 hash of its binary representation.
func FingerprintHash(feat []float32) (string, error) {

	buf := new(bytes.Buffer)

	// write each float32 in little‐endian
	for _, v := range feat {
		if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
			return "", err
		}
	}

	sum := sha256.Sum256(buf.Bytes())

	return hex.EncodeToString(sum[:]), nil
}

// CosineSimilarity returns the cosine of the angle between vectors a and b.
// Assumes len(a)==len(b). If you have already L2‐normalized them,
// this is just their dot-product.
func CosineSimilarity(a, b []float32) float32 {

	var dot float32

	for i := range a {
		dot += a[i] * b[i]
	}

	// If not already normalized, you’d divide by norms here.
	return dot
}

// CosineDistance returns 1 – cosine similarity, which is a proper distance metric
// in [0,2]. For L2-normalized vectors this is in [0,2], and small values mean
// "very similar."
func CosineDistance(a, b []float32) float32 {
	return 1 - CosineSimilarity(a, b)
}

// EuclideanDistance returns the L2 distance between two vectors.
// Lower means "more similar" when your features are L2-normalized.
func EuclideanDistance(a, b []float32) float32 {
	var sum float32

	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}

	return float32(math.Sqrt(float64(sum)))
}

// NormalizeVec normalizes the input float32 slice to unit length and returns
// a new slice. If the input vector has zero magnitude, it returns the original
// slice unchanged.
func NormalizeVec(v []float32) []float32 {

	norm := float32(0.0)

	for _, x := range v {
		norm += x * x
	}

	if norm == 0 {
		return v // avoid division by zero
	}

	norm = float32(math.Sqrt(float64(norm)))

	out := make([]float32, len(v))

	for i, x := range v {
		out[i] = x / norm
	}

	return out
}
