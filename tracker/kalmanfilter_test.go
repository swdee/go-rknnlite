package tracker

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

// floatsEqual compares slices of float32
func floatsEqual(a, b []float32, epsilon float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if diff := a[i] - b[i]; diff > epsilon || diff < -epsilon {
			return false
		}
	}
	return true
}

// matricesEqual compare matrices
func matricesEqual(a, b mat.Matrix, epsilon float64) bool {
	r1, c1 := a.Dims()
	r2, c2 := b.Dims()

	if r1 != r2 || c1 != c2 {
		return false
	}

	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			if diff := a.At(i, j) - b.At(i, j); diff > epsilon || diff < -epsilon {
				return false
			}
		}
	}

	return true
}

// TestKalmanFilter tests for expect output from Kalman Filter.  Input and output
// values are derived from C++ code to compare against
func TestKalmanFilter(t *testing.T) {
	kf := NewKalmanFilter(1.0/20, 1.0/160)

	// Initial state mean and covariance
	mean := make(StateMean, 8)
	covariance := &StateCov{mat.NewDense(8, 8, nil)}

	measurement := DetectBox{100.0, 200.0, 1.0, 50.0}

	// Initialize the filter
	kf.Initiate(mean, covariance, measurement)

	expectedMeanInit := StateMean{100.0, 200.0, 1.0, 50.0, 0.0, 0.0, 0.0, 0.0}

	expectedCovarianceInit := mat.NewDense(8, 8, []float64{
		25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 9.999999747378752e-05, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 9.765625, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 9.765625, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.999999439624929e-11, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.765625,
	})

	if !floatsEqual(mean, expectedMeanInit, 1e-4) {
		t.Errorf("expected mean %v, got %v", expectedMeanInit, mean)
	}

	if !matricesEqual(covariance, expectedCovarianceInit, 1e-4) {
		t.Errorf("expected covariance %v, got %v",
			mat.Formatted(expectedCovarianceInit, mat.Prefix(""), mat.Excerpt(0)),
			mat.Formatted(covariance, mat.Prefix(""), mat.Excerpt(0)),
		)
	}

	// Predict the next state
	kf.Predict(mean, covariance)

	expectedMeanPredict := StateMean{100.0, 200.0, 1.0, 50.0, 0.0, 0.0, 0.0, 0.0}
	expectedCovariancePredict := mat.NewDense(8, 8, []float64{
		41.015625, 0.0, 0.0, 0.0, 9.765625, 0.0, 0.0, 0.0,
		0.0, 41.015625, 0.0, 0.0, 0.0, 9.765625, 0.0, 0.0,
		0.0, 0.0, 0.00020000009494756943, 0.0, 0.0, 0.0, 9.999999439624929e-11, 0.0,
		0.0, 0.0, 0.0, 41.015625, 0.0, 0.0, 0.0, 9.765625,
		9.765625, 0.0, 0.0, 0.0, 9.86328125, 0.0, 0.0, 0.0,
		0.0, 9.765625, 0.0, 0.0, 0.0, 9.86328125, 0.0, 0.0,
		0.0, 0.0, 9.999999439624929e-11, 0.0, 0.0, 0.0, 1.9999998879249858e-10, 0.0,
		0.0, 0.0, 0.0, 9.765625, 0.0, 0.0, 0.0, 9.86328125,
	})

	if !floatsEqual(mean, expectedMeanPredict, 1e-4) {
		t.Errorf("expected mean %v, got %v", expectedMeanPredict, mean)
	}

	if !matricesEqual(covariance, expectedCovariancePredict, 1e-4) {
		t.Errorf("expected covariance %v, got %v",
			mat.Formatted(expectedCovariancePredict, mat.Prefix(""), mat.Excerpt(0)),
			mat.Formatted(covariance, mat.Prefix(""), mat.Excerpt(0)),
		)
	}

	// New measurement
	measurement = DetectBox{105.0, 205.0, 1.1, 55.0}

	// Update the filter with the new measurement
	err := kf.Update(mean, covariance, measurement)

	if err != nil {
		t.Errorf("failed to update: %v", err)
	}

	expectedMeanUpdate := StateMean{104.338844, 204.338837, 1.001961, 54.338844, 1.033058, 1.033058, 0.0, 1.033058}
	expectedCovarianceUpdate := mat.NewDense(8, 8, []float64{
		5.423553719008268, 0.0, 0.0, 0.0, 1.2913223140495873, 0.0, 0.0, 0.0,
		0.0, 5.423553719008268, 0.0, 0.0, 0.0, 1.2913223140495873, 0.0, 0.0,
		0.0, 0.0, 0.00019607852290531608, 0.0, 0.0, 0.0, 9.803920941585902e-11, 0.0,
		0.0, 0.0, 0.0, 5.423553719008268, 0.0, 0.0, 0.0, 1.2913223140495873,
		1.291322314049589, 0.0, 0.0, 0.0, 7.845590134297521, 0.0, 0.0, 0.0,
		0.0, 1.291322314049589, 0.0, 0.0, 0.0, 7.845590134297521, 0.0, 0.0,
		0.0, 0.0, 9.803920941585902e-11, 0.0, 0.0, 0.0, 1.9999998781210662e-10, 0.0,
		0.0, 0.0, 0.0, 1.291322314049589, 0.0, 0.0, 0.0, 7.845590134297521,
	})

	if !floatsEqual(mean, expectedMeanUpdate, 1e-4) {
		t.Errorf("expected mean %v, got %v", expectedMeanUpdate, mean)
	}

	if !matricesEqual(covariance, expectedCovarianceUpdate, 1e-4) {
		t.Errorf("expected covariance %v, got %v",
			mat.Formatted(expectedCovarianceUpdate, mat.Prefix(""), mat.Excerpt(0)),
			mat.Formatted(covariance, mat.Prefix(""), mat.Excerpt(0)),
		)
	}
}
