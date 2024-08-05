package tracker

import (
	"errors"
	"fmt"
	"gonum.org/v1/gonum/mat"
)

// DetectBox represents a 1x4 matrix using a slice of float32
type DetectBox []float32

// StateMean represents a 1x8 matrix using a slice of float32
type StateMean []float32

// StateCov represents an 8x8 matrix
type StateCov struct {
	*mat.Dense
}

// StateHMean represents a 1x4 matrix using a slice of float32
type StateHMean []float32

// StateHCov represents a 4x4 matrix
type StateHCov struct {
	*mat.SymDense
}

// KalmanFilter represents the Kalman filter
type KalmanFilter struct {
	stdWeightPosition float32
	stdWeightVelocity float32
	motionMat         *mat.Dense
	updateMat         *mat.Dense
}

// NewKalmanFilter initializes and returns a new KalmanFilter
func NewKalmanFilter(stdWeightPosition, stdWeightVelocity float32) *KalmanFilter {

	ndim := 4
	dt := float32(1.0)

	// create identity matrix for motionMat
	motionMat := mat.NewDense(8, 8, nil)

	for i := 0; i < 8; i++ {
		motionMat.Set(i, i, float64(1.0))
	}

	for i := 0; i < ndim; i++ {
		motionMat.Set(i, ndim+i, float64(dt))
	}

	// create updateMat as a 4x8 matrix with first 4 diagonal elements set to 1
	updateMat := mat.NewDense(4, 8, nil)

	for i := 0; i < 4; i++ {
		updateMat.Set(i, i, float64(1.0))
	}

	return &KalmanFilter{
		stdWeightPosition: stdWeightPosition,
		stdWeightVelocity: stdWeightVelocity,
		motionMat:         motionMat,
		updateMat:         updateMat,
	}
}

// Initiate initializes the state mean and covariance
func (kf *KalmanFilter) Initiate(mean StateMean, covariance *StateCov,
	measurement DetectBox) {

	// copy the first four elements of the measurement into the mean
	copy(mean[:4], measurement[:4])

	// set the last four elements of the mean to 0 (velocity components)
	for i := 4; i < 8; i++ {
		mean[i] = 0.0
	}

	// initialize the standard deviation array for the state variables
	std := make(StateMean, 8)
	std[0] = 2 * kf.stdWeightPosition * measurement[3]  // x position
	std[1] = 2 * kf.stdWeightPosition * measurement[3]  // y position
	std[2] = 1e-2                                       // aspect ratio
	std[3] = 2 * kf.stdWeightPosition * measurement[3]  // height
	std[4] = 10 * kf.stdWeightVelocity * measurement[3] // x velocity
	std[5] = 10 * kf.stdWeightVelocity * measurement[3] // y velocity
	std[6] = 1e-5                                       // aspect ratio velocity
	std[7] = 10 * kf.stdWeightVelocity * measurement[3] // height velocity

	// square the standard deviations to get the variances
	tmp := make(StateMean, 8)

	for i, v := range std {
		tmp[i] = v * v
	}

	// set the diagonal elements of the covariance matrix to the variances
	for i := 0; i < 8; i++ {
		covariance.Set(i, i, float64(tmp[i]))
	}
}

// Predict predicts the next state mean and covariance
func (kf *KalmanFilter) Predict(mean StateMean, covariance *StateCov) {

	// initialize the standard deviation array for the state variables
	std := make(StateMean, 8)
	std[0] = kf.stdWeightPosition * mean[3] // x position
	std[1] = kf.stdWeightPosition * mean[3] // y position
	std[2] = 1e-2                           // aspect ratio
	std[3] = kf.stdWeightPosition * mean[3] // height
	std[4] = kf.stdWeightVelocity * mean[3] // x velocity
	std[5] = kf.stdWeightVelocity * mean[3] // y velocity
	std[6] = 1e-5                           // aspect ratio velocity
	std[7] = kf.stdWeightVelocity * mean[3] // height velocity

	// square the standard deviation values to get the variances
	tmp := make(StateMean, 8)

	for i, v := range std {
		tmp[i] = v * v
	}

	// create the motion covariance matrix with variances on the diagonal
	motionCov := mat.NewDense(8, 8, nil)

	for i := 0; i < 8; i++ {
		motionCov.Set(i, i, float64(tmp[i]))
	}

	// convert the mean state vector to a matrix for multiplication
	meanVec := mat.NewVecDense(8, nil)

	for i := 0; i < 8; i++ {
		meanVec.SetVec(i, float64(mean[i]))
	}

	meanMat := mat.NewDense(8, 1, meanVec.RawVector().Data)

	// predict the next state mean using the motion model
	meanMat.Mul(kf.motionMat, meanMat)

	for i := 0; i < 8; i++ {
		mean[i] = float32(meanMat.At(i, 0))
	}

	// predict the next state covariance using the motion model
	cov := covariance.Dense
	cov.Mul(kf.motionMat, cov)
	cov.Mul(cov, kf.motionMat.T())
	cov.Add(cov, motionCov)
}

// Update updates the state mean and covariance
func (kf *KalmanFilter) Update(mean StateMean, covariance *StateCov,
	measurement DetectBox) error {

	// project the state mean and covariance to measurement space
	projectedMean, projectedCov := kf.project(mean, covariance)

	// perform Cholesky factorization of the projected covariance matrix
	chol := mat.Cholesky{}

	if ok := chol.Factorize(projectedCov); !ok {
		return errors.New("failed to factorize projected covariance")
	}

	// compute the matrix B for Kalman gain calculation
	B := mat.NewDense(8, 4, nil)
	B.Mul(covariance.Dense, kf.updateMat.T())

	// compute the Kalman gain using the Cholesky factorization
	var kalmanGain mat.Dense
	err := chol.SolveTo(&kalmanGain, B.T())

	if err != nil {
		return fmt.Errorf("failed to compute kalman gain: %w", err)
	}

	// compute the innovation (measurement residual)
	innovation := make([]float64, 4)

	for i := 0; i < 4; i++ {
		innovation[i] = float64(measurement[i] - projectedMean[i])
	}

	// update the state mean with the innovation
	innovationVec := mat.NewVecDense(4, innovation)
	tmp := mat.NewVecDense(8, nil)
	tmp.MulVec(kalmanGain.T(), innovationVec)

	for i := 0; i < 8; i++ {
		mean[i] += float32(tmp.AtVec(i))
	}

	// update the state covariance
	temp := mat.NewDense(8, 4, nil)
	temp.Mul(kalmanGain.T(), projectedCov)

	temp2 := mat.NewDense(8, 8, nil)
	temp2.Mul(temp, &kalmanGain)

	newCov := mat.NewDense(8, 8, nil)
	newCov.Sub(covariance.Dense, temp2)

	covariance.Dense = newCov

	return nil
}

// Project projects the state mean and covariance to measurement space
func (kf *KalmanFilter) project(mean StateMean,
	covariance *StateCov) (StateHMean, *StateHCov) {

	// compute standard deviations for the measurement noise
	std := make(DetectBox, 4)
	std[0] = kf.stdWeightPosition * mean[3]
	std[1] = kf.stdWeightPosition * mean[3]
	std[2] = 1e-1
	std[3] = kf.stdWeightPosition * mean[3]

	// create the innovation covariance matrix (measurement noise covariance)
	innovationCov := mat.NewSymDense(4, nil)

	for i := 0; i < 4; i++ {
		innovationCov.SetSym(i, i, float64(std[i]*std[i]))
	}

	// project the state mean to measurement space
	projectedMeanVec := mat.NewVecDense(4, nil)
	projectedMeanVec.MulVec(
		kf.updateMat, mat.NewVecDense(8, func() []float64 {
			data := make([]float64, 8)
			for i, v := range mean {
				data[i] = float64(v)
			}
			return data
		}()),
	)

	// project the state covariance to measurement space
	projectedCov := mat.NewSymDense(4, nil)
	temp := mat.NewDense(4, 8, nil)
	temp.Mul(kf.updateMat, covariance.Dense)
	temp2 := mat.NewDense(4, 4, nil)
	temp2.Mul(temp, kf.updateMat.T())

	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			projectedCov.SetSym(i, j, temp2.At(i, j))
		}
	}

	// add the innovation covariance to the projected covariance
	projectedCov.AddSym(projectedCov, innovationCov)

	// convert the projected mean to StateHMean type
	projectedMean := make(StateHMean, 4)

	for i := 0; i < 4; i++ {
		projectedMean[i] = float32(projectedMeanVec.AtVec(i))
	}

	// return the projected mean and covariance
	return projectedMean, &StateHCov{projectedCov}
}
