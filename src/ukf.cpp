#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#define PI 3.14159265
#define RADAR_NZ 3
#define LASER_NZ 2
#define SIGNA_POINTS 15
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  // is initialized
  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.9;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  //State Dimension
  n_x_ = 5;
  
  //Augmented Dimension
  n_aug_ = n_x_ + 2;
  
  //Lambda
  lambda_ = 3 - n_aug_;
  
  //Weights
  weights_ = VectorXd::Zero(2 * n_aug_ + 1);
  weights_.segment(1, 2 * n_aug_).fill(0.5 / (n_aug_ + lambda_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  
  //Initiate Matrices
  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1;
  // Sigma Point Matrices
  Xsig_aug_ = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /* 
   INITIALIZATION
   */
   if (!is_initialized_) { 
     double px = 0;
     double py = 0;
     cout << "Begin Initialization" << endl;
     if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
       double rho = meas_package.raw_measurements_[0];
       double phi = meas_package.raw_measurements_[1];
       px = rho * cos(phi);
       py = rho * sin(phi);
       //Check for zeros, if either are zero, initialize with high uncertainty
       if (fabs(px) < 0.001){
         px = 1;
         P_(0,0) = 1000;
       }
       if (fabs(py) < 0.001){
         py = 1;
         P_(1,1) = 1000;
       }
       cout << "First measurement is RADAR" << endl;
     }
     else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
       cout << "First measurement is LASER" << endl;
       px = meas_package.raw_measurements_[0];
       py = meas_package.raw_measurements_[1];
     }
     x_ << px, py, 0., 0., 0.;
     cout << x_ << endl;
     previous_timestamp = meas_package.timestamp_;
     is_initialized_ = true;
     return;
  }
  cout << "ProcessMeasurement" << endl;
  
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ == true) {
    double delta_t = (meas_package.timestamp_ - previous_timestamp) / 1000000.0;
  	previous_timestamp = meas_package.timestamp_;
  	Prediction(delta_t);
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ == true) {
    double delta_t = (meas_package.timestamp_ - previous_timestamp) / 1000000.0;
  	previous_timestamp = meas_package.timestamp_;
  	Prediction(delta_t);
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  cout << "PREDICT" << endl << "Elapsed Time: " << delta_t << endl;
  MatrixXd Q_ = MatrixXd(2, 2);
  Q_ << std_a_ * std_a_, 0,
        0, std_yawdd_* std_yawdd_;
  MatrixXd P_aug_;
  P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_.bottomRightCorner(Q_.rows(), Q_.cols()) = Q_;
  cout << "P_aug_" << endl << P_aug_ << endl;
  VectorXd x_aug_ = VectorXd(n_aug_);
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;
  MatrixXd A = P_aug_.llt().matrixL();
  Xsig_aug_.col(0) = x_aug_;
  for (int i=0; i < n_aug_; i++) {
    Xsig_aug_.col(i + 1) = x_aug_ + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * A.col(i);
  }
  cout << "Xsig_aug_" << endl << Xsig_aug_ << endl;
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v / yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v / yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
    }
    else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  //cout << "Xsig_pred_" << endl << Xsig_pred_ << endl;
  //cout << "weights_" << endl << weights_ << endl;
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_temp = Xsig_pred_.col(i) - x_;
    while (x_temp(3) > PI) {
      x_temp(3) -= 2.0 * PI;
    }
    while (x_temp(3) > PI) {
      x_temp(3) += 2.0 * PI;
    }
    P_ = P_ + weights_(i) * x_temp * x_temp.transpose();
  }  
  //cout << "x_" << endl << x_ << endl;
  //cout << "P_" << endl << P_ << endl;
  
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  cout << "Update LASER" << endl;
  VectorXd z_ = VectorXd(2);
  z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
  MatrixXd H_;
  H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;
  MatrixXd R_ = MatrixXd(2, 2);
  R_ << std_laspx_ * std_laspy_, 0,
        0, std_laspy_ * std_laspy_;
  VectorXd y_ = z_ - H_ * x_;
  MatrixXd Ht_ = H_.transpose();
  MatrixXd S_ = H_ * P_ * Ht_ + R_;
  MatrixXd Si_ = S_.inverse();
  MatrixXd K_ = P_ * Ht_ * Si_;
  x_ = x_ + K_ * y_;
  int size = sizeof P_;
  MatrixXd I_ = MatrixXd::Identity(5,5);
  P_ = ( I_ - K_ * H_) * P_ ;
  cout << "End Update LASER" << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  cout << "Update RADAR" << endl;
  int n_z_ = 3;
  VectorXd z_ = VectorXd(n_z_);
  z_ = meas_package.raw_measurements_;
  MatrixXd Zsig_ = MatrixXd(n_z_, n_aug_ * 2 + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		// extract values for better readibility
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double v1 = cos(yaw) * v;
		double v2 = sin(yaw) * v; 
	
		// measurement model
		Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
		Zsig_(1,i) = atan2(p_y,p_x);                                 //phi
		Zsig_(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
	}
	//mean predicted measurement
  VectorXd z_pred_ = VectorXd(n_z_);
  z_pred_.fill(0.0);
  for (int i = 0; i < 2*n_aug_ + 1; i++) {
      z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S_ = MatrixXd(n_z_,n_z_);
  S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff_ = Zsig_.col(i) - z_pred_;

    //angle normalization
    while (z_diff_(1) > PI) z_diff_(1) -= 2. * PI;
    while (z_diff_(1) < -PI) z_diff_(1) += 2. * PI;

    S_ = S_ + weights_(i) * z_diff_ * z_diff_.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R_ = MatrixXd(n_z_,n_z_);
  R_ <<    std_radr_ * std_radr_, 0, 0,
          0, std_radphi_ * std_radphi_, 0,
          0, 0, std_radrd_*std_radrd_;
  S_ = S_ + R_;
  //cout << "z_pred_" << endl << z_pred_ << endl;
  //cout << "S_" << endl << S_ << endl;
  
  MatrixXd Tc_ = MatrixXd(n_x_, n_z_);
  //calculate cross correlation matrix
  Tc_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff_ = Zsig_.col(i) - z_pred_;
    //angle normalization
    while (z_diff_(1) > PI) z_diff_(1) -= 2. * PI;
    while (z_diff_(1) < -PI) z_diff_(1) += 2. * PI;

    // state difference
    VectorXd x_diff_ = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff_(3) > PI) x_diff_(3) -= 2. * PI;
    while (x_diff_(3) < -PI) x_diff_(3) += 2. * PI;

    Tc_ = Tc_ + weights_(i) * x_diff_ * z_diff_.transpose();
  }

  //Kalman gain K;
  MatrixXd K_ = Tc_ * S_.inverse();

  //residual
  VectorXd z_diff_ = z_ - z_pred_;

  //angle normalization
  while (z_diff_(1) > PI) z_diff_(1) -= 2. * PI;
  while (z_diff_(1) < -PI) z_diff_(1) += 2. * PI;

  //update state mean and covariance matrix
  x_ = x_ + K_ * z_diff_;
  P_ = P_ - K_ * S_ * K_.transpose();
  
  //cout << "x_" << endl << x_ << endl;
  //cout << "P_" << endl << P_ << endl;
  cout << "End Update RADAR" << endl;
}
