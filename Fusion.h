#ifndef FUSION_H
#define FUSION_H

#include "KalmanFilter.h"
#include <Eigen/Dense>

class Fusion {
public:
    KalmanFilter kf;

    Fusion() : kf(9, 6) {  // 9 state variables, 6 measurements
        // Initialize FTS and IMU matrices
        kf.F = Eigen::MatrixXd::Identity(9, 9);  // 9x9 state transition matrix
        kf.P = Eigen::MatrixXd::Identity(9, 9);  // 9x9 covariance matrix
        kf.Q = Eigen::MatrixXd::Identity(9, 9);  // 9x9 process noise covariance matrix
        kf.R = Eigen::MatrixXd::Identity(6, 6);  // 6x6 measurement noise covariance matrix

        // Initialize the observation matrix H (6x9)
        kf.H = Eigen::MatrixXd(6, 9);  // 6 rows, 9 columns
        kf.H << 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0;  // 6 rows for measurements
    }

    // Method to estimate contact wrench using IMU and FTS data
    Eigen::VectorXd estimateContactWrench(const Eigen::VectorXd &imu_data, const Eigen::VectorXd &fts_data) {
        // IMU data used in the prediction phase (acceleration input)
        kf.predict(imu_data);

        // FTS data used in the update phase (force and torque input)
        kf.update(fts_data);

        // Return state vector with estimated wrench (forces and torques)
        return kf.x;
    }
};

#endif
