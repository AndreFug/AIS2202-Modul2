#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <Eigen/Dense>

class KalmanFilter {
public:
    Eigen::VectorXd x;  // State vector
    Eigen::MatrixXd P;  // Covariance matrix
    Eigen::MatrixXd F;  // State transition matrix
    Eigen::MatrixXd H;  // Observation matrix
    Eigen::MatrixXd R;  // Measurement noise covariance
    Eigen::MatrixXd Q;  // Process noise covariance
    Eigen::MatrixXd I;  // Identity matrix

    KalmanFilter(int state_size, int measurement_size) {
        x = Eigen::VectorXd::Zero(state_size);
        P = Eigen::MatrixXd::Identity(state_size, state_size);
        F = Eigen::MatrixXd::Identity(state_size, state_size);
        H = Eigen::MatrixXd::Zero(measurement_size, state_size);
        R = Eigen::MatrixXd::Identity(measurement_size, measurement_size);
        Q = Eigen::MatrixXd::Identity(state_size, state_size);
        I = Eigen::MatrixXd::Identity(state_size, state_size);
    }

    void predict(const Eigen::VectorXd &u) {
        x = F * x + u;
        P = F * P * F.transpose() + Q;
    }

    void update(const Eigen::VectorXd &z) {
        Eigen::VectorXd y = z - H * x;
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();

        x = x + K * y;
        P = (I - K * H) * P;
    }
};

#endif
