import numpy as np
import pandas as pd

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x0):
        # State transition matrix
        self.A = A
        # Control input matrix
        self.B = B
        # Observation matrix
        self.H = H
        # Process noise covariance
        self.Q = Q
        # Measurement noise covariance
        self.R = R
        # Estimate error covariance
        self.P = P
        # State estimate
        self.x = x0

    def predict(self, u=0):
        # Predict the next state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        # Predict the next covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        # Calculate the Kalman Gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Update the state estimate
        y = z - np.dot(self.H, self.x)  # Measurement residual
        self.x = self.x + np.dot(K, y)
        # Update the estimate error covariance
        self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, self.H)), self.P)
        return self.x

def calculate_variance(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    # Calculate the variance for each column
    variances = data.var()
    return variances

def main():
    # File paths
    imu_file = './Data/0-steady-state_accel.csv'
    fts_file = './Data/0-steady-state_wrench.csv'
    
    # Calculate variances
    imu_variances = calculate_variance(imu_file)
    fts_variances = calculate_variance(fts_file)
    
    # Print the results
    print("IMU Signal Variances:")
    print(imu_variances)
    print("\nFTS Signal Variances:")
    print(fts_variances)

    # Example Kalman Filter initialization
    A = np.array([[1]])  # State transition matrix
    B = np.array([[0]])  # Control input matrix
    H = np.array([[1]])  # Observation matrix
    Q = np.array([[1]])  # Process noise covariance
    R = np.array([[1]])  # Measurement noise covariance
    P = np.array([[1]])  # Estimate error covariance
    x0 = np.array([[0]])  # Initial state estimate

    kf = KalmanFilter(A, B, H, Q, R, P, x0)
    
    # Example usage of the Kalman Filter
    predictions = []
    measurements = [1, 2, 3, 4, 5, 6]  # Example measurements
    for z in measurements:
        pred = kf.predict()
        update = kf.update(z)
        predictions.append(update)
        print(f"Prediction: {pred}, Update: {update}")

main()
