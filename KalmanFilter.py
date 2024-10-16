import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x0):

        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
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