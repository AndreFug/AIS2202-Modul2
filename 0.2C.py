import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    def predict(self, u=np.zeros((6, 1))):
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

class Fusion:
    def __init__(self, A, B, H, Q, R, P, x0):
        # Initialize Kalman Filter with given state space model
        self.kalman_filter = KalmanFilter(A, B, H, Q, R, P, x0)

    def run_experiment(self, accel_file, wrench_file, orientation_file, output_file):
        # Load data from CSV files
        accel_data = pd.read_csv(accel_file)
        wrench_data = pd.read_csv(wrench_file)
        orientation_data = pd.read_csv(orientation_file)

        # Drop any rows with missing values
        accel_data.dropna(inplace=True)
        wrench_data.dropna(inplace=True)
        orientation_data.dropna(inplace=True)

        # Inspect the first few rows of the data to check the timestamps
        print("First few rows of Accel Data:", accel_data.head())
        print("First few rows of Wrench Data:", wrench_data.head())
        print("First few rows of Orientation Data:", orientation_data.head())

        # Set the index to the timestamp column
        accel_data.set_index('t', inplace=True)
        wrench_data.set_index('t', inplace=True)
        orientation_data.set_index('t', inplace=True)

        # Optional: round or convert timestamps to the nearest second to avoid precision mismatches
        accel_data.index = pd.to_datetime(accel_data.index).round('S')
        wrench_data.index = pd.to_datetime(wrench_data.index).round('S')
        orientation_data.index = pd.to_datetime(orientation_data.index).round('S')

        # Interpolate to synchronize data
        accel_data = accel_data.interpolate()
        wrench_data = wrench_data.interpolate()
        orientation_data = orientation_data.interpolate()

        # Merge data on timestamps
        merged_data = pd.concat([accel_data, wrench_data, orientation_data], axis=1, join='inner')

        # After merging, check how many samples are available
        print(f"Number of samples after merging: {len(merged_data)}")
        if len(merged_data) == 0:
            print("No matching timestamps found after merging. Adjust timestamp handling.")
            return

        results = []
        # Iterate over each sample and update the Kalman filter
        for i in range(len(merged_data)):
            # Access sensor data
            accel_sample = merged_data.iloc[i][['ax', 'ay', 'az']].values.reshape(-1, 1)
            wrench_sample = merged_data.iloc[i][['fx', 'fy', 'fz', 'tx', 'ty', 'tz']].values.reshape(-1, 1)

            # Predict and update the Kalman filter
            pred = self.kalman_filter.predict()
            update = self.kalman_filter.update(wrench_sample)
            print(f"Index: {i}, Prediction: {pred.flatten()}, Update: {update.flatten()}")

            results.append({'Index': i, 'Prediction': pred.flatten(), 'Update': update.flatten()})

        # Check if results are populated
        if len(results) > 0:
            # Convert list of dictionaries to DataFrame properly
            results_df = pd.DataFrame(results)
            # Expand the 'Prediction' and 'Update' arrays into separate columns
            pred_df = pd.DataFrame(results_df['Prediction'].tolist(), columns=['Pred_fx', 'Pred_fy', 'Pred_fz', 'Pred_tx', 'Pred_ty', 'Pred_tz'])
            update_df = pd.DataFrame(results_df['Update'].tolist(), columns=['Upd_fx', 'Upd_fy', 'Upd_fz', 'Upd_tx', 'Upd_ty', 'Upd_tz'])
            results_combined = pd.concat([results_df['Index'], pred_df, update_df], axis=1)
            # Save the results to a CSV file
            results_combined.to_csv(output_file, index=False)
            print(f"Experiment results saved to {output_file}")
        else:
            print("No data in results to save.")        

def calculate_variance(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    # Calculate the variance for each column
    variances = data.var()
    return variances

def main():
    # File paths for steady-state sensor data
    imu_file = './Data/0-steady-state_accel.csv'
    fts_file = './Data/0-steady-state_wrench.csv'
    
    # Calculate variances
    imu_variances = calculate_variance(imu_file)
    fts_variances = calculate_variance(fts_file)
    
    # Print the calculated variances
    print("IMU Signal Variances:")
    print(imu_variances)
    print("\nFTS Signal Variances:")
    print(fts_variances)

    # Ensure the correct column names are used
    imu_variances = imu_variances[['ax', 'ay', 'az']]
    fts_variances = fts_variances[['fx', 'fy', 'fz', 'tx', 'ty', 'tz']]
    
    # Example Kalman Filter initialization
    # State vector includes force and torque in x, y, z (6 dimensions)
    A = np.eye(6)  # State transition matrix (6x6 identity matrix)
    B = np.zeros((6, 3))  # Control input matrix (no control input assumed)
    H = np.eye(6)  # Observation matrix (direct observation of state)

    # Use calculated variances to set process and measurement noise covariance matrices
    # For simplicity, we can set Q and R to be small non-zero values if variances are zero
    Q_values = imu_variances.values
    R_values = fts_variances[['tx', 'ty', 'tz']].values  # Adjust if necessary

    # Ensure Q and R have correct dimensions
    Q = np.diag(np.concatenate((Q_values, fts_variances[['fx', 'fy', 'fz']].values)))
    R = np.diag(fts_variances[['tx', 'ty', 'tz']].values)

    # Initial estimate error covariance and initial state
    P = np.eye(6)  # Estimate error covariance
    x0 = np.zeros((6, 1))  # Initial state estimate (6x1 zero vector)

    # Initialize the Fusion class with the Kalman filter
    fusion = Fusion(A, B, H, Q, R, P, x0)
    
    # Run experiments and save results to CSV
    fusion.run_experiment('./Data/1-baseline_accel.csv', './Data/1-baseline_wrench.csv', './Data/1-baseline_orientations.csv', './Data/experiment_results.csv')

    # Optionally, plot the results (if visualization is needed)
    results_df = pd.read_csv('./Data/experiment_results.csv')
    time = results_df['Index']
    plt.figure(figsize=(12, 6))
    plt.plot(time, results_df['Upd_fx'], label='Updated Force X')
    plt.plot(time, results_df['Upd_fy'], label='Updated Force Y')
    plt.plot(time, results_df['Upd_fz'], label='Updated Force Z')
    plt.xlabel('Time')
    plt.ylabel('Force Estimates')
    plt.legend()
    plt.show()

main()
