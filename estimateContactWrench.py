import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV files
dfBaseline_accel = pd.read_csv('./Data/1-baseline_accel.csv')
dfBaseline_wrench = pd.read_csv('./Data/1-baseline_wrench.csv')
dfBaseline_orientations = pd.read_csv('./Data/1-baseline_orientations.csv')

dfVibrations_accel = pd.read_csv('./Data/2-vibrations_accel.csv')
dfVibrations_wrench = pd.read_csv('./Data/2-vibrations_wrench.csv')
dfVibrations_orientations = pd.read_csv('./Data/2-vibrations_orientations.csv')

dfVibrationsAndContact_accel = pd.read_csv('./Data/3-vibrations-contact_accel.csv')
dfVibrationsAndContact_wrench = pd.read_csv('./Data/3-vibrations-contact_wrench.csv')
dfVibrationsAndContact_orientations = pd.read_csv('./Data/3-vibrations-contact_orientations.csv')

class Fusion:
    def __init__(self):
        # Define the dimensions of your state vector and measurement vector
        self.n = 6  # Number of states (forces and torques in world frame)

        # Initialize the state vector (forces and torques in world frame)
        self.x = np.zeros((self.n, 1))

        # Initialize the state transition matrix (F)
        # Assuming a simple model where the state remains constant between measurements
        self.F = np.eye(self.n)

        # Initialize the process noise covariance matrix (Q)
        self.Q = np.eye(self.n) * 0.0001  # Adjust based on system dynamics

        # Initialize the error covariance matrix (P)
        self.P = np.eye(self.n) * 1.0  # Initial uncertainty

        # Measurement noise covariance matrices (to be tuned)
        self.R_accel = np.eye(3) * 0.1
        self.R_wrench = np.eye(6) * 0.05

        # Placeholder for the latest rotation matrix
        self.rotation_matrix = np.eye(3)
        
        # Mass of the end effector (replace with actual mass)
        self.mass = 1.0  # kg

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z, H, R):
        # Update the state with measurement z
        y = z - np.dot(H, self.x)
        S = np.dot(np.dot(H, self.P), H.T) + R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, H), self.P)

    def process_data(self, accel_data, wrench_data, orientation_data):
        # Add a 'type' column to each dataframe
        accel_data['type'] = 'accel'
        wrench_data['type'] = 'wrench'
        orientation_data['type'] = 'orientation'

        # Ensure that the timestamp column is correctly formatted
        accel_data['t'] = pd.to_datetime(accel_data['t'])
        wrench_data['t'] = pd.to_datetime(wrench_data['t'])
        orientation_data['t'] = pd.to_datetime(orientation_data['t'])

        # Combine dataframes
        combined_data = pd.concat([accel_data, wrench_data, orientation_data], ignore_index=True)

        # Sort by timestamp
        combined_data.sort_values(by='t', inplace=True)

        # Reset index after sorting
        combined_data.reset_index(drop=True, inplace=True)

        # Initialize variables to store the latest rotation matrix
        rotation_matrix = np.eye(3)

        # Store estimated states for plotting
        estimated_states = []

        # Process data in order
        for index, row in combined_data.iterrows():
            measurement_type = row['type']

            if measurement_type == 'orientations':
                # Update the rotation matrix
                rotation_matrix = self.extract_rotation_matrix(row)
                self.rotation_matrix = rotation_matrix
            else:
                # Predict step
                self.predict()

                if measurement_type == 'accel':
                    # Extract acceleration measurements
                    accel_measurement = row[['ax', 'ay', 'az']].values.reshape(3, 1)

                    # Transform acceleration to world frame using the rotation matrix
                    accel_world = np.dot(self.rotation_matrix, accel_measurement)

                    # Construct observation matrix H_accel
                    # Define how accelerations relate to forces (Newton's second law)
                    H_accel = np.zeros((3, self.n))
                    H_accel[:, 0:3] = np.eye(3) / self.mass  # ax = fx / m, etc.

                    z = accel_world  # Measurement in world frame
                    R = self.R_accel
                    self.update(z, H_accel, R)

                elif measurement_type == 'wrench':
                    # Extract wrench measurements
                    wrench_measurement = row[['fx', 'fy', 'fz', 'tx', 'ty', 'tz']].values.reshape(6, 1)

                    # Transform wrench to world frame using the rotation matrix
                    force = wrench_measurement[0:3]
                    torque = wrench_measurement[3:6]
                    force_world = np.dot(self.rotation_matrix, force)
                    torque_world = np.dot(self.rotation_matrix, torque)
                    wrench_world = np.vstack((force_world, torque_world))

                    # Observation matrix H_wrench
                    H_wrench = np.eye(self.n)  # Direct measurement in world frame

                    z = wrench_world
                    R = self.R_wrench
                    self.update(z, H_wrench, R)

                # Store the estimated state
                estimated_states.append({
                    't': row['t'],
                    'fx': self.x[0, 0],
                    'fy': self.x[1, 0],
                    'fz': self.x[2, 0],
                    'tx': self.x[3, 0],
                    'ty': self.x[4, 0],
                    'tz': self.x[5, 0]
                })

        # Convert estimated states to DataFrame
        estimated_states_df = pd.DataFrame(estimated_states)

        return estimated_states_df

    def extract_rotation_matrix(self, row):
        # Extract rotation matrix components from the row
        r11 = row['r11']
        r12 = row['r12']
        r13 = row['r13']
        r21 = row['r21']
        r22 = row['r22']
        r23 = row['r23']
        r31 = row['r31']
        r32 = row['r32']
        r33 = row['r33']

        rotation_matrix = np.array([[r11, r12, r13],
                                    [r21, r22, r23],
                                    [r31, r32, r33]])
        return rotation_matrix

# Usage example:

# Create an instance of the Fusion class
fusion = Fusion()

# Process the data for the baseline experiment
estimated_states_baseline = fusion.process_data(dfBaseline_accel, dfBaseline_wrench, dfBaseline_orientations)

# Optionally, plot the estimated forces and torques over time
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(estimated_states_baseline['t'], estimated_states_baseline['fx'], label='Estimated fx')
plt.plot(estimated_states_baseline['t'], estimated_states_baseline['fy'], label='Estimated fy')
plt.plot(estimated_states_baseline['t'], estimated_states_baseline['fz'], label='Estimated fz')
plt.legend()
plt.title('Estimated Forces over Time')
plt.xlabel('Time')
plt.ylabel('Force (N)')

plt.subplot(2, 1, 2)
plt.plot(estimated_states_baseline['t'], estimated_states_baseline['tx'], label='Estimated tx')
plt.plot(estimated_states_baseline['t'], estimated_states_baseline['ty'], label='Estimated ty')
plt.plot(estimated_states_baseline['t'], estimated_states_baseline['tz'], label='Estimated tz')
plt.legend()
plt.title('Estimated Torques over Time')
plt.xlabel('Time')
plt.ylabel('Torque (Nm)')

plt.tight_layout()
plt.show()


# Save the estimated states to a CSV file
# estimatedSignal = estimated_states_baseline + estimated_states_vibrations + estimated_states_vibrationsAndContact
estimated_states_baseline.to_csv('./estimated.csv', index=False)
