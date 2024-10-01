import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('sensor_data.csv')

# Step 1: Bias Estimation
# Calculate the bias by averaging each force and torque component
force_bias = df[['fx', 'fy', 'fz']].mean()
torque_bias = df[['tx', 'ty', 'tz']].mean()

# Subtract the bias from the data to correct the measurements
df['fx_corr'] = df['fx'] - force_bias['fx']
df['fy_corr'] = df['fy'] - force_bias['fy']
df['fz_corr'] = df['fz'] - force_bias['fz']

df['tx_corr'] = df['tx'] - torque_bias['tx']
df['ty_corr'] = df['ty'] - torque_bias['ty']
df['tz_corr'] = df['tz'] - torque_bias['tz']

# Step 2: Gravity Compensation
# Gravity vector in the world frame (assuming gravity is in the negative z direction)
g_world = np.array([0, 0, -9.81])

# Extract the rotation matrices and reshape them into 3x3 matrices
R_wf = df[['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']].values.reshape(-1, 3, 3)

# Apply gravity compensation
gravity_force = []
for i in range(len(df)):
    g_ft = R_wf[i].dot(g_world)  # Transform gravity vector into FTS frame
    gravity_force.append(g_ft)

# Convert gravity_force list to a DataFrame
gravity_force_df = pd.DataFrame(gravity_force, columns=['gx_comp', 'gy_comp', 'gz_comp'])

# Subtract gravity compensation from the corrected force values
df['fx_final'] = df['fx_corr'] - gravity_force_df['gx_comp']
df['fy_final'] = df['fy_corr'] - gravity_force_df['gy_comp']
df['fz_final'] = df['fz_corr'] - gravity_force_df['gz_comp']

# Step 3: Save the Corrected Data
# Save the corrected force and torque data to a new CSV file
df[['t', 'fx_final', 'fy_final', 'fz_final', 'tx_corr', 'ty_corr', 'tz_corr']].to_csv('corrected_sensor_data.csv', index=False)

print("Bias estimation and gravity compensation applied. Corrected data saved to 'corrected_sensor_data.csv'.")
