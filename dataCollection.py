import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the CSV file
df = pd.read_csv('Data/0-calibration_fts-accel.csv', header=0, delimiter=',')

df.columns = df.columns.str.strip()
print(df.columns)
# Step 1: Bias Estimation
# Calculate the bias by averaging each force and torque component
force_bias = df.iloc[:, 0:3].mean()
torque_bias = df.iloc[:, 3:6].mean()
print(force_bias)
print(torque_bias)
# Subtract the bias from the data to correct the measurements
df['fx_corr'] = df['fx'].astype(float) - force_bias['fx']
df['fy_corr'] = df['fy'].astype(float) - force_bias['fy']
df['fz_corr'] = df['fz'].astype(float) - force_bias['fz']

df['tx_corr'] = df['tx'].astype(float) - torque_bias['tx']
df['ty_corr'] = df['ty'].astype(float) - torque_bias['ty']
df['tz_corr'] = df['tz'].astype(float) - torque_bias['tz']


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

plt.figure(figsize=(14, 10))

# Plot Force X
plt.subplot(3, 2, 1)
plt.plot(df['fx'], label='fx (original)', linestyle='--')
plt.plot(df['fx_final'], label='fx_final (corrected)')
plt.title('Force X (N)')
plt.legend()

# Plot Force Y
plt.subplot(3, 2, 2)
plt.plot(df['fy'], label='fy (original)', linestyle='--')
plt.plot(df['fy_final'], label='fy_final (corrected)')
plt.title('Force Y (N)')
plt.legend()

# Plot Force Z
plt.subplot(3, 2, 3)
plt.plot(df['fz'], label='fz (original)', linestyle='--')
plt.plot(df['fz_final'], label='fz_final (corrected)')
plt.title('Force Z (N)')
plt.legend()

# Plot Torque X
plt.subplot(3, 2, 4)
plt.plot(df['tx'], label='tx (original)', linestyle='--')
plt.plot(df['tx_corr'], label='tx_corr (corrected)')
plt.title('Torque X (Nm)')
plt.legend()

# Plot Torque Y
plt.subplot(3, 2, 5)
plt.plot(df['ty'], label='ty (original)', linestyle='--')
plt.plot(df['ty_corr'], label='ty_corr (corrected)')
plt.title('Torque Y (Nm)')
plt.legend()

# Plot Torque Z
plt.subplot(3, 2, 6)
plt.plot(df['tz'], label='tz (original)', linestyle='--')
plt.plot(df['tz_corr'], label='tz_corr (corrected)')
plt.title('Torque Z (Nm)')
plt.legend()

plt.tight_layout()
plt.show()