import pandas as pd

pd.set_option('display.precision', 15)

imu_file = './Data/0-steady-state_accel.csv'
fts_file = './Data/0-steady-state_wrench.csv'

imu_data = pd.read_csv(imu_file)
fts_data = pd.read_csv(fts_file)

imu_variances = imu_data.var()
fts_variances = fts_data.var()

fts_f_xyz_variances = fts_variances[['fx', 'fy', 'fz']]
fts_t_xyz_variances = fts_variances[['tx', 'ty', 'tz']]

print("IMU Signal Variances:\n",imu_variances)
print("\nFTS Force Variances:\n", fts_f_xyz_variances)
print("\nFTS Torque Variances:\n", fts_t_xyz_variances)