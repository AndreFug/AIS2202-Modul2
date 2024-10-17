import pandas as pd

imu_file = './Data/0-steady-state_accel.csv'
fts_file = './Data/0-steady-state_wrench.csv'

imu_data = pd.read_csv(imu_file)
fts_data = pd.read_csv(fts_file)

imu_variances = imu_data.var()
fts_variances = fts_data.var()

print("IMU Signal Variances:\n",imu_variances)
print("\nFTS Signal Variances:\n",fts_variances)
