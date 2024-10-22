import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


# Read data from the imu and fts csv files

imuData = pd.read_csv('./Data/0-steady-state_accel.csv')
ax = imuData['ax'].values
ay = imuData['ay'].values
az = imuData['az'].values

ftsData = pd.read_csv('./Data/0-steady-state_wrench.csv')
fx = ftsData['fx'].values
fy = ftsData['fy'].values
fz = ftsData['fz'].values
tx = ftsData['tx'].values
ty = ftsData['ty'].values
tz = ftsData['tz'].values


mean_ax = np.mean(ax)
mean_ay = np.mean(ay)
mean_az = np.mean(az)

mean_fx = np.mean(fx)
mean_fy = np.mean(fy)
mean_fz = np.mean(fz)
mean_tx = np.mean(tx)
mean_ty = np.mean(ty)
mean_tz = np.mean(tz)

var_ax = np.var(ax)
var_ay = np.var(ay)
var_az = np.var(az)

var_fx = np.var(fx)
var_fy = np.var(fy)
var_fz = np.var(fz)
var_tx = np.var(tx)
var_ty = np.var(ty)
var_tz = np.var(tz)


print("IMU Signal Variances:")
print(f"Var(ax): {var_ax}")
print(f"Var(ay): {var_ay}")
print(f"Var(az): {var_az}")

print("\nFTS Signal Variances:")
print(f"Var(fx): {var_fx}")
print(f"Var(fy): {var_fy}")
print(f"Var(fz): {var_fz}")
print(f"Var(tx): {var_tx}")
print(f"Var(ty): {var_ty}")
print(f"Var(tz): {var_tz}")

fig, axs = plt.subplots(4, 1, figsize=(10, 15))

axs[0].plot(fx, label='fx')
axs[0].plot(fy, label='fy')
axs[0].plot(fz, label='fz')
axs[0].set_title('FTS Force Measurements at Rest')
axs[0].set_xlabel('Sample Index')
axs[0].set_ylabel('Force (N)')
axs[0].legend()

axs[1].plot(ax, label='ax')
axs[1].plot(ay, label='ay')
axs[1].plot(az, label='az')
axs[1].set_title('IMU Acceleration Measurements at Rest')
axs[1].set_xlabel('Sample Index')
axs[1].set_ylabel('Acceleration (m/s²)')
axs[1].legend()

axs[2].hist(fx, bins=30)
axs[2].set_title('Histogram of fx Measurements')
axs[2].set_xlabel('fx (N)')
axs[2].set_ylabel('Frequency')

acf_fx = acf(fx - np.mean(fx), nlags=50)
axs[3].stem(acf_fx)
axs[3].set_title('Autocorrelation of fx')
axs[3].set_xlabel('Lag')
axs[3].set_ylabel('Autocorrelation')

plt.tight_layout()
plt.show()
