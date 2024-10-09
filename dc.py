import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('./Data/0-calibration_fts-accel.csv', header=0, delimiter=',')
# Reading the data
Fx = df['fx'].values
Fy = df['fy'].values
Fz = df['fz'].values
Tx = df['tx'].values
Ty = df['ty'].values
Tz = df['tz'].values
gsx = df['gx'].values
gsy = df['gy'].values
gsz = df['gz'].values

# Estimating the force bias and mass
N = len(Fx)

aForce = []
bForce = []

for i in range(N):
    aForce.append([1, 0, 0, gsx[i]])
    aForce.append([0, 1, 0, gsy[i]])
    aForce.append([0, 0, 1, gsz[i]])

    bForce.append(Fx[i])
    bForce.append(Fy[i])
    bForce.append(Fz[i])

aForce = np.array(aForce)
bForce = np.array(bForce)

xForce, residuals, rank, s = np.linalg.lstsq(aForce, bForce, rcond=None)
Fb_x, Fb_y, Fb_z, m_est = xForce

# print(xForce)


# Estimating the tourque bias and mass
aTorque = []
bTorque = []

m = m_est

for i in range(N):
    Ai = np.array([
        [0, -gsz[i], gsy[i]],
        [gsz[i], 0, -gsx[i]],
        [-gsy[i], gsx[i], 0]
    ])
    aTorque.append([1, 0, 0, m * Ai[0, 0], m * Ai[0, 1], m * Ai[0, 2]])
    aTorque.append([0, 1, 0, m * Ai[1, 0], m * Ai[1, 1], m * Ai[1, 2]])
    aTorque.append([0, 0, 1, m * Ai[2, 0], m * Ai[2, 1], m * Ai[2, 2]])

    bTorque.append(Tx[i])
    bTorque.append(Ty[i])
    bTorque.append(Tz[i])

A_torque = np.array(aTorque)
bTorque = np.array(bTorque)

xTorque, residuals, rank, s = np.linalg.lstsq(A_torque, bTorque, rcond=None)
tau_bx, tau_by, tau_bz, r_x, r_y, r_z = xTorque

# print(xTorque)

# printing the data
Fb = np.array([Fb_x, Fb_y, Fb_z])
tau_b = np.array([tau_bx, tau_by, tau_bz])
r = np.array([r_x, r_y, r_z])
print("Estimated Force Biases:", Fb)
print("Estimated Mass:", m)
print("Estimated Torque Biases:", tau_b)
print("Estimated Center of Mass:", r)
