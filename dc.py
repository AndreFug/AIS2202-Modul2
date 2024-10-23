import pandas as pd
import numpy as np

df = pd.read_csv('./Data/0-calibration_fts-accel.csv', header=0, delimiter=',')
# Reading the data
Fx = df['fx'].values
Fy = df['fy'].values
Fz = df['fz'].values
Tx = df['tx'].values
Ty = df['ty'].values
Tz = df['tz'].values
gx = df['gx'].values
gy = df['gy'].values
gz = df['gz'].values

# Estimating the force bias and mass
K = len(Fx) # 24 Samples

aForce = []
bForce = []

for i in range(K):
    aForce.append([1, 0, 0, gx[i]])
    aForce.append([0, 1, 0, gy[i]])
    aForce.append([0, 0, 1, gz[i]])

    bForce.append(Fx[i])
    bForce.append(Fy[i])
    bForce.append(Fz[i])

aForce = np.array(aForce)
bForce = np.array(bForce)

xForce, residuals, rank, s = np.linalg.lstsq(aForce, bForce, rcond=None)
Fb_x, Fb_y, Fb_z, m = xForce


# Estimating the tourque bias and mass
aTorque = []
bTorque = []

for i in range(K):
    Ai = np.array([
        [0, -gz[i], gy[i]],
        [gz[i], 0, -gx[i]],
        [-gy[i], gx[i], 0]
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

Fb = np.array([Fb_x, Fb_y, Fb_z])
tau_b = np.array([tau_bx, tau_by, tau_bz])
r = np.array([r_x, r_y, r_z])

print("Estimated Force Biases:", Fb)
print("Estimated Mass:", m)
print("Estimated Torque Biases:", tau_b)
print("Estimated Center of Mass:", r)