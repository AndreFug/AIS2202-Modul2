# State estimation in [AIS2202 Cybernetics](https://www.ntnu.edu/studies/courses/AIS2202)
This is the solution for the state estimation modul in AIS2202 Cybernetics, the task is to first implement parameter estimation and then state estimation.

## Parameter estimation
Parameter estimation of the end effector of the sensor and end-effector tool, where the following;
1. Sensor bias.
2. Mass of the tool.
3. Mass center of the tool.

must be estimated using the [datasets](https://zenodo.org/records/11096791) and using the techniques from [Bias Estimation and Gravity Compensation
For Force-Torque Sensors.](http://wseas.us/e-library/conferences/crete2002/papers/444-809.pdf)

## State estimation
The contact wrench and gravity compensated force, torque  and acceleration vectors for the three experimetns is estimated by using the [datasets](https://zenodo.org/records/11096791) and implementing the methods from [A Linear Discrete Kalman Filter to Estimate the Contact Wrench of an Unknown Robot End Effector.](https://ieeexplore.ieee.org/document/10671273)

The states of the system is force, torque and acceleration of the end-effector.
1. Without external factors.
2. With mechanical vibrations.
3. With mechanical vibrations and contact with the environment.
