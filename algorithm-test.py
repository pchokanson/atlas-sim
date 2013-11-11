# algorithm-test.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""Simple test of the algorithms used in the Atlas 6DOF simulation framework."""

import numpy as np
import numpy.matlib as m
from scipy.integrate import odeint

#import datetime
from EarthFrame import *



# The state vector X is laid out as follows:
# 0  - x     - x axis coordinate (m)
# 1  - y     - y axis coordinate (m)
# 2  - z     - z axis coordinate (m)
# 3  - vx    - x velocity (m/s)
# 4  - vy    - y velocity (m/s)
# 5  - vz    - z velocity (m/s)
# 6  - psi   - first Euler angle (rad)
# 7  - phi   - second Euler angle (rad)
# 8  - theta - third Euler angle (rad)
# 9  - wx    - rotation about x axis (rad/s)
# 10 - wy    - rotation about y axis (rad/s)
# 11 - wz    - rotation about z axis (rad/s)


def force_torque(y):
	x = y[0:3]
	f = [-1, 0, 0] #(-G * mass * earth_mass / (np.linalg.norm(x) ** 3)) * x
	t = m.matrix("0 0 0")
	return (f, t)

def dX(y, t):
	dXdt = [0 for i in range(12)]
	force, torque = force_torque(y)

	# Linear acceleration
	dXdt[3] = force[0] / mass
	dXdt[4] = force[1] / mass
	dXdt[5] = force[2] / mass

	# Angular acceleration
	alpha = I_cm_inv * np.asmatrix(torque).T
	dXdt[9] = float(alpha[0])
	dXdt[10] = float(alpha[1])
	dXdt[11] = float(alpha[2])

	# Linear Velocity (xyz coordinates)
	dXdt[0:3] = y[3:6]

	# Angular velocity
	
	# Precomputed terms
	sin_theta = np.sin(y[8])
	cos_theta = np.cos(y[8])
	sin_phi = np.sin(y[7])
	cos_phi = np.sin(y[7])

	# Angular terms derived from (3.5-2) in thomson1986introduction
	# \dot{\psi} = (\omega_x \sin\phi + \omega_y \cos\phi)/\sin\theta
	dXdt[6] = (y[9] * sin_phi + y[10] * cos_phi) / sin_theta
	# \dot{\phi} = \omega_z - \frac{\cos\theta}{\sin\theta}(\omega_x \sin\phi + \omega_y \cos\phi)
	dXdt[7] = y[11] - (y[9] * sin_phi + y[10] * cos_phi) * cos_theta / sin_theta
	# \dot{\theta} = \omega_x \cos\phi - \omega_y \sin\phi
	dXdt[8] = y[9] * cos_phi - y[10] * sin_phi

	#print(dXdt)
	return dXdt

X0 = [i+1 for i in range(12)]
mass = 1.0
r = 1.0
# Sphere moment of inertia
I_cm = [[0.4*mass*(r**2), 0, 0],
[0, 0.4*mass*(r**2), 0],
[0, 0, 0.4*mass*(r**2)]]

I_cm_inv = np.linalg.inv(np.asmatrix(I_cm))

print("   X = %s,\ndXdt = %s\n" % (X0, dX(X0, 0)))


time = np.arange(0, 1, 0.01)
y = odeint(dX, X0, time)

print(y[-1], time[-1])



