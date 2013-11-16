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
# 6  - q0    - first quaternion component
# 7  - q1    - second quaternion component
# 8  - q2    - third quaternion component
# 9  - q3    - fourth quaternion component
# 10 - wx    - rotation about x axis (rad/s)
# 11 - wy    - rotation about y axis (rad/s)
# 12 - wz    - rotation about z axis (rad/s)


def force_torque(y):
	x = y[0:3]
	f = [0, 0, 0] #(-G * mass * earth_mass / (np.linalg.norm(x) ** 3)) * x
	t = m.matrix("0 0 0")
	return (f, t)

def dX(y, t):
	dXdt = [0 for i in range(13)]
	force, torque = force_torque(y)

	# Linear acceleration
	dXdt[3] = force[0] / mass
	dXdt[4] = force[1] / mass
	dXdt[5] = force[2] / mass

	# Angular acceleration
	alpha = I_cm_inv * np.asmatrix(torque).T
	dXdt[10] = float(alpha[0])
	dXdt[11] = float(alpha[1])
	dXdt[12] = float(alpha[2])

	# Linear Velocity (xyz coordinates)
	dXdt[0:3] = y[3:6]

	# Angular velocity

	# Quaternion derivative math: \dot{q} = Q \vec{w} / 2
	# \dot{q}_0 = 0.5 * (-q1*w0 - q2*w1 - q3*w2)
	dXdt[6] = 0.5 * (-y[7]*y[10] - y[8]*y[11] - y[9]*y[12])
	# \dot{q}_1 = 0.5 * (+q0*w0 - q3*w1 - q2*w2)
	dXdt[7] = 0.5 * ( y[6]*y[10] - y[9]*y[11] - y[8]*y[12])
	# \dot{q}_2 = 0.5 * (+q3*w0 + q0*w1 - q1*w2)
	dXdt[8] = 0.5 * ( y[9]*y[10] + y[6]*y[11] - y[7]*y[12])
	# \dot{q}_3 = 0.5 * (-q2*w0 + q1*w1 + q0*w2)
	dXdt[9] = 0.5 * (-y[8]*y[10] + y[7]*y[11] + y[6]*y[12])
	
	#print(dXdt)
	return dXdt

X0 = [0 for i in range(13)]
mass = 1.0
r = 1.0
# Sphere moment of inertia
I_cm = [[0.4*mass*(r**2), 0, 0],
        [0, 0.4*mass*(r**2), 0],
        [0, 0, 0.4*mass*(r**2)]]

I_cm_inv = np.linalg.inv(np.asmatrix(I_cm))

# rotation about x axis
X0 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4*np.pi, 0, 0]

print("   X = %s,\ndXdt = %s\n" % (X0, dX(X0, 0)))

time = np.arange(0, 1, 0.01)
y = odeint(dX, X0, time)

for i in range(10):
	print(y[i], time[i])

print(y[-1], time[-1])

for i in range(100):
	print(np.sqrt(y[i][6]**2 + y[i][7]**2 + y[i][8]**2 + y[i][9]**2))


