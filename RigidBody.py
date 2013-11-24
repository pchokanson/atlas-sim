# algorithm-test.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""Simple object-based rigid body implementation."""

import datetime
import numpy as np
import numpy.matlib as m
from scipy.integrate import odeint
from Quaternion import Quaternion
from EarthFrame import *

class RigidBody(object):
	def Zero(y, t, *a):
		return 0.0

	def F_DX(y, t, *a):
		dy = [0.0 for i in range(len(y))]

		force, torque = a[0](y, t)
		I_cm = a[1](y, t)
		alpha = np.linalg.solve(I_cm,np.asmatrix(torque).T)
		
		dy[0] = y[3]
		dy[1] = y[4]
		dy[2] = y[5]
		
		dy[3] = force[0] / y[13]
		dy[4] = force[1] / y[13]
		dy[5] = force[2] / y[13]
		
		dy[6] = 0.5 * (-y[7]*y[10] - y[8]*y[11] - y[9]*y[12])
		dy[7] = 0.5 * ( y[6]*y[10] - y[9]*y[11] - y[8]*y[12])
		dy[8] = 0.5 * ( y[9]*y[10] + y[6]*y[11] - y[7]*y[12])
		dy[9] = 0.5 * (-y[8]*y[10] + y[7]*y[11] + y[6]*y[12])

		dy[10] = alpha[0]
		dy[11] = alpha[1]
		dy[12] = alpha[2]

		dy[13:] = [l(y,t) for l in a[2]]
		
		return dy
	
	def __init__(self):
		self.state_vector = [0.0 for i in range(14)]
		self.state_vector[6] = 1.0
		self.state_vector[13] = 1.0
		self.state_names = ["x", "y", "z", "vx", "vy", "vz", "q0", "q1", "q2", "q3",
		                    "wx", "wy", "wz", "mass"]
		self.state_f_dx = [RigidBody.Zero for i in range(14)]

		self.force_torque = lambda y, t: ([0,0,0],[0,0,0])
		# I_cm for sphere = 0.4*mr^2
		self.f_Icm = lambda y, t: np.eye(3) * 0.4*y[13] * 1.0**2


		self.t = 0.0
		self.epoch = datetime.datetime.now()

		self.facc = [0,0,0]
		self.tacc = [0,0,0]
		
	def add_state(self, x0, name, f_dx):
		"""Add a state to the state vector.
		 x0 - initial value at current time step
		 name - string name used for data output
		 f_dx - function(state, time, args)"""
		self.state_vector.append(x0)
		self.state_names.append(name)
		self.state_f_dx.append(f_dx)

	def set_xyz(self, xyz):
		assert(len(xyz) == 3)
		self.state_vector[0:3] = xyz

	def get_xyz(self):
		return self.state_vector[0:3]

	def set_vxyz(self, vxyz):
		assert(len(vxyz) == 3)
		self.state_vector[3:6] = vxyz
			
	def get_vxyz(self):
		return self.state_vector[3:6]

	def set_wxyz(self, wxyz):
		assert(len(wxyz) == 3)
		self.state_vector[10:13] = wxyz
		
	def get_wxyz(self):
		return self.state_vector[10:13]

	def set_Q(self, Q):
		assert(len(Q) == 4)
		assert(Q != [0,0,0,0])
		self.state_vector[6:10] = Q
		self.normalize_Q()

	def get_Q(self):
		return self.state_vector[6:10]

	def normalize_Q(self):
		Q = self.state_vector[6:10]
		vlen = np.sqrt(Q[0]**2 + Q[1]**2 + Q[2]**2 + Q[3]**2)
		self.state_vector[6:10] = [Q[i] / vlen for i in range(4)]

	def step(self, dt):
		new_state = odeint(RigidBody.F_DX, self.state_vector, [self.t, self.t+dt],
		                   args=(self.force_torque, self.f_Icm, self.state_f_dx[13:]))
		self.state_vector = new_state[1]
		self.t += dt

	def getDatetime(self):
		return self.epoch + datetime.timedelta(seconds=self.t)

	def __str__(self):
		return str(self.getDatetime()) + " " + str(dict(zip(self.state_names, self.state_vector)))

if __name__ == "__main__":
	b = RigidBody()

	#print(b)
	b.step(0.1)
	#print(b)
	
	import unittest
	from RigidBodyTests import *

	unittest.main(verbosity=2)
