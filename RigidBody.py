# RigidBody.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""Simple object-based rigid body implementation.

The states are:
0  - px    - x axis coordinate (m)
1  - py    - y axis coordinate (m)
2  - pz    - z axis coordinate (m)
3  - vx    - x velocity (m/s)
4  - vy    - y velocity (m/s)
5  - vz    - z velocity (m/s)
6  - q0    - first quaternion component
7  - q1    - second quaternion component
8  - q2    - third quaternion component
9  - q3    - fourth quaternion component
10 - wx    - rotation about x axis (rad/s)
11 - wy    - rotation about y axis (rad/s)
12 - wz    - rotation about z axis (rad/s)
"""

import datetime
import numpy as np
import numpy.matlib as m
from scipy.integrate import odeint
import scipy
from Quaternion import Quaternion
import Quaternion as Q
from EarthFrame import *

class RigidBody(object):
	def Zero(y, t, *a):
		return 0.0

	def local2body(q, x):
		"""Convert a vector in the local inertial space to the body reference 
		frame."""
		q1 = Quaternion(q)
		x = Quaternion([0, x[0], x[1], x[2]])
		
		return (q1 * x * q1.inverse()).q[1:4]
	
	def body2local(q, x):
		"""Convert a vector in the body space to the local inertial reference 
		frame."""
		q1 = Quaternion(q)
		x = Quaternion([0, x[0], x[1], x[2]])
		
		return (q1.inverse() * x * q1).q[1:4]
	
	def F_DX(t, y, a):
		#print("F_DX:\n\ty=%s\n\tt=%s\n\ta=%s" % (y, t, a))
		dy = [0.0 for i in range(len(y))]

		force, torque = a[0](y, t)
		I_cm = a[1](y, t)
		mass = a[2](y, t)
		
		# The (w x I_cm w) term originates in the Newton-Euler equations, and
		# should correspond to torque-free precession.
		q = y[6:10]
		q_w = np.asarray([0, y[10], y[11], y[12]])
		q_dot = Q.mult_s_q(0.5, Q.mult_q_q(q, q_w))

		pt_q_vec = np.asmatrix(Q.mult_q_q(Q.inv_q(q), q_dot)[1:4]).T
		pseudo_torque = 4 * np.cross(pt_q_vec.T, (I_cm * pt_q_vec).T).T

		alpha = np.linalg.solve(I_cm,np.asmatrix(torque).T - pseudo_torque)
		
		# This assertion only holds for systems with diagonal I_cm, which is not
		# the general case
		#assert(np.linalg.norm(pseudo_torque) < 1e-6)
		
		#q_len = np.sqrt(y[6]**2 + y[7]**2 + y[8]**2 + y[9]**2)
		#q = [yq / q_len for yq in y[6:10]]
		
		dy[0] = y[3]
		dy[1] = y[4]
		dy[2] = y[5]
		
		dy[3] = force[0] / mass
		dy[4] = force[1] / mass
		dy[5] = force[2] / mass
		
		dy[6:10] = q_dot
		#dy[6] = 0.5 * (-y[7]*y[10] - y[8]*y[11] - y[9]*y[12])
		#dy[7] = 0.5 * ( y[6]*y[10] - y[9]*y[11] - y[8]*y[12])
		#dy[8] = 0.5 * ( y[9]*y[10] + y[6]*y[11] - y[7]*y[12])
		#dy[9] = 0.5 * (-y[8]*y[10] + y[7]*y[11] + y[6]*y[12])

		dy[10] = alpha[0]
		dy[11] = alpha[1]
		dy[12] = alpha[2]

		dy[13:] = [l(y,t) for l in a[3]]
		#print("dy=%s" % dy)
		return np.asarray(dy)
	
	def __init__(self):
		self.state_vector = [0.0 for i in range(13)]
		self.state_vector[6] = 1.0
		self.state_names = ["x", "y", "z", "vx", "vy", "vz", "q0", "q1", "q2", "q3",
		                    "wx", "wy", "wz"]
		self.state_f_dx = [RigidBody.Zero for i in range(13)]
		self.mass = 1.0
		self.force_torque = lambda y, t: ([0,0,0],[0,0,0])
		# I_cm for sphere = 0.4*mr^2
		self.f_Icm = lambda y, t: np.eye(3) * 0.4*self.mass * 1.0**2
		
		self.f_mass = lambda y, t: self.mass

		self.t = 0.0
		self.epoch = datetime.datetime.now()

		self.facc = [0,0,0]
		self.tacc = [0,0,0]
		
		self.integrator = scipy.integrate.ode(RigidBody.F_DX)
		self.integrator.set_integrator("dopri5", first_step=0.1, max_step=1.0)
		self.__started = False

	def start(self):
		#self.F_DX = partial(
		self.__started = True
		self.integrator.set_initial_value(self.state_vector, self.t)
		self.integrator.set_f_params((self.force_torque, self.f_Icm, self.f_mass, self.state_f_dx[13:]))

	def add_state(self, x0, name, f_dx):
		"""Add a state to the state vector.
		 x0 - initial value at current time step
		 name - string name used for data output
		 f_dx - function(state, time, args)"""
		assert self.__started == False
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
	
	def get_wxyz_global(self):
		"""Return angular velocity in global (nonrotating) reference frame."""
		q = Quaternion(self.get_Q())
		w_global = q * Quaternion([0] + list(self.get_wxyz())) * q.inverse()
		return w_global[1:4]
	
	def get_Lxyz_global(self):
		"""Return angular momentum in global (nonrotating) reference frame."""
		q = Quaternion(self.get_Q())
		# Note that local angular momentum doesn't really make sense, but it's part
		# of our conversion process
		I_cm = self.f_Icm(self.state_vector, self.t)
		L_local = I_cm * np.asmatrix(self.get_wxyz()).T
		L_global = q * Quaternion([0] + list(L_local)) * q.inverse()
		return L_global[1:4]
	
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

	def set_mass(self, mass):
		self.mass = mass

	def get_mass(self):
		return self.mass

	def step(self, dt):
		
		#new_state = odeint(RigidBody.F_DX, self.state_vector, [self.t, self.t+dt],
		                   #args=(self.force_torque, self.f_Icm, self.f_mass, self.state_f_dx[13:]))
		self.state_vector = self.integrator.integrate(self.t + dt)
		#self.state_vector = new_state[1]
		self.t += dt
		#print("t: %f, s=%s" % (self.t, self.state_vector))

	def getDatetime(self):
		return self.epoch + datetime.timedelta(seconds=self.t)

	def __str__(self):
		return str(self.getDatetime()) + " " + str(dict(zip(self.state_names, self.state_vector)))

if __name__ == "__main__":
	PROFILE = False
	b = RigidBody()

	##print(b)
	#b.step(0.1)
	##print(b)
	
	import unittest
	import cProfile, pstats
	from RigidBodyTests import *

	if PROFILE:
		pr = cProfile.Profile()
		print("Running unit tests with cProfile")
		pr.enable()
		unittest.main(exit=False)
		pr.disable()
		print("Printing results to RigidBody.log")
		pr.dump_stats("RigidBody.log")
		ps = pstats.Stats(pr)
		ps.sort_stats('cumulative')
		ps.print_stats()
	else:
		unittest.main(verbosity=2, exit=False)
