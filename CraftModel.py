# RigidBody.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""Simple object-based rigid body implementation."""

import numpy as np
from skaero.atmosphere import coesa

import RigidBody
import EarthFrame

class CraftModel(object):
	"""Abstract craft model.  This provides:
	- The force_torque callback
	- Mass and I_cm callbacks
	- Initial conditions
	- Additional state variables
	"""

	
	def __init__(self):
		self.force_torque = lambda y, t: ([0,0,0],[0,0,0])
		self.f_mass = lambda y, t: 1.0
		self.f_Icm = lambda y, t: np.eye(3)
	
	def sum_forces_factory(forces_torques):
		def sum_forces(y, t):
			F_acc = np.asarray([0,0,0])
			T_acc = np.asarray([0,0,0])
			for ft in forces_torques:
				F, T = ft(y, t)
				F_acc += np.asarray(F)
				T_acc += np.asarray(T)
			return (F_acc, T_acc)
		return sum_forces
	
	def gravity_force_factory(mass=1.0, mass_earth = EarthFrame.earth_mass):
		def gravity_force(y, t):
			px = y[0:3]
			norm_px = np.sqrt(px[0]**2 + px[1]**2 + px[2]**2)
			F = [-EarthFrame.G * mass * mass_earth * px[i] / (norm_px**3) for i in range(3)]
			#print("F=%s" % F)
			return (F, [0,0,0])
		
		return gravity_force

	def drag_force_factory(self, cop=[0,0,0], CD = 0.47, A=1.0, wind=[0,0,0]):
		pass
		
	def setup(self, body):
		body.f_mass = self.f_mass
		body.f_Icm = self.f_Icm
		body.force_torque = self.force_torque

class SimpleRocketModel(object):
	"""A simple rocket that should follow the rocket equation."""
	def __init__(self, **params):
		ICraftModel.__init__(self, **params)
		
		def f_mass(y, t):
			return y[13]
		


if __name__ == "__main__":
	b = RigidBody.RigidBody()
	c = CraftModel()
	
	print("g= %f" % (EarthFrame.G * EarthFrame.earth_mass / (EarthFrame.earth_radius)**2))
	
	mass = 1.0
	b.f_mass = lambda y,t: mass
	b.f_Icm = lambda y,t: 0.4 * np.eye(3) * mass
	g_ft = CraftModel.gravity_force_factory()
	b.force_torque = CraftModel.gravity_force_factory()
	r = (EarthFrame.G * EarthFrame.earth_mass / (EarthFrame.omega_earth**2)) ** (1.0/3.0)
	v = r * EarthFrame.omega_earth
	print("r=%f"%r)
	b.set_xyz([r,0,0])
	b.set_vxyz([0,-v,0])
	b.start()
	#print(b)
	for i in range(60*24):
		b.step(1.0)
		if i % 60 == 0:
			lle = EarthFrame.ECITime2LatLonElev(b.get_xyz(), b.getDatetime())
			print(lle)
			print(b)
	