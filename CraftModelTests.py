# CraftModelTests.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""Simple cases for CraftModel."""

from RigidBody import RigidBody
from CraftModel import CraftModel
import numpy as np
import math
import unittest
import random
from Quaternion import Quaternion
import EarthFrame

class CraftGravityTests(unittest.TestCase):
	def test_geosynchronous(self):
		b = RigidBody()
		c = CraftModel()
		mass = 5.0
		b.f_mass = lambda y,t: mass
		b.f_Icm = lambda y,t: 0.4 * np.eye(3) * mass
		#g_ft = CraftModel.gravity_force_factory()
		b.force_torque = CraftModel.gravity_force_factory(mass=5.0)
		r = (EarthFrame.G * EarthFrame.earth_mass / (EarthFrame.omega_earth**2)) ** (1.0/3.0)
		v = r * EarthFrame.omega_earth
		#print("r = %f, v = %f" % (r,v))
		b.set_xyz([r,0,0])
		b.set_vxyz([0,-v,0])
		b.set_epoch(EarthFrame.epoch_j2000)
		b.start()
		#print(b)
		for i in range(60):
			b.step(60.0)
			lat, lon, elev = EarthFrame.ECITime2LatLonElev(b.get_xyz(), b.getDatetime())
			#print(b)
			#print(lat, lon, elev)
			self.assertTrue(abs(lat - 0.0) < 1e-3)
			self.assertTrue(abs(lon - 0.0) < 1e-3)
			self.assertTrue(abs(elev + EarthFrame.earth_radius - r) < 1e-3)
			self.assertTrue(abs(np.linalg.norm(b.get_vxyz()) - v) < 1e-3)

	#def test_ballistic(self):
		#for i in range(50):
			#b = RigidBody()
			#c = CraftModel()

if __name__ == "__main__":
	unittest.main()
