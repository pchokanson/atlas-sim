# algorithm-test.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""Simple cases for RigidBody."""


from RigidBody import RigidBody
import numpy as np
import unittest
import random

EPS = 1e-10

def vlength(vector):
	return np.sqrt(sum([v**2 for v in vector]))

def vdiff_len(v1, v2):
	return vlength([vx1 - vx2 for vx1,vx2 in zip(v1,v2)])

class RigidBodyTests(unittest.TestCase):
	def setUp(self):
		self.testVectors = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
		for i in range(30):
			self.testVectors.append([random.uniform(-5e8,5e8) for i in range(3)])
		
	def test_set_get_Q(self):
		b = RigidBody()

		Q = [1,0,0,0]
		b.set_Q(Q)
		self.assertEqual(b.state_vector[6:10], Q)
		self.assertEqual(b.get_Q(), Q)
		
		Q = [0,1,0,0]
		b.set_Q(Q)
		self.assertEqual(b.state_vector[6:10], Q)
		self.assertEqual(b.get_Q(), Q)
		
		Q = [0,0,1,0]
		b.set_Q(Q)
		self.assertEqual(b.state_vector[6:10], Q)
		self.assertEqual(b.get_Q(), Q)
		
		Q = [0,0,0,1]
		b.set_Q(Q)
		self.assertEqual(b.state_vector[6:10], Q)
		self.assertEqual(b.get_Q(), Q)

		Q = [0.5,0,0,0]
		b.set_Q(Q)
		Q = [1,0,0,0]
		for i in range(len(Q)):
			self.assertTrue(b.get_Q()[i] - Q[i] < EPS)
			self.assertTrue(b.state_vector[6+i] - Q[i] < EPS)

		Q = [3,-4,0,0]
		b.set_Q(Q)
		Q = [3/5,-4/5,0,0]
		for i in range(len(Q)):
			self.assertTrue(b.get_Q()[i] - Q[i] < EPS)
			self.assertTrue(b.state_vector[6+i] - Q[i] < EPS)

	def test_set_get_xyz(self):
		b = RigidBody()
		for xyz in self.testVectors:
			b.set_xyz(xyz)
			self.assertEqual(b.state_vector[0:3], xyz)
			self.assertEqual(b.get_xyz(), xyz)

	def test_set_get_vxyz(self):
		b = RigidBody()
		for vxyz in self.testVectors:
			b.set_vxyz(vxyz)
			self.assertEqual(b.state_vector[3:6], vxyz)
			self.assertEqual(b.get_vxyz(), vxyz)

	def test_set_get_wxyz(self):
		b = RigidBody()
		for wxyz in self.testVectors:
			b.set_wxyz(wxyz)
			self.assertEqual(b.state_vector[10:13], wxyz)
			self.assertEqual(b.get_wxyz(), wxyz)

class RigidBodyMotionTests(unittest.TestCase):
	def setUp(self):
		self.N = 100

		# These basically need to be pythagorean triples/quadruples with even sums
		self.w_states = [[      0,      0,      0],
		                 [      0,      0,2*np.pi],
		                 [      0,2*np.pi,      0],
		                 [2*np.pi,      0,      0],
		                 [4*np.pi,      0,      0],
		                 [np.sqrt(2)*np.pi,0,np.sqrt(2)*np.pi],
		                 [np.sqrt(4/3)*np.pi, np.sqrt(4/3)*np.pi, np.sqrt(4/3)*np.pi],
		                 [6*np.pi,8*np.pi,      0],
		                 [2*np.pi,4*np.pi,4*np.pi],
		                 [4*np.pi,6*np.pi,12*np.pi],
		                 [-4*np.pi,6*np.pi,-12*np.pi],
		                 [4*np.pi,-6*np.pi,12*np.pi]]
	def test_time(self):
		b = RigidBody()
		b.t = 0
		b.step(0.1)
		self.assertTrue(abs(b.t - 0.1) < EPS)
		b.step(0.2)
		self.assertTrue(abs(b.t - 0.3) < EPS)
		b.step(0.3)
		self.assertTrue(abs(b.t - 0.6) < EPS)
	
	def test_static(self):
		b = RigidBody()
		b.set_Q([1,0,0,0])
		b.set_xyz([0,0,0])
		b.set_wxyz([0,0,0])
		b.set_vxyz([0,0,0])
		b.force_torque = lambda y, t: ([0,0,0],[0,0,0])
		b.f_Icm = lambda y, t: np.eye(3) * 0.4*y[13] * 1.0**2
		for i in range(self.N):
			b.step(1.0/self.N)
			self.assertTrue(vdiff_len(b.get_Q(),[1,0,0,0]) < EPS)
			self.assertTrue(vdiff_len(b.get_xyz(),[0,0,0]) < EPS)
			self.assertTrue(vdiff_len(b.get_wxyz(),[0,0,0]) < EPS)
			self.assertTrue(vdiff_len(b.get_vxyz(),[0,0,0]) < EPS)

	def test_stationary_rotation(self):
		"""Test rotation of stationary body in all axes."""
		
		
		for w in self.w_states:
			b = RigidBody()
			b.set_wxyz(w)
			for i in range(self.N):
				b.step(1.0 / self.N)
				if i % 10 == 0:
					b.normalize_Q()
			#print("w%s -> %f:%s" % (b.get_wxyz(), vlength(b.get_Q()),b.get_Q()))
			self.assertTrue(vdiff_len(b.get_Q(), [1,0,0,0]) < 1e-4 or
			                vdiff_len(b.get_Q(), [-1,0,0,0])< 1e-4)
			self.assertTrue(vdiff_len(b.get_xyz(),[0,0,0]) < EPS)

if __name__ == "__main__":
	unittest.main(verbosity=2)