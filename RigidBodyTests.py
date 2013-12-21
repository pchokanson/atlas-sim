# algorithm-test.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""Simple cases for RigidBody."""


from RigidBody import RigidBody
import numpy as np
import math
import unittest
import random
from Quaternion import Quaternion

EPS_A = 1e-10 # tolerance for simple calculations
EPS_B = 1e-5 # tolerance for more lengthy calculations

def vlength(vector):
	return np.sqrt(sum([v**2 for v in vector]))

def vdiff_len(v1, v2):
	return vlength([vx1 - vx2 for vx1,vx2 in zip(v1,v2)])

def random_vector(n=3,scale=5e8):
	return [random.uniform(-scale,scale) for i in range(n)]

def random_body(n_points=10, mass_scale=5e3, xyz_scale=5e8):
	masses = [random.uniform(0.0, mass_scale/n_points) for i in range(n_points)]
	x_p = [random.uniform(-xyz_scale,xyz_scale) for i in range(n_points)]
	y_p = [random.uniform(-xyz_scale,xyz_scale) for i in range(n_points)]
	z_p = [random.uniform(-xyz_scale,xyz_scale) for i in range(n_points)]
	
	mass = sum(masses)
	x_cm = sum([x_i * m_i for x_i, m_i in zip(masses, x_p)]) / mass
	y_cm = sum([y_i * m_i for y_i, m_i in zip(masses, y_p)]) / mass
	z_cm = sum([z_i * m_i for z_i, m_i in zip(masses, z_p)]) / mass
	
	I_xx = sum([((y_i - y_cm)**2 + (z_i - z_cm)**2) * m_i for x_i, y_i, z_i, m_i in zip(x_p, y_p, z_p, masses)])
	I_yy = sum([((x_i - x_cm)**2 + (z_i - z_cm)**2) * m_i for x_i, y_i, z_i, m_i in zip(x_p, y_p, z_p, masses)])
	I_zz = sum([((x_i - x_cm)**2 + (y_i - y_cm)**2) * m_i for x_i, y_i, z_i, m_i in zip(x_p, y_p, z_p, masses)])
	I_xy = -sum([(x_i - x_cm) * (y_i - y_cm) * m_i for x_i, y_i, z_i, m_i in zip(x_p, y_p, z_p, masses)])
	I_yz = -sum([(y_i - y_cm) * (z_i - z_cm) * m_i for x_i, y_i, z_i, m_i in zip(x_p, y_p, z_p, masses)])
	I_xz = -sum([(x_i - x_cm) * (z_i - z_cm) * m_i for x_i, y_i, z_i, m_i in zip(x_p, y_p, z_p, masses)])
	return mass, np.matrix([[I_xx, I_xy, I_xz],
                          [I_xy, I_yy, I_yz],
                          [I_xz, I_yz, I_zz]])


class RigidBodyTests(unittest.TestCase):
	def setUp(self):
		self.testVectors = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
		for i in range(30):
			self.testVectors.append([random.uniform(-5e8,5e8) for i in range(3)])
		
	def test_set_get_Q(self):
		"""Test setters and getters for state quaternion"""
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
			self.assertTrue(b.get_Q()[i] - Q[i] < EPS_A)
			self.assertTrue(b.state_vector[6+i] - Q[i] < EPS_A)

		Q = [3,-4,0,0]
		b.set_Q(Q)
		Q = [3/5,-4/5,0,0]
		for i in range(len(Q)):
			self.assertTrue(b.get_Q()[i] - Q[i] < EPS_A)
			self.assertTrue(b.state_vector[6+i] - Q[i] < EPS_A)

	def test_set_get_xyz(self):
		"""Test setters and getters for state position vector"""
		b = RigidBody()
		for xyz in self.testVectors:
			b.set_xyz(xyz)
			self.assertEqual(b.state_vector[0:3], xyz)
			self.assertEqual(b.get_xyz(), xyz)

	def test_set_get_vxyz(self):
		"""Test setters and getters for state linear velocity"""
		b = RigidBody()
		for vxyz in self.testVectors:
			b.set_vxyz(vxyz)
			self.assertEqual(b.state_vector[3:6], vxyz)
			self.assertEqual(b.get_vxyz(), vxyz)

	def test_set_get_mass(self):
		b = RigidBody()
		for i in range(100):
			m = random.uniform(1e-6,1e6)
			b.set_mass(m)
			self.assertEqual(b.get_mass(), m)
			self.assertEqual(b.mass, m)

	def test_set_get_wxyz(self):
		"""Test setters and getters for state angular velocity"""
		b = RigidBody()
		for wxyz in self.testVectors:
			b.set_wxyz(wxyz)
			self.assertEqual(b.state_vector[10:13], wxyz)
			self.assertEqual(b.get_wxyz(), wxyz)

class RigidBodyMotionTests(unittest.TestCase):
	def setUp(self):
		self.N = 100 # number of steps
		self.M = 10 # number of test cases

		# These basically need to be pythagorean triples/quadruples with even sums
		self.w_states = [[      0,      0,      0],
		                 [      0,      0,2*np.pi],
		                 [      0,2*np.pi,      0],
		                 [2*np.pi,      0,      0],
		                 [4*np.pi,      0,      0],
		                 [np.sqrt(2)*np.pi,0,np.sqrt(2)*np.pi],
		                 [6*np.pi,8*np.pi,      0],
		                 [2*np.pi,4*np.pi,4*np.pi],
		                 [-2*np.pi,4*np.pi,4*np.pi],
		                 [2*np.pi,-4*np.pi,-4*np.pi],
		                 [-2*np.pi,-4*np.pi,-4*np.pi],
		                 ## TODO:
		                 ## The following cases are somewhat problematic.  I'm not
		                 ## sure if this is a numerical issue or algorithmic.
		                 #[np.sqrt(4/3)*np.pi, np.sqrt(4/3)*np.pi, np.sqrt(4/3)*np.pi],
		                 #[4*np.pi,6*np.pi,12*np.pi],
		                 #[-4*np.pi,6*np.pi,-12*np.pi],
		                 #[4*np.pi,-6*np.pi,12*np.pi]
		                 ]
		
	def test_time(self):
		"""Test that time passes correctly."""
		b = RigidBody()
		b.t = 0
		b.step(0.1)
		self.assertTrue(abs(b.t - 0.1) < EPS_A)
		b.step(0.2)
		self.assertTrue(abs(b.t - 0.3) < EPS_A)
		b.step(0.3)
		self.assertTrue(abs(b.t - 0.6) < EPS_A)
	
	def test_static(self):
		"""Test unmoving body case."""
		b = RigidBody()
		b.set_Q([1,0,0,0])
		b.set_xyz([0,0,0])
		b.set_wxyz([0,0,0])
		b.set_vxyz([0,0,0])
		b.force_torque = lambda y, t: ([0,0,0],[0,0,0])
		b.f_Icm = lambda y, t: np.eye(3) * 0.4*b.mass * 1.0**2
		for i in range(self.N):
			b.step(1.0/self.N)
			self.assertTrue(vdiff_len(b.get_Q(),[1,0,0,0]) < EPS_A)
			self.assertTrue(vdiff_len(b.get_xyz(),[0,0,0]) < EPS_A)
			self.assertTrue(vdiff_len(b.get_wxyz(),[0,0,0]) < EPS_A)
			self.assertTrue(vdiff_len(b.get_vxyz(),[0,0,0]) < EPS_A)

	def test_constant_linear(self):
		"""Test constant linear motion in three dimensions without rotation."""
		for i in range(self.M):
			b = RigidBody()
			vxyz = random_vector()
			b.set_Q([1,0,0,0])
			b.set_xyz([0,0,0])
			b.set_wxyz([0,0,0])
			b.set_vxyz(vxyz)
			b.force_torque = lambda y, t: ([0,0,0],[0,0,0])
			b.f_Icm = lambda y, t: np.eye(3) * 0.4*b.mass * 1.0**2
			for i in range(self.N):
				b.step(1.0/self.N)
				self.assertTrue(vdiff_len(b.get_Q(),[1,0,0,0]) < EPS_A)
				self.assertTrue(vdiff_len(b.get_wxyz(),[0,0,0]) < EPS_A)
				self.assertTrue(vdiff_len(b.get_vxyz(),vxyz) < EPS_A)
			#print(vxyz,b,vdiff_len(b.get_xyz(),vxyz))
			self.assertTrue(vdiff_len(b.get_xyz(),vxyz) / vlength(vxyz) < EPS_B)

	def test_accelerating_linear(self):
		"""Test linear acceleration in three dimensions without rotation."""
		for i in range(self.M):
			b = RigidBody()
			fxyz = random_vector()
			mass = random.uniform(0.1,100)
			b.set_Q([1,0,0,0])
			b.set_xyz([0,0,0])
			b.set_wxyz([0,0,0])
			b.set_vxyz([0,0,0])
			b.set_mass(mass)
			b.force_torque = lambda y, t: (fxyz,[0,0,0])
			b.f_Icm = lambda y, t: np.eye(3) * 0.4*mass * 1.0**2
			for i in range(self.N):
				b.step(1.0/self.N)
				self.assertTrue(vdiff_len(b.get_Q(),[1,0,0,0]) < EPS_A)
				self.assertTrue(vdiff_len(b.get_wxyz(),[0,0,0]) < EPS_A)
			axyz = [f/mass for f in fxyz]
			axyz_len = vlength(axyz)
			xyz_f = [0.5*a for a in axyz]
			self.assertTrue(vdiff_len(b.get_vxyz(), axyz)/axyz_len < EPS_B)
			self.assertTrue(vdiff_len(b.get_xyz(), xyz_f)/axyz_len < EPS_B)

	def test_fixed_linear_constant_rotation(self):
		"""Test rotation of stationary body in all axes."""
		for w in self.w_states:
			b = RigidBody()
			b.set_wxyz(w)
			for i in range(self.N):
				b.step(1.0 / self.N)
				#if i % 10 == 0:
				b.normalize_Q()
			#print("w%s -> %f:%s" % (b.get_wxyz(), vlength(b.get_Q()),b.get_Q()))
			self.assertTrue(vdiff_len(b.get_Q(), [1,0,0,0]) < 1e-4 or
			                vdiff_len(b.get_Q(), [-1,0,0,0])< 1e-4)
			self.assertTrue(vdiff_len(b.get_xyz(),[0,0,0]) < EPS_A)

	def test_constant_linear_constant_rotation(self):
		"""Test rotation of stationary body in all axes while in constant linear motion."""
		for w in self.w_states:
			b = RigidBody()
			vxyz = random_vector()
			b.set_vxyz(vxyz)
			b.set_wxyz(w)
			for i in range(self.N):
				b.step(1.0 / self.N)
				#if i % 10 == 0:
				b.normalize_Q()
			#print("w%s -> %f:%s" % (b.get_wxyz(), vlength(b.get_Q()),b.get_Q()))
			self.assertTrue(vdiff_len(b.get_Q(), [1,0,0,0]) < 1e-4 or
			                vdiff_len(b.get_Q(), [-1,0,0,0])< 1e-4)
			self.assertTrue(vdiff_len(b.get_xyz(),vxyz)/vlength(vxyz) < EPS_A)

	def test_accelerating_linear_constant_rotation(self):
		"""Test rotation of stationary body in all axes while in constant linear motion."""
		for w in self.w_states:
			b = RigidBody()
			fxyz = random_vector()
			mass = random.uniform(0.1,100)
			b.set_mass(mass)
			b.force_torque = lambda y, t: (fxyz,[0,0,0])
			b.f_Icm = lambda y, t: np.eye(3) * 0.4*mass * 1.0**2
			b.set_wxyz(w)
			for i in range(self.N):
				b.step(1.0 / self.N)
				#if i % 10 == 0:
				b.normalize_Q()
			#print("w%s -> %f:%s" % (b.get_wxyz(), vlength(b.get_Q()),b.get_Q()))
			self.assertTrue(vdiff_len(b.get_Q(), [1,0,0,0]) < 1e-4 or
			                vdiff_len(b.get_Q(), [-1,0,0,0])< 1e-4)
			self.assertTrue(vdiff_len(b.get_wxyz(), w) < EPS_A)
			axyz = [f/mass for f in fxyz]
			axyz_len = vlength(axyz)
			xyz_f = [0.5*a for a in axyz]
			self.assertTrue(vdiff_len(b.get_vxyz(), axyz)/axyz_len < EPS_B)
			self.assertTrue(vdiff_len(b.get_xyz(), xyz_f)/axyz_len < EPS_B)

	def test_fixed_linear_accelerating_rotation(self):
		"""Test rotational acceleration of stationary body in each axis."""
		for axis in range(3):
			for i in range(self.M):
				b = RigidBody()
				txyz = [random.uniform(-1.0,1.0) if axis==i else 0 for i in range(3)]
				mass = random.uniform(0.1,100)
				I_cm = np.eye(3) *mass*0.4
				b.set_mass(mass)
				b.force_torque = lambda y, t: ([0,0,0],txyz)
				b.f_Icm = lambda y, t: I_cm
				for i in range(self.N):
					b.step(1.0 / self.N)
					#if i % 10 == 0:
					b.normalize_Q()

				wxyz_f = [t / (0.4*mass) for t in txyz]
				self.assertTrue(vdiff_len(b.get_wxyz(), wxyz_f) < EPS_B)
				# TODO: make an assertion that the quaternion does what we expect

	def test_constant_linear_accelerating_rotation(self):
		"""Test rotational acceleration of constant-motion body in each axis."""
		for axis in range(3):
			for i in range(self.M):
				b = RigidBody()
				txyz = [random.uniform(-1.0,1.0) if axis==i else 0 for i in range(3)]
				mass = random.uniform(0.1,100)
				vxyz = random_vector()
				b.set_vxyz(vxyz)
				I_cm = np.eye(3) *mass*0.4
				b.set_mass(mass)
				b.force_torque = lambda y, t: ([0,0,0],txyz)
				b.f_Icm = lambda y, t: I_cm
				for i in range(self.N):
					b.step(1.0 / self.N)
					#if i % 10 == 0:
					b.normalize_Q()

				wxyz_f = [t / (0.4*mass) for t in txyz]
				self.assertTrue(vdiff_len(b.get_wxyz(), wxyz_f) < EPS_B)
				self.assertTrue(vdiff_len(b.get_vxyz(), vxyz) < EPS_A)
				self.assertTrue(vdiff_len(b.get_xyz(), vxyz)/vlength(vxyz) < EPS_B)
				# TODO: make an assertion that the quaternion does what we expect

	def test_accelerating_linear_accelerating_rotation(self):
		"""Test rotational acceleration of accelerating body in each axis."""
		for axis in range(3):
			for i in range(self.M):
				b = RigidBody()
				txyz = [random.uniform(-1.0,1.0) if axis==i else 0 for i in range(3)]
				mass = random.uniform(0.1,100)
				fxyz = random_vector()
				I_cm = np.eye(3) *mass*0.4
				b.set_mass(mass)
				b.force_torque = lambda y, t: (fxyz,txyz)
				b.f_Icm = lambda y, t: I_cm
				for i in range(self.N):
					b.step(1.0 / self.N)
					#if i % 10 == 0:
					b.normalize_Q()

				wxyz_f = [t / (0.4*mass) for t in txyz]
				axyz = [f/mass for f in fxyz]
				axyz_len = vlength(axyz)
				xyz_f = [0.5*a for a in axyz]
				self.assertTrue(vdiff_len(b.get_wxyz(), wxyz_f) < EPS_B)
				self.assertTrue(vdiff_len(b.get_vxyz(), axyz)/axyz_len < EPS_B)
				self.assertTrue(vdiff_len(b.get_xyz(), xyz_f)/axyz_len < EPS_B)
				# TODO: make an assertion that the quaternion does what we expect

class RigidBodyPhysicsTests(unittest.TestCase):
	def setUp(self):
		self.N = 100 # number of steps
		self.M = 10 # number of test cases
	
	def test_torque_free_precession(self):
		"""Verify that angular momentum L is conserved in the global reference frame for 
		asymmetric bodies."""
		for i in range(self.M):
			b = RigidBody()
			mass, I_cm = random_body()
			#print("%s\n%s\n" %(I_cm, np.linalg.cholesky(I_cm)))
			
			b.f_mass = lambda y,t: mass
			b.f_Icm = lambda y,t: I_cm
			b.set_wxyz(random_vector(scale=1e1))
			
			L_g_0 = b.get_Lxyz_global()
			len_L_g_0 = vlength(L_g_0)
			#print("L_g_0 = %s" % L_g_0)
			for i in range(self.N):
				b.step(1.0/self.N)
				#print("L_g = %s, %f" % (b.get_Lxyz_global(), vdiff_len(b.get_Lxyz_global(), L_g_0)/len_L_g_0))
				self.assertTrue(vdiff_len(b.get_Lxyz_global(), L_g_0) / len_L_g_0 < EPS_B)
				
	def test_gyroscope_precession(self):
		"""Verify that gyroscopic precession works as expected."""
		b = RigidBody()
		
		w_mult = 100
		w = w_mult / (2 * np.pi)
		I_cm = np.eye(3)
		mass = 1.0
		
		b.f_mass = lambda y,t: mass
		b.f_Icm = lambda y,t: I_cm
		b.set_wxyz([w,0,0])
		
		def force_torque(y, t):
			# Force is actually zero (at least for this case, where the center of
			# mass isn't moving).
			force = [0,0,0]
			
			# Torque is constantly applied from 
			q = Quaternion(y[6:10])
			q.normalize()
			F = Quaternion([0,0,0,-1])
			
			F_lcl = (q.inverse() * F * q)[1:4]
			torque = np.cross([1,0,0], F_lcl)
			return (force, torque)
		b.force_torque = force_torque
		
		EPS = 1e-2
		N = 200
		for i in range(N):
			b.step(w_mult/(N*4))
			b.normalize_Q()
		#print("t: %8f, %s" % (b.t, RigidBody.local2body(b.get_Q(),[1,0,0])))
		#print("diff = %f" % vdiff_len([0,1,0], RigidBody.local2body(b.get_Q(),[1,0,0])))
		self.assertTrue(vdiff_len([0,1,0], RigidBody.local2body(b.get_Q(),[1,0,0])) < EPS)
		
		for i in range(N):
			b.step(w_mult/(N*4))
			b.normalize_Q()
		#print("t: %8f, %s" % (b.t, RigidBody.local2body(b.get_Q(),[1,0,0])))
		#print("diff = %f" % vdiff_len([-1,0,0], RigidBody.local2body(b.get_Q(),[1,0,0])))
		self.assertTrue(vdiff_len([-1,0,0], RigidBody.local2body(b.get_Q(),[1,0,0])) < EPS)
		
		for i in range(N):
			b.step(w_mult/(N*4))
			b.normalize_Q()
		#print("t: %8f, %s" % (b.t, RigidBody.local2body(b.get_Q(),[1,0,0])))
		#print("diff = %f" % vdiff_len([0,-1,0], RigidBody.local2body(b.get_Q(),[1,0,0])))
		self.assertTrue(vdiff_len([0,-1,0], RigidBody.local2body(b.get_Q(),[1,0,0])) < EPS)
		
		for i in range(N):
			b.step(w_mult/(N*4))
			b.normalize_Q()
		#print("t: %8f, %s" % (b.t, RigidBody.local2body(b.get_Q(),[1,0,0])))
		#print("diff = %f" % vdiff_len([1,0,0], RigidBody.local2body(b.get_Q(),[1,0,0])))
		self.assertTrue(vdiff_len([1,0,0], RigidBody.local2body(b.get_Q(),[1,0,0])) < EPS)


if __name__ == "__main__":
	unittest.main(verbosity=2)