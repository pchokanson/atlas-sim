# algorithm-test.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""Simple test of the algorithms used in the Atlas 6DOF simulation framework."""

#import yaml
import numpy as np
import numpy.matlib as m
import datetime

from EarthFrame import *

x = m.matrix([earth_radius + 191190.0,0,0]).T
w = m.matrix("0 0 0").T

mass = 1.0

x_dot = m.matrix("0 7792.8 3000").T
w_dot = m.matrix("0 0 0").T

dt = 1.0 #0.1

def f(x, x_dot):
	return (-G * mass * earth_mass / (np.linalg.norm(x) ** 3)) * x

for i in range(10000):
	x_dot1 = x_dot
	x1 = x

	x_dot2 = x_dot1 + f(x1,x_dot1) * dt / (2 * mass)
	x2 = x1 + x_dot2 * dt / 2

	x_dot3 = x_dot2 + f(x2,x_dot2) * dt / (2 * mass)
	x3 = x1 + x_dot2 * dt / 2

	x_dot4 = x_dot1 + f(x3, x_dot3) * dt / mass
	x4 = x1 + x_dot4 * dt / 2

	x_dot = (x_dot1 + 2*x_dot2 + 2*x_dot3 + x_dot4) / 6.0
	x = (x1 + 2*x2 + 2*x3 + x4) / 6.0


	if i % 1000 == 0:
		print(ECITime2LatLonElev(x, datetime.datetime.now()))
		#print("%s %s" % (x.T, x_dot.T))
		#print("N= %f, %f" % (np.linalg.norm(x), np.linalg.norm(x_dot)))

		#E_p = (2*np.pi * earth_density * G/3) * (np.linalg.norm(x)**2 - (3*earth_radius**2))
		#E_p = - mass * G *  earth_mass / np.linalg.norm(x)
		#E_k = (mass * (np.linalg.norm(x_dot)**2) / 2)
		#E =  E_p + E_k

		#print("Ep + Ek = %f + %f = %f" % (E_p, E_k, E))

