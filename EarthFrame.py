# EarthFrame.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""
EarthFrame provides several useful constants and conversion functions between
reference frames that are used in the Vertical Limit Labs project.
Specifically, this covers the Earth-centered inertial (ECI) cartesian system
and traditional latitude and longitude.

Note that ECI takes the Earth's rotation into account, so an absolute reference
epoch is required.  By default, this is J2000, (a common standard).  Times
should *not* include leap seconds (the Earth keeps spinning regardless).
Python datetime UTC should be acceptable.

This provides the following functions:
ECITime2LatLonElev - Takes given (x,y,z) tuple and a reference time, and
  converts to latitude, longitude, and height over reference sphere.
LatLonElevTime2ECI - Takes latitude, longitude, and elevation over reference
  sphere, and converts it to (x,y,z) in inertial space.

"""

import datetime
import unittest
from math import *
import random

earth_radius = 6371000.0 # Radius of the earth, m
earth_mass = 5.972E24 # mass of the earth, kg
earth_density = earth_mass / ((4*pi/3)*(earth_radius **3))
G = 6.67384e-11 # m^3 / kg-s^2, universal gravitational constant

# The J2000 epoch is defined in terms of Terrestrial Time.  Using UTC, this is
# the equivalent time, only off by a minute or so due to leap seconds over the
# years.  This value comes from the Wikipedia article on Epoch_(astronomy).
epoch_j2000 = datetime.datetime.strptime("1 Jan 2000 11:58:55.816",
                                         "%d %b %Y %H:%M:%S.%f")
# A sidereal day is 23:56:04.09054 of mean solar time
sidereal_day = datetime.timedelta(0,4,540,90,56,23)
omega_earth = 7.2921150e-5 # rad/s, Earth's inertial rotation, per wikipedia

def LatLonElevTime2ECI(lat, lon, elev, time, epoch=epoch_j2000):
	"""Converts a given latitude, longitude, elevation (above MSL) and time to
	earth-centered inertial frame with the given epoch (defaults to J2000).
	
	NOTE: This currently does not account for the non-spherical nature of the Earth,
	since the approximation is good enough for the moment. LS-IAS-06 documents the
	conversion with an ellipsoid"""
	delta_t = time - epoch
	#print("Time from epoch is %s" % delta_t)
	
	z = (earth_radius + elev) * sin(radians(lat))
	x = (earth_radius + elev) * cos(radians(lat)) * cos(radians(lon) - omega_earth * delta_t.total_seconds())
	y = (earth_radius + elev) * cos(radians(lat)) * sin(radians(lon) - omega_earth * delta_t.total_seconds())
	
	return (x,y,z)

def ECITime2LatLonElev(eci, time, epoch=epoch_j2000):
	"""Takes an earth-centered inertia <x,y,z> and converts it to latitude and longitude
	at a given time."""

	delta_t = time - epoch

	(x,y,z) = eci
	r = sqrt(x**2 + y**2 + z**2)
	lat = degrees(asin(z/r))
	lon = fmod(degrees(atan2(y,x) + omega_earth * delta_t.total_seconds()), 360)
	lon_epoch = fmod(lon + omega_earth * delta_t.total_seconds(), 360)
	if lat > 90.0:
		lat -= 180.0
	if lon > 180.0:
		lon -= 360.0
	
	return (lat, lon, r - earth_radius)


_EPS = 1e-6
class TestEarthFrame(unittest.TestCase):
	def test_constants(self):
		g = G * earth_mass / (earth_radius)**2
		#print(g)
		self.assertTrue(abs(g - 9.80665) < 0.05)

	def test_LatLonElevTime2ECI(self):
		test_input = [( 0.0, 0.0, 0.0, epoch_j2000), # Off the coast of Africa
		              (90.0, 0.0, 0.0, epoch_j2000), # At the north pole
		              (90.0, 0.0, 0.0, datetime.datetime.utcnow()),# north pole, now
		              ( 0.0, 0.0,1000, epoch_j2000)] # nonzero elevation
		test_output = [(earth_radius, 0, 0),
		               (0, 0, earth_radius),
		               (0, 0, earth_radius),
		               (earth_radius + 1000, 0, 0)]
		for inp, outp in zip(test_input, test_output):
			calc = LatLonElevTime2ECI(*inp)
			self.assertTrue(abs(calc[0] - outp[0]) < _EPS)
			self.assertTrue(abs(calc[1] - outp[1]) < _EPS)
			self.assertTrue(abs(calc[2] - outp[2]) < _EPS)
			#print(calc, outp)

	def test_eci_latlon_loopback(self):
		for i in range(100):
			lat = random.uniform( -90.0,  90.0)
			lon = random.uniform(-180.0, 180.0)
			elev = random.uniform(-1000, 1e6)
			#time = epoch_j2000
			time = epoch_j2000 + datetime.timedelta(seconds=random.uniform(0.0, 1e10))

			xyz = LatLonElevTime2ECI(lat, lon, elev, time)
			new_lat, new_lon, new_elev = ECITime2LatLonElev(xyz, time)

			#print("\nTime %s" % time)
			#print("Lat %8f -> %8f" % (lat, new_lat))
			#print("Lon %8f -> %8f" % (lon, new_lon))
			#print("Elev %8f -> %8f" % (elev, new_elev))
			self.assertTrue(abs(lat - new_lat) < _EPS)
			self.assertTrue(abs(lon - new_lon) < _EPS)
			self.assertTrue(abs(elev - new_elev) < _EPS)

	def test_ecitime2latlon(self):
		test_input =  [((earth_radius, 0.0, 0.0), epoch_j2000), # Off the coast of Africa
		               ((0.0, 0.0, earth_radius), epoch_j2000)]
		test_output = [( 0.0, 0.0, 0.0),
		               (90.0, 0.0, 0.0)]

		for inp, outp in zip(test_input, test_output):
			calc = ECITime2LatLonElev(*inp)
			#print("\nInput: %s\nExpected: %s\nCalculated: %s" % (inp, outp, calc))
			self.assertTrue(abs(calc[0] - outp[0]) < _EPS)
			self.assertTrue(abs(calc[1] - outp[1]) < _EPS)
			self.assertTrue(abs(calc[2] - outp[2]) < _EPS)
