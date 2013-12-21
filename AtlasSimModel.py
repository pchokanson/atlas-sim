# RigidBody.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""MVC model interfaces for Atlas-sim."""

import queue
import sched, time

from RigidBody import RigidBody
from AtlasSimView import AtlasSimViewLogging
from AtlasSimData import AtlasSimData, AtlasSimDataZero

class IAtlasSimModel(object):
	"""Generic AtlasSim model interface."""
	def __init__(self, realtime=False):
		self.views = []
	
	def attach_view(self, view):
		self.views.append(view)
		
	def getSimData(self):
		return AtlasSimDataZero
		
	def update_views(self, data):
		for view in self.views:
			view.update(data)


class AtlasSimModelSimulation(IAtlasSimModel):
	
	def __init__(self, realtime=False):
		IAtlasSimModel.__init__(self, realtime)
		self.sim_model = RigidBody()
		self.stepsize = 0.1 # s
	
	def getSimData(self):
		self.sim_model.step(self.stepsize)
		sv = self.sim_model.state_vector
		date = self.sim_model.getDatetime()
		t = self.sim_model.t
		pxyz = sv[0:3]
		vxyz = sv[3:6]
		q = sv[6:10]
		wxyz = sv[10:12]
		costates = sv[13:]
		return AtlasSimData(date, t, pxyz, vxyz, q, wxyz, costates)
	
	def run(self):
		s = sched.scheduler(time.time, time.sleep)
		for i in range(100):
			d = self.getSimData()
			self.update_views(d)

class AtlasSimModelPlayback(IAtlasSimModel):
	"""Reads a simulation data file for testing"""
	def __init__(self, realtime=False):
		IAtlasSimModel.__init__(self, realtime)

if __name__ == "__main__":
	m = AtlasSimModelSimulation()
	v = AtlasSimViewLogging()
	m.attach_view(v)
	m.run()
	
