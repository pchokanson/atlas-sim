# AtlasSimData.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""Data transfer and interpretation classes."""

import datetime

class AtlasSimData(object):
	"""Container class for state data (from simulation or elsewhere)."""
	__slots__ = ("date", "t", "pxyz", "vxyz", "q", "wxyz", "costates")
	
	def __init__(self, date, t, pxyz, vxyz, q, wxyz, costates):
		self.date = date
		self.t = t
		self.pxyz = pxyz
		self.vxyz = vxyz
		self.q = q
		self.wxyz = wxyz
		self.costates = costates

AtlasSimDataZero = AtlasSimData(
	datetime.fromtimestamp(0),
	0,
	[0,0,0],
	[0,0,0],
	[1,0,0,0],
	[0,0,0],
	[])

class AtlasSimDataInterpreter(object):
	pass
