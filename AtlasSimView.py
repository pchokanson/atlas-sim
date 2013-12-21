# AtlasSimView.py
# Vertical Limit Labs
# Copyright (c) 2013 Peter Hokanson
"""MVC view interfaces for Atlas-sim."""

import queue
import sched, time
import sys
import json
import pickle

from AtlasSimData import AtlasSimData

class IAtlasSimView(object):
	"""Generic AtlasSim model interface."""
	def __init__(self):
		pass
	
	def update(self, data):
		pass
	
class AtlasSimViewLogging(IAtlasSimView):
	def __init__(self):
		IAtlasSimView.__init__(self)
		self.logfile = sys.stdout
		
		
	def setFile(self, logfile):
		self.logfile = logfile
	
	def update(self, data):
		"""Update this view with new data -- dump it to a file"""
		d = {}
		for slot in AtlasSimData.__slots__:
			if slot == "date":
				d["date"] = data.date.isoformat()
			elif slot == "t":
				d["t"] = data.t
			else:
				d[slot] = list(getattr(data, slot))

		json.dump(d, self.logfile)
		print("\n")

if __name__ == "__main__":
	pass
