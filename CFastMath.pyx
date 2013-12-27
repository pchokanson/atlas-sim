#!python
# cython: profile=True
# 
# Fast Cython implementations of vector math operations
# - multiply two quaternions
#
# Buid with the following:
# $ cython3 -a CFastMath.pyx && gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python3.3/ -o CFastMath.so CFastMath.c -march=native

import numpy as np
cimport numpy as np

import cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def mult_q_q(np.ndarray[DTYPE_t, ndim=1] q1 not None, 
						 np.ndarray[DTYPE_t, ndim=1] q2 not None):
	"""Quaternion-multiply (Grassman product) two 4-vectors."""
	#assert q1.dtype = DTYPE and q2.dtype = DTYPE
	cdef np.ndarray[DTYPE_t, ndim=1] x = np.zeros([4], dtype=DTYPE)
	x[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
	x[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
	x[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
	x[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
	
	return x
