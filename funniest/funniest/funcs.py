
# coding: utf-8

# In[ ]:

import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
from fractions import Fraction
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import toimage
from numba import jit
import timeit
from numba import vectorize, int64, float64,guvectorize


# In[ ]:

get_ipython().magic('load_ext Cython')


# In[ ]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef calcM_cython(double[:,:] Z,int Kplus,double sigmaX,double sigmaA): \n    cdef double [:,:] temp =np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])\n    cdef float temp2=(sigmaX/sigmaA)**2\n    cdef double [:,:] iden=(temp2)*np.identity(Kplus)\n\n    cdef double [:,:] resu = iden\n    for i in range(Kplus):\n        for j in range(Kplus):\n            resu[i,j]=iden[i,j]+temp[i,j]\n    \n    return np.linalg.inv(resu)')


# In[ ]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef calcM2_cython(double[:,:] Z,int Kplus,int kterm,double sigmaX,double sigmaA): \n    cdef double [:,:] temp =np.dot(Z[:,0:Kplus+kterm].T,Z[:,0:Kplus+kterm])\n    cdef float temp2=(sigmaX/sigmaA)**2\n    cdef double [:,:] iden=(temp2)*np.identity(Kplus+kterm)\n\n    cdef double [:,:] resu = iden\n    for i in range(Kplus+kterm):\n        for j in range(Kplus+kterm):\n            resu[i,j]=iden[i,j]+temp[i,j]\n    \n    return np.linalg.inv(resu)')


# In[ ]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef log_likelihood_cython(double[:,:] X,double[:,:] Z, double[:,:] M,double sigmaA,double sigmaX,int Kplus,int N,int D):  \n    cdef double determinant = np.linalg.det(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))\n    cdef double constant = N*D*0.5*np.log(2*np.pi) + (N-Kplus)*D*np.log(sigmaX) + Kplus*D*np.log(sigmaA) + D*0.5*np.log(determinant)\n    \n    cdef double[:,:] middle = np.identity(N) - np.dot(np.dot(Z, M),Z.T)\n    cdef double trace = np.trace(np.dot(np.dot(X.T,middle),X))\n    cdef double kernel = -0.5*np.reciprocal(sigmaX**2)*trace\n    \n    cdef double result = -constant + kernel\n    return  result')

