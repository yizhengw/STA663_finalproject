
# coding: utf-8

# In[39]:

#load library
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


# In[40]:

#test the K from the IBP function is non-negtive


# In[41]:

def IBP( N,alpha ):
    #Generate initial number of features from a Poisson
    n_init = np.random.poisson(alpha,1)
    Z = np.zeros(shape=(N,n_init))
    Z[0,:] = 1
    m = np.sum(Z,0)
    K = n_init
    for i in range(1,N):
        #Calculate probability of visiting past dishes
        prob = m/(i+1)
        index = np.greater(prob,np.random.rand(1,K))
        Z[i,:] = index.astype(int);
        #Calculate the number of new dishes visited by customer i
        knew = np.random.poisson(alpha/(i+1),1)
        Z=np.concatenate((Z,np.zeros(shape=(N,knew))), axis=1)
        Z[i,K:K+knew:1] = 1
        #Update matrix size and number of features 
        K = K+knew
        m = sum(Z,0)
    return Z, K


# In[42]:

N=100
alpha =1


# In[43]:

Z,testk=IBP(N,alpha)


# In[44]:

print("Is K non-negtive", testk>=0)


# In[45]:

#test regular calcM and optimize calcM


# In[46]:

get_ipython().magic('load_ext Cython')


# In[47]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef calcM_cython(double[:,:] Z,int Kplus,double sigmaX,double sigmaA): \n    cdef double [:,:] temp =np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])\n    cdef float temp2=(sigmaX/sigmaA)**2\n    cdef double [:,:] iden=(temp2)*np.identity(Kplus)\n\n    cdef double [:,:] resu = iden\n    for i in range(Kplus):\n        for j in range(Kplus):\n            resu[i,j]=iden[i,j]+temp[i,j]\n    \n    return np.linalg.inv(resu)')


# In[48]:

def calcM(Z,Kplus,sigmaX,sigmaA):
    return np.linalg.inv(np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])+((sigmaX/sigmaA)**2)*np.identity(Kplus))


# In[49]:

sX=1.7
sA=1.2
sX0=0.5


# In[50]:

testm=calcM(Z,testk,sX,sA)


# In[51]:

testm2=calcM_cython(Z,testk,sX,sA)


# In[52]:

np.testing.assert_almost_equal(testm,testm2)
print("pass")


# In[53]:

#test log_likelyhood


# In[54]:

def log_likelihood(X,Z,M,sigmaA,sigmaX,Kplus,N,D):  
    determinant = np.linalg.det(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))
    constant = N*D*0.5*np.log(2*np.pi) + (N-Kplus)*D*np.log(sigmaX) + Kplus*D*np.log(sigmaA) + D*0.5*np.log(determinant)
    middle = np.identity(N) - np.dot(np.dot(Z, M),Z.T)
    trace = np.trace(np.dot(np.dot(X.T,middle),X))
    kernel = -0.5*np.reciprocal(sigmaX**2)*trace
    log_lik = -constant + kernel
    return log_lik


# In[55]:

def samplex(objects,sX0,dimension,Z0):
    X = np.zeros((objects,dimension))
    for i in range(objects):
        while True:
            pp=np.round(stats.uniform.rvs(loc=0,scale=1,size=feature)).astype(int)
            if np.sum(pp)!=0:
                Z0[i,:] =pp
                break
        X[i,:] =  np.dot(Z0[i,:],A)+np.random.normal(size=dimension)*sX0
    return X


# In[56]:

dimension = 36
objects = 100
feature = 4
paperImagesetting=6
iteration = 200


# In[57]:

b_1 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,1,0,0,0,0],[1,1,1,0,0,0],[0,1,0,0,0,0]])
b_2 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,1,1,1],[0,0,0,1,0,1],[0,0,0,1,1,1]])
b_3 = np.array([[1,1,1,0,0,0],[1,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
b_4 = np.array([[0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
basis_4=np.array([b_1,b_2,b_3,b_4])


# In[58]:

A = np.array([b_1.reshape(dimension),b_2.reshape(dimension),b_3.reshape(dimension),b_4.reshape(dimension)])


# In[ ]:




# In[59]:

Z0= np.zeros((N,feature))
X=samplex(objects,sX0,dimension,Z0)


# In[60]:

plog=log_likelihood(X,Z,testm,sA,sX,testk,N,dimension)


# In[62]:

test=np.exp(plog)


# In[63]:

assert 0<=test and test<=1
print("pass")


# In[ ]:




# In[ ]:



