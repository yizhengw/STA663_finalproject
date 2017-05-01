
# coding: utf-8

# In[25]:

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


# In[26]:

def IBP( N,alpha ):
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


# In[9]:

def simulated_image(objects,basis_4,coef,noise,paperImagesetting,feature):
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
    SimulatedImage = np.zeros((objects,paperImagesetting,paperImagesetting))
    for i in range(objects):
        for j in range(feature):
            SimulatedImage[i,:,:]+=coef[i,j]*basis_4[j,:,:]
        SimulatedImage[i,:,:]+= noise[i,:,:]
    return SimulatedImage


# In[10]:

def samplex(objects,sX0,dimension,Z0,feature,A):
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
    X = np.zeros((objects,dimension))
    for i in range(objects):
        while True:
            pp=np.round(stats.uniform.rvs(loc=0,scale=1,size=feature)).astype(int)
            if np.sum(pp)!=0:
                Z0[i,:] =pp
                break
        X[i,:] =  np.dot(Z0[i,:],A)+np.random.normal(size=dimension)*sX0
    return X


# In[11]:

def calcM(Z,Kplus,sigmaX,sigmaA):
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
    return np.linalg.inv(np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])+((sigmaX/sigmaA)**2)*np.identity(Kplus))
def calcM2(Z,Kplus,kterm,sigmaX,sigmaA):
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
    return np.linalg.inv(np.dot(Z[:,0:Kplus+kterm].T,Z[:,0:Kplus+kterm])+((sigmaX/sigmaA)**2)*np.identity(Kplus+kterm))
def log_likelihood(X,Z,M,sigmaA,sigmaX,Kplus,N,D):  
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
    determinant = np.linalg.det(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))
    constant = N*D*0.5*np.log(2*np.pi) + (N-Kplus)*D*np.log(sigmaX) + Kplus*D*np.log(sigmaA) + D*0.5*np.log(determinant)
    middle = np.identity(N) - np.dot(np.dot(Z, M),Z.T)
    trace = np.trace(np.dot(np.dot(X.T,middle),X))
    kernel = -0.5*np.reciprocal(sigmaX**2)*trace
    log_lik = -constant + kernel
    return log_lik


# In[12]:

@jit
def calcM_jit(Z,Kplus,sigmaX,sigmaA):
    return np.linalg.inv(np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])+((sigmaX/sigmaA)**2)*np.identity(Kplus))


# In[13]:

@jit
def calcM2_jit(Z,Kplus,kterm,sigmaX,sigmaA):
    return np.linalg.inv(np.dot(Z[:,0:Kplus+kterm].T,Z[:,0:Kplus+kterm])+((sigmaX/sigmaA)**2)*np.identity(Kplus+kterm))


# In[14]:

@jit
def log_likelihood_jit(X,Z,M,sigmaA,sigmaX,Kplus,N,D):  
    determinant = np.linalg.det(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))
    constant = N*D*0.5*np.log(2*np.pi) + (N-Kplus)*D*np.log(sigmaX) + Kplus*D*np.log(sigmaA) + D*0.5*np.log(determinant)
    
    middle = np.identity(N) - np.dot(np.dot(Z, M),Z.T)
    trace = np.trace(np.dot(np.dot(X.T,middle),X))
    kernel = -0.5*np.reciprocal(sigmaX**2)*trace
    
    log_lik = -constant + kernel
    return log_lik


# In[15]:

get_ipython().magic('load_ext Cython')


# In[16]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef calcM_cython(double[:,:] Z,int Kplus,double sigmaX,double sigmaA): \n    cdef double [:,:] temp =np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])\n    cdef float temp2=(sigmaX/sigmaA)**2\n    cdef double [:,:] iden=(temp2)*np.identity(Kplus)\n\n    cdef double [:,:] resu = iden\n    for i in range(Kplus):\n        for j in range(Kplus):\n            resu[i,j]=iden[i,j]+temp[i,j]\n    \n    return np.linalg.inv(resu)')


# In[17]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef calcM2_cython(double[:,:] Z,int Kplus,int kterm,double sigmaX,double sigmaA): \n    cdef double [:,:] temp =np.dot(Z[:,0:Kplus+kterm].T,Z[:,0:Kplus+kterm])\n    cdef float temp2=(sigmaX/sigmaA)**2\n    cdef double [:,:] iden=(temp2)*np.identity(Kplus+kterm)\n\n    cdef double [:,:] resu = iden\n    for i in range(Kplus+kterm):\n        for j in range(Kplus+kterm):\n            resu[i,j]=iden[i,j]+temp[i,j]\n    \n    return np.linalg.inv(resu)')


# In[18]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef log_likelihood_cython(double[:,:] X,double[:,:] Z, double[:,:] M,double sigmaA,double sigmaX,int Kplus,int N,int D):  \n    cdef double determinant = np.linalg.det(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))\n    cdef double constant = N*D*0.5*np.log(2*np.pi) + (N-Kplus)*D*np.log(sigmaX) + Kplus*D*np.log(sigmaA) + D*0.5*np.log(determinant)\n    \n    cdef double[:,:] middle = np.identity(N) - np.dot(np.dot(Z, M),Z.T)\n    cdef double trace = np.trace(np.dot(np.dot(X.T,middle),X))\n    cdef double kernel = -0.5*np.reciprocal(sigmaX**2)*trace\n    \n    cdef double result = -constant + kernel\n    return  result\n    ')


# In[ ]:




# In[19]:

def mainsolver_improve(iteration,Z,alpha,K,sX,sA,objects,X,dimension,feature,HN,ratioX = 0,ratioA = 0): 
    Z_mcmc= np.zeros((iteration,objects,1000))
    K_mcmc = np.zeros(iteration)
    sX_mcmc = np.zeros(iteration)
    sA_mcmc = np.zeros(iteration)
    alpha_mcmc = np.zeros(iteration)
    for it in range(iteration):
        sX_mcmc[it] = sX
        sA_mcmc[it] = sA
        alpha_mcmc[it] = alpha
        Z_mcmc[it,:,0:K] = Z[:,0:K]
        K_mcmc[it] = K
        for i in range(objects):
            for k in range(K):
                if k+1>K:
                    break
                if Z[i,k]>0:
                    if np.sum(Z[:,k])-Z[i,k]<=0:
                        Z[:,k:(K-1)] = Z[:,(k+1):K]
                        K -= 1
                        continue
                Z[i,k] = 1
                M1 = calcM_jit(Z,K,sX,sA) 
                posterior = np.zeros(2)
                posterior[1] = np.log(sum(Z[:,k])-Z[i,k]) - np.log(objects)+log_likelihood_jit(X,Z[:,0:K],M1,sA,sX,K,objects,dimension) 
                Z[i,k] = 0
                M0 = calcM_jit(Z,K,sX,sA) 
                posterior[0] = np.log(objects-sum(Z[:,k])) - np.log(objects)+log_likelihood_jit(X,Z[:,0:K],M0,sA,sX,K,objects,dimension) 
                posterior = np.exp(posterior - max(posterior))
                p = stats.uniform.rvs(0,1,1)           
                if p < posterior[0]/(posterior[0]+posterior[1]):
                    Z[i,k] = 0
                else:
                    Z[i,k] = 1
            
            sele=np.zeros(feature)
            alphan=alpha/objects
            for k_i in range(4):
                if k_i > 0:
                    temp = np.zeros((objects,k_i))
                    temp[i,:] = 1
                    Z = np.hstack((Z[:,0:K],temp))
                M = calcM2_jit(Z,K,k_i,sX,sA)
                sele[k_i] = log_likelihood_jit(X,Z[:,0:(K+k_i)],M,sA,sX,K+k_i,objects,dimension) +(k_i)*np.log(alphan) - alphan - np.log(np.math.factorial(k_i)) 
            Z[i,K:(K+3)] = 0
            sele = np.exp(sele-max(sele))
            sele=sele/np.sum(sele) 
            t = 0
            for ki in range(4):
                t += sele[ki]
                if p < t:
                    ff1 = ki
                    break
            Z[i,K:(K+ff1)] = 1
            K += ff1
        M=calcM_jit(Z,K,sX,sA)
        l1=log_likelihood_jit(X, Z[:,0:K], M, sA, sX, K, objects, dimension)
        if p<0.5:
            pppsX = sX - p/20
        else:
            pppsX = sX + p/20
        M_X = calcM_jit(Z, K, pppsX, sA)
        l2=log_likelihood_jit(X, Z[:,0:K], M_X, sA, pppsX, K, objects, dimension)
        acc_X = np.exp(min(0, l2-l1))
        if p<0.5:
            pppsA = sA - p/20
        else:
            pppsA = sA + p/20
        M_A = calcM_jit(Z, K, sX, pppsA)
        l3=log_likelihood_jit(X, Z[:,0:K], M_A, pppsA, sX, K, objects, dimension)
        acc_A = np.exp(min(0, l3-l1)) 
        if p < acc_X:
            sX = pppsX
            ratioX += 1
        if p < acc_A:
            sA = pppsA
            ratioA += 1
        alpha = stats.gamma.rvs(1+K,  0,  np.reciprocal(1+HN),1)[0]   
    return Z_mcmc, K_mcmc,sX_mcmc,sA_mcmc,alpha_mcmc


# In[20]:

def mainsolver_improve_cython(iteration,Z,alpha,K,sX,sA,objects,X,dimension,feature,HN,ratioX = 0,ratioA = 0): 
    Z_mcmc= np.zeros((iteration,objects,1000))
    K_mcmc = np.zeros(iteration)
    sX_mcmc = np.zeros(iteration)
    sA_mcmc = np.zeros(iteration)
    alpha_mcmc = np.zeros(iteration)
    for it in range(iteration):
        print("we are at ",it,"th iteration")
        sX_mcmc[it] = sX
        sA_mcmc[it] = sA
        alpha_mcmc[it] = alpha
        Z_mcmc[it,:,0:K] = Z[:,0:K]
        K_mcmc[it] = K

        for i in range(objects):
            for k in range(K):
                if k+1>K:
                    break
                if Z[i,k]>0:
                    if np.sum(Z[:,k])-Z[i,k]<=0:
                        Z[:,k:(K-1)] = Z[:,(k+1):K]
                        K -= 1
                        continue
                
                Z[i,k] = 1
                M1 = calcM_cython(Z,K,sX,sA)
                posterior = np.zeros(2)
                posterior[1] = np.log(sum(Z[:,k])-Z[i,k]) - np.log(objects)+log_likelihood_cython(X,Z[:,0:K],M1,sA,sX,K,objects,dimension) 
                Z[i,k] = 0
                M0 = calcM_cython(Z,K,sX,sA) 
                posterior[0] = np.log(objects-sum(Z[:,k])) - np.log(objects)+log_likelihood_cython(X,Z[:,0:K],M0,sA,sX,K,objects,dimension) 
                posterior = np.exp(posterior - max(posterior))
                p = stats.uniform.rvs(0,1,1)           
                if p < posterior[0]/(posterior[0]+posterior[1]):
                    Z[i,k] = 0
                else:
                    Z[i,k] = 1
            sele=np.zeros(feature)
            alphan=alpha/objects
            for k_i in range(4):
                if k_i > 0:
                    temp = np.zeros((objects,k_i))
                    temp[i,:] = 1
                    Z = np.hstack((Z[:,0:K],temp))
                M = calcM2_cython(Z,K,k_i,sX,sA)
                sele[k_i] = log_likelihood_cython(X,Z[:,0:(K+k_i)],M,sA,sX,K+k_i,objects,dimension)+(k_i)*np.log(alphan) - alphan - np.log(np.math.factorial(k_i)) 
            Z[i,K:(K+3)] = 0
            sele = np.exp(sele-max(sele))
            sele = sele/np.sum(sele) 
            t = 0
            for ki in range(4):
                t += sele[ki]
                if p < t:
                    ff1 = ki
                    break
            Z[i,K:(K+ff1)] = 1
            K += ff1
        M=calcM_cython(Z,K,sX,sA)
        l1=log_likelihood_cython(X, Z[:,0:K], M, sA, sX, K, objects, dimension)
        if p<0.5:
            pppsX = sX - p/20
        else:
            pppsX = sX + p/20
        M_X = calcM_cython(Z, K, pppsX, sA)
        l2=log_likelihood_cython(X, Z[:,0:K], M_X, sA, pppsX, K, objects, dimension)
        acc_X = np.exp(min(0, l2-l1))
        if p<0.5:
            pppsA = sA - p/20
        else:
            pppsA = sA + p/20
        M_A = calcM_cython(Z, K, sX, pppsA)
        l3=log_likelihood_cython(X, Z[:,0:K], M_A, pppsA, sX, K, objects, dimension)
        acc_A = np.exp(min(0, l3-l1)) 
        if p < acc_X:
            sX = pppsX
            ratioX += 1
        if p < acc_A:
            sA = pppsA
            ratioA += 1
        alpha = stats.gamma.rvs(1+K,  0,np.reciprocal(1+HN),1)[0]   
    return Z_mcmc, K_mcmc,sX_mcmc,sA_mcmc,alpha_mcmc


# In[21]:

def mainsolver(iteration,Z,alpha,K,sX,sA,objects,X,dimension,feature,HN,ratioX = 0,ratioA = 0):
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
    #initialize the mcmc varibles
    Z_mcmc= np.zeros((iteration,objects,1000))
    K_mcmc = np.zeros(iteration)
    sX_mcmc = np.zeros(iteration)
    sA_mcmc = np.zeros(iteration)
    alpha_mcmc = np.zeros(iteration)
    #start point of mcmc
    for it in range(iteration):
        #initial point
        print("we are at ",it,"th iteration")
        Z_mcmc[it,:,0:K] = Z[:,0:K]
        alpha_mcmc[it] = alpha
        K_mcmc[it] = K
        sX_mcmc[it] = sX
        sA_mcmc[it] = sA
        for i in range(objects):
            for k in range(K):
                #just in case when K over-decreased
                if k+1>K:
                    break
                if Z[i,k]>0:
                    #sigular dishes
                    if np.sum(Z[:,k])-Z[i,k]<=0:
                        Z[:,k:(K-1)] = Z[:,(k+1):K]
                        K -= 1
                        continue
                
                Z[i,k] = 1
                #based on the paper to calculate M equation 51-54
                M1 = calcM(Z,K,sX,sA) 
                posterior = np.zeros(2)
                #posterior
                posterior[1] = np.log(sum(Z[:,k])-Z[i,k])- np.log(objects)+ log_likelihood(X,Z[:,0:K],M1,sA,sX,K,objects,dimension)  
                Z[i,k] = 0
                M0 = calcM(Z,K,sX,sA) 
                posterior[0] = np.log(objects-sum(Z[:,k])) - np.log(objects) + log_likelihood(X,Z[:,0:K],M0,sA,sX,K,objects,dimension) 
                posterior = np.exp(posterior - max(posterior))
                p = stats.uniform.rvs(0,1,1)  
                #MH steps
                if p < posterior[0]/(posterior[0]+posterior[1]):
                    Z[i,k] = 0
                else:
                    Z[i,k] = 1
            #sample new dish for current customer
            sele=np.zeros(feature)
            alphan=alpha/objects
            for k_i in range(4):
                if k_i > 0:
                    temp = np.zeros((objects,k_i))
                    temp[i,:] = 1
                    Z = np.hstack((Z[:,0:K],temp))
                M = calcM2(Z,K,k_i,sX,sA)
                sele[k_i] = log_likelihood(X,Z[:,0:(K+k_i)],M,sA,sX,K+k_i,objects,dimension)+(k_i)*np.log(alphan) - alphan - np.log(np.math.factorial(k_i)) 
            Z[i,K:(K+3)] = 0
            sele = np.exp(sele-max(sele))
            sele=sele/np.sum(sele) 
            t = 0
            for ki in range(4):
                t += sele[ki]
                if p < t:
                    ff1 = ki
                    break
            Z[i,K:(K+ff1)] = 1
            K += ff1
        #MH steps
        M=calcM(Z,K,sX,sA)
        l1=log_likelihood(X, Z[:,0:K], M, sA, sX, K, objects, dimension)
        if p<0.5:
            pppsX = sX - p/20
        else:
            pppsX = sX + p/20
        M_X = calcM(Z, K, pppsX, sA)
        l2=log_likelihood(X, Z[:,0:K], M_X, sA, pppsX, K, objects, dimension)
        acc_X = np.exp(min(0, l2-l1))
        if p<0.5:
            pppsA = sA - p/20
        else:
            pppsA = sA + p/20
        M_A = calcM(Z, K, sX, pppsA)
        l3=log_likelihood(X, Z[:,0:K], M_A, pppsA, sX, K, objects, dimension)
        acc_A = np.exp(min(0, l3-l1)) 
        if p < acc_X:
            sX = pppsX
            ratioX += 1
        if p < acc_A:
            sA = pppsA
            ratioA += 1
        alpha = stats.gamma.rvs(a = 1+K, loc = 0, scale = np.reciprocal(1+HN),size=1)[0]   
    return Z_mcmc, K_mcmc,sX_mcmc,sA_mcmc,alpha_mcmc


# In[23]:

def main(iteration=200,method="regular"):
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
    dimension = 36
    objects = 100
    feature = 4
    paperImagesetting=6
    b_1 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,1,0,0,0,0],[1,1,1,0,0,0],[0,1,0,0,0,0]])
    b_2 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,1,1,1],[0,0,0,1,0,1],[0,0,0,1,1,1]])
    b_3 = np.array([[1,1,1,0,0,0],[1,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    b_4 = np.array([[0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    basis_4=np.array([b_1,b_2,b_3,b_4])
    fig = plt.figure(figsize=(12,3))
    fig1 = fig.add_subplot(141)
    fig1.pcolormesh(b_1,cmap=plt.cm.gray)     
    fig2 = fig.add_subplot(142)
    fig2.pcolormesh(b_2,cmap=plt.cm.gray)  
    fig3 = fig.add_subplot(143)
    fig3.pcolormesh(b_3,cmap=plt.cm.gray)  
    fig4 = fig.add_subplot(144)
    fig4.pcolormesh(b_4,cmap=plt.cm.gray) 
    plt.show()
    A = np.array([b_1.reshape(dimension),b_2.reshape(dimension),b_3.reshape(dimension),b_4.reshape(dimension)])
    SimulatedImage = np.zeros((objects,paperImagesetting,paperImagesetting)) 
    coef = stats.bernoulli.rvs(0.5,size=(objects,feature))
    noise = stats.norm.rvs(loc=0,scale=0.5,size = (objects,paperImagesetting,paperImagesetting)) 
    SimulatedImage=simulated_image(objects,basis_4,coef,noise,paperImagesetting,feature)
    fig=plt.figure(figsize=(12,3))
    fig1 = fig.add_subplot(141)
    fig1.pcolormesh(SimulatedImage[2],cmap=plt.cm.gray)
    fig1 = fig.add_subplot(142)
    fig1.pcolormesh(SimulatedImage[10],cmap=plt.cm.gray)
    fig1 = fig.add_subplot(143)
    fig1.pcolormesh(SimulatedImage[30],cmap=plt.cm.gray)
    fig1 = fig.add_subplot(144)
    fig1.pcolormesh(SimulatedImage[80],cmap=plt.cm.gray)
    plt.show()
    sA = 1.2
    sX = 1.7
    alpha=1
    Z,K=IBP(objects,alpha)
    Z_mcmc= np.zeros((iteration,objects,2000))
    K_mcmc = np.zeros(iteration)
    sX_mcmc = np.zeros(iteration)
    sA_mcmc = np.zeros(iteration)
    alpha_mcmc = np.zeros(iteration)
    ratioX = 0
    ratioA = 0
    X = np.zeros((objects,dimension))
    Z0= np.zeros((objects,feature))
    HN = 0
    for i in range(objects):
        HN+=1/(i+1)
    sX0=0.5
    X=samplex(objects,sX0,dimension,Z0,feature,A)
    fig=plt.figure(figsize=(12,3))
    fig1 = fig.add_subplot(141)
    fig1.pcolormesh(X[2].reshape(6,6),cmap=plt.cm.gray)
    fig1 = fig.add_subplot(142)
    fig1.pcolormesh(X[10].reshape(6,6),cmap=plt.cm.gray)
    fig1 = fig.add_subplot(143)
    fig1.pcolormesh(X[20].reshape(6,6),cmap=plt.cm.gray)
    fig1 = fig.add_subplot(144)
    fig1.pcolormesh(X[50].reshape(6,6),cmap=plt.cm.gray)
    plt.show()
    if method=="vectorize":
        Z_mcmc, K_mcmc,sX_mcmc,sA_mcmc,alpha_mcmc=mainsolver(iteration,Z,alpha,K,sX,sA,objects,X,dimension,feature,HN)
    if method=="jit":
        Z_mcmc, K_mcmc,sX_mcmc,sA_mcmc,alpha_mcmc=mainsolver_improve(iteration,Z,alpha,K,sX,sA,objects,X,dimension,feature,HN)
    #if method=="cython":
        #Z_mcmc, K_mcmc,sX_mcmc,sA_mcmc,alpha_mcmc=mainsolver_improve_cython(iteration,Z,alpha,K,sX,sA,objects,X,dimension,feature,HN)
    #elif :
       # Z_mcmc, K_mcmc,sX_mcmc,sA_mcmc,alpha_mcmc=mainsolver(iteration,Z,alpha,K,sX,sA,objects,X,dimension,feature,HN)
    fig = plt.figure(figsize=(8,8))
    fig1 = fig.add_subplot(411)
    fig1.plot(K_mcmc)
    fig1.set_ylabel('K')
    fig2 = fig.add_subplot(412)
    fig2.plot(alpha_mcmc)
    fig2.set_ylabel('alpha')
    fig3 = fig.add_subplot(413)
    fig3.plot(sX_mcmc)
    fig3.set_ylabel('sX')
    fig4 = fig.add_subplot(414)
    fig4.plot(sA_mcmc)
    fig4.set_ylabel('sA')
    fig4.set_xlabel('Iteration')
    fig.savefig('mcmc_varible_plot.png')
    plt.show()
    samplen=iteration-1
    Kf = K_mcmc[samplen]
    Zf = Z_mcmc[samplen,:,0:Kf].reshape(objects,Kf)
    sXf = sX_mcmc[samplen]
    sAf = sA_mcmc[samplen]
    A_inf = np.dot(np.linalg.inv(np.dot(Zf.T,Zf) +  ((sXf/sAf)**2)*np.identity(Kf)),np.dot(Zf.T,X))
    fig = plt.figure(figsize=(12,3))
    fig1 = fig.add_subplot(141)
    fig1.pcolormesh(A_inf[2,:].reshape(6,6),cmap=plt.cm.gray)
    fig2 = fig.add_subplot(142)
    fig2.pcolormesh(A_inf[1,:].reshape(6,6),cmap=plt.cm.gray)
    fig3 = fig.add_subplot(143)
    fig3.pcolormesh(A_inf[3,:].reshape(6,6),cmap=plt.cm.gray)
    fig4 = fig.add_subplot(144)
    fig4.pcolormesh(A_inf[0,:].reshape(6,6),cmap=plt.cm.gray)
    plt.show()


# In[24]:




# In[ ]:




# In[ ]:



