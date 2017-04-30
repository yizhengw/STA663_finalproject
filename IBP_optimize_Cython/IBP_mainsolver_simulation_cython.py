
# coding: utf-8

# In[ ]:

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


# In[ ]:

#this function is the main solver for the indian buffet
#input: N is the number of the objects, and alpha is any integer
#output: Z is the binary matrix, K is the number of features (prior)
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


# In[ ]:

#initialize varible values
dimension = 36
objects = 100
feature = 4
paperImagesetting=6
iteration = 200


# In[ ]:

#basis setting as paper
b_1 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,1,0,0,0,0],[1,1,1,0,0,0],[0,1,0,0,0,0]])
b_2 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,1,1,1],[0,0,0,1,0,1],[0,0,0,1,1,1]])
b_3 = np.array([[1,1,1,0,0,0],[1,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
b_4 = np.array([[0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
basis_4=np.array([b_1,b_2,b_3,b_4])


# In[ ]:

#plot the basis
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
fig.savefig('original_cython_plots.png')


# In[ ]:

A = np.array([b_1.reshape(dimension),b_2.reshape(dimension),b_3.reshape(dimension),b_4.reshape(dimension)])


# In[ ]:

#Simulate figure
SimulatedImage = np.zeros((objects,paperImagesetting,paperImagesetting)) 
coef = stats.bernoulli.rvs(0.5,size=(objects,feature))
noise = stats.norm.rvs(loc=0,scale=0.5,size = (objects,paperImagesetting,paperImagesetting)) 


# In[ ]:

#this function is used to generate simulate figure based on the basis and linear model
def simulated_image(objects,basis_4,coef,noise,paperImagesetting,feature):
    SimulatedImage = np.zeros((objects,paperImagesetting,paperImagesetting))
    for i in range(objects):
        for j in range(feature):
            SimulatedImage[i,:,:]+=coef[i,j]*basis_4[j,:,:]
        SimulatedImage[i,:,:]+= noise[i,:,:]
    return SimulatedImage


# In[ ]:

SimulatedImage=simulated_image(objects,basis_4,coef,noise,paperImagesetting,feature)


# In[ ]:

#plot simulate plots
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
fig.savefig('simulation_cython.png')


# In[ ]:

sA = 1.2
sX = 1.7
alpha=1


# In[ ]:

#generate binary matrix and features based on the indian buffet process
while True:
    Z,K=IBP(objects,alpha)
    if K!=0:
        break


# In[ ]:

Z_mcmc= np.zeros((iteration,objects,2000))
K_mcmc = np.zeros(iteration)
sX_mcmc = np.zeros(iteration)
sA_mcmc = np.zeros(iteration)
alpha_mcmc = np.zeros(iteration)
ratioX = 0
ratioA = 0
X = np.zeros((objects,dimension))
Z0= np.zeros((objects,feature))


# In[ ]:

HN = 0
for i in range(objects):
    HN+=1/(i+1)


# In[ ]:

sX0=0.5


# In[ ]:

def samplex(objects,sX0,dimension,Z0,feature):
    X = np.zeros((objects,dimension))
    for i in range(objects):
        while True:
            pp=np.round(stats.uniform.rvs(loc=0,scale=1,size=feature)).astype(int)
            if np.sum(pp)!=0:
                Z0[i,:] =pp
                break
        X[i,:] =  np.dot(Z0[i,:],A)+np.random.normal(size=dimension)*sX0
    return X


# In[ ]:

X=samplex(objects,sX0,dimension,Z0,feature)


# In[ ]:

fig=plt.figure(figsize=(12,3))
fig1 = fig.add_subplot(141)
fig1.pcolormesh(X[2].reshape(6,6),cmap=plt.cm.gray)
fig1 = fig.add_subplot(142)
fig1.pcolormesh(X[10].reshape(6,6),cmap=plt.cm.gray)
fig1 = fig.add_subplot(143)
fig1.pcolormesh(X[20].reshape(6,6),cmap=plt.cm.gray)
fig1 = fig.add_subplot(144)
fig1.pcolormesh(X[50].reshape(6,6),cmap=plt.cm.gray)
fig.savefig('Ximages_simulation_cython.png')


# In[ ]:

get_ipython().magic('load_ext Cython')


# In[ ]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef calcM_cython(double[:,:] Z,int Kplus,double sigmaX,double sigmaA): \n    cdef double [:,:] temp =np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])\n    cdef float temp2=(sigmaX/sigmaA)**2\n    cdef double [:,:] iden=(temp2)*np.identity(Kplus)\n\n    cdef double [:,:] resu = iden\n    for i in range(Kplus):\n        for j in range(Kplus):\n            resu[i,j]=iden[i,j]+temp[i,j]\n    \n    return np.linalg.inv(resu)')


# In[ ]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef calcM2_cython(double[:,:] Z,int Kplus,int kterm,double sigmaX,double sigmaA): \n    cdef double [:,:] temp =np.dot(Z[:,0:Kplus+kterm].T,Z[:,0:Kplus+kterm])\n    cdef float temp2=(sigmaX/sigmaA)**2\n    cdef double [:,:] iden=(temp2)*np.identity(Kplus+kterm)\n\n    cdef double [:,:] resu = iden\n    for i in range(Kplus+kterm):\n        for j in range(Kplus+kterm):\n            resu[i,j]=iden[i,j]+temp[i,j]\n    \n    return np.linalg.inv(resu)')


# In[ ]:

get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\ncimport numpy as np\n@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n\ndef log_likelihood_cython(double[:,:] X,double[:,:] Z, double[:,:] M,double sigmaA,double sigmaX,int Kplus,int N,int D):  \n    cdef double determinant = np.linalg.det(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))\n    cdef double constant = N*D*0.5*np.log(2*np.pi) + (N-Kplus)*D*np.log(sigmaX) + Kplus*D*np.log(sigmaA) + D*0.5*np.log(determinant)\n    \n    cdef double[:,:] middle = np.identity(N) - np.dot(np.dot(Z, M),Z.T)\n    cdef double trace = np.trace(np.dot(np.dot(X.T,middle),X))\n    cdef double kernel = -0.5*np.reciprocal(sigmaX**2)*trace\n    \n    cdef double result = -constant + kernel\n    return  result\n    ')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

def mainsolver_improve(iteration,Z,alpha,K,sX,sA,ratioX = 0,ratioA = 0): 
    Z_mcmc= np.zeros((iteration,objects,1000))
    K_mcmc = np.zeros(iteration)
    sX_mcmc = np.zeros(iteration)
    sA_mcmc = np.zeros(iteration)
    alpha_mcmc = np.zeros(iteration)
    for it in range(iteration):
        Z_mcmc[it,:,0:K] = Z[:,0:K]
        alpha_mcmc[it] = alpha
        K_mcmc[it] = K
        sX_mcmc[it] = sX
        sA_mcmc[it] = sA
        for i in range(objects):
            for k in range(K):
                if k+1>K:
                    break
                if Z[i,k]>0:
                    if np.sum(Z[:,k])-Z[i,k]<=0:
                        Z[:,k:(K-1)] = Z[:,(k+1):K]
                        K -= 1
                        continue
                P = np.zeros(2)
                Z[i,k] = 1
                M1 = calcM_cython(Z,K,sX,sA) 
                P[1] = log_likelihood_cython(X,Z[:,0:K],M1,sA,sX,K,objects,dimension) + np.log(sum(Z[:,k])-Z[i,k]) - np.log(objects)
                Z[i,k] = 0
                M0 = calcM_cython(Z,K,sX,sA) 
                P[0] = log_likelihood_cython(X,Z[:,0:K],M0,sA,sX,K,objects,dimension) + np.log(objects-sum(Z[:,k])) - np.log(objects)
                P = np.exp(P - max(P))
                p = stats.uniform.rvs(loc=0,scale=1,size=1)           
                if p < P[0]/(P[0]+P[1]):
                    Z[i,k] = 0
                else:
                    Z[i,k] = 1
            trun=np.zeros(feature)
            alphan=alpha/objects
            for k_i in range(4):
                if k_i > 0:
                    temp = np.zeros((objects,k_i))
                    temp[i,:] = 1
                    Z = np.hstack((Z[:,0:K],temp))
                M = calcM2_cython(Z,K,k_i,sX,sA)
                trun[k_i] = (k_i)*np.log(alphan) - alphan - np.log(np.math.factorial(k_i)) +log_likelihood_cython(X,Z[:,0:(K+k_i)],M,sA,sX,K+k_i,objects,dimension)
            Z[i,K:(K+3)] = 0
            trun = np.exp(trun-max(trun))
            trun = trun/np.sum(trun) 
            t = 0
            for ki in range(4):
                t += trun[ki]
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
        alpha = stats.gamma.rvs(a = 1+K, loc = 0, scale = np.reciprocal(1+HN),size=1)[0]   
    return Z_mcmc, K_mcmc,sX_mcmc,sA_mcmc,alpha_mcmc


# In[ ]:

Z_mcmc, K_mcmc,sX_mcmc,sA_mcmc,alpha_mcmc=mainsolver_improve(iteration,Z,alpha,K,sX,sA,ratioX,ratioA)


# In[ ]:

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
fig.savefig('mcmc_varible_plot_cython.png')
plt.show()


# In[ ]:

samplen=iteration-1
Kf = K_mcmc[samplen]
Zf = Z_mcmc[samplen,:,0:Kf].reshape(objects,Kf)
sXf = sX_mcmc[samplen]
sAf = sA_mcmc[samplen]
A_inf = np.dot(np.linalg.inv(np.dot(Zf.T,Zf) +  ((sXf/sAf)**2)*np.identity(Kf)),np.dot(Zf.T,X))
fig = plt.figure(figsize=(12,3))
fig1 = fig.add_subplot(141)
fig1.pcolormesh(A_inf[0,:].reshape(6,6),cmap=plt.cm.gray)
fig2 = fig.add_subplot(142)
fig2.pcolormesh(A_inf[1,:].reshape(6,6),cmap=plt.cm.gray)
fig3 = fig.add_subplot(143)
fig3.pcolormesh(A_inf[0,:].reshape(6,6),cmap=plt.cm.gray)
fig4 = fig.add_subplot(144)
fig4.pcolormesh(A_inf[1,:].reshape(6,6),cmap=plt.cm.gray)
plt.show()
fig.savefig("final_result_cython.png")


# In[ ]:

img=mpimg.imread('dukeii.png')
merged_total=[]
aaa=img[:,:,3]
for i in range(len(aaa)):
    temp=aaa[i].tolist()
    merged_total=merged_total+temp 


# In[ ]:

ccc=np.round(merged_total)
pppp=np.array(ccc).reshape(25,25)


# In[ ]:

dimension = 625
objects = 100
feature = 1
paperImagesetting=25


# In[ ]:

basis_4=np.array([pppp])


# In[ ]:

A = np.array([pppp.reshape(dimension)])

