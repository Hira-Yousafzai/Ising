import numpy as np
from numba import jit
import pandas as pd
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

def initialize(N):   
    lattice = 2*np.random.randint(2, size=(N,N))-1
    return lattice

@jit(nopython=True)
def update(state, beta):
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  state[a, b]
                nb = state[(a+1)%N,b] + state[a,(b+1)%N] + state[(a-1)%N,b] + state[a,(b-1)%N]
                cost = 2*s*nb
                
                if cost < 0:
                    s *= -1
                elif np.random.rand() < np.exp(-cost*beta):
                    s *= -1
                state[a, b] = s
    return state

@jit(nopython=True)
def energy(state):
    energy = 0 
    
    for i in range(len(state)):
        for j in range(len(state)):
            S = state[i,j]
            nb = state[(i+1)%N, j] + state[i,(j+1)%N] + state[(i-1)%N, j] + state[i,(j-1)%N]
            energy += -nb*S
    return energy/2.

@jit(nopython=True)
def mag(state):
    m = np.sum(state)
    return np.abs(m)

def thermalize(beta):
    dumE = 0
    dumM = 0
    ilist=[]
    thM=np.zeros(thermSteps)   
    thE=np.zeros(thermSteps) 
    thState = initialize(N)  
    for i in range(thermSteps):
        ilist.append(i)
        update(thState, beta)
        oE=energy(thState)
        oM=mag(thState)
        thE[i] = oE/(N*N)
        thM[i] = oM/(N*N)
    return thState,thE,thM,ilist

N=16
tempVals = 100
thermSteps = 2500
updateSteps = 10**5
nTot, nTot2  = 1/(updateSteps*N*N), 1/(updateSteps*updateSteps*N*N)
tempList = np.linspace(0, 5,tempVals)
E,M,C,X = np.zeros(tempVals), np.zeros(tempVals), np.zeros(tempVals), np.zeros(tempVals)
errE,errM,errC,errX = np.zeros(tempVals), np.zeros(tempVals), np.zeros(tempVals), np.zeros(tempVals)
thState,thE,thM,ilist = thermalize(0.89)


for temp in range(tempVals):
    iState = thState#initialize(N)         
    
    dummyE = dummyM = dummyE2 = dummyM2 = 0
    beta=1.0/tempList[temp]
    beta2=beta*beta
    
    for i in range(updateSteps):
        update(iState, beta)           
        oneE = energy(iState)    
        oneM = mag(iState)        

        dummyE = dummyE + oneE
        dummyM  = dummyM + oneM
        dummyE2 = dummyE2 + oneE*oneE
        dummyM2 = dummyM2 + oneM*oneM

      
    E[temp] = dummyE*nTot
    errE[temp] = np.sqrt((dummyE2*nTot - dummyE * dummyE*nTot2)*nTot)
    M[temp] = dummyM*nTot
    errM[temp] = np.sqrt((dummyM2*nTot - dummyM * dummyM*nTot2)*nTot)
    C[temp] = (dummyE2*nTot - dummyE * dummyE*nTot2)*beta2
    errC[temp] = np.sqrt((dummyM2*nTot - dummyM * dummyM*nTot2)*nTot)
    X[temp] = (dummyM2*nTot - dummyM * dummyM*nTot2)*beta

plt.scatter(tempList,X, marker='+', color='r')
plt.show()
