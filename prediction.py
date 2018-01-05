# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:49:19 2018

@author: User
"""
import matplotlib.pyplot as mpl
from PastSampler import PastSampler
 
#%%Features are channels
C = np.hstack((Di[CN] for Di in D))[:, None, :]
HP = 1                 #Holdout period
A = C[0:-HP]                
SV = A.mean(axis = 0)   #Scale vector
C /= SV                 #Basic scaling of data
#%%Make samples of temporal sequences of pricing data (channel)
NPS, NFS = 128, 8         #Number of past and future samples
ps = PastSampler(NPS, NFS)
B, Y = ps.transform(A)

#%%Architecture of the neural network
from TFANN import ANNR
 
NC = B.shape[2]
#2 1-D conv layers with relu followed by 1-d conv output layer
ns = [('C1d', [8, NC, NC * 2], 4), ('AF', 'relu'), 
      ('C1d', [8, NC * 2, NC * 2], 2), ('AF', 'relu'), 
      ('C1d', [8, NC * 2, NC], 2)]
#Create the neural network in TensorFlow
cnnr = ANNR(B[0].shape, ns, batchSize = 16, learnRate = 2e-5, 
            maxIter = 64, reg = 1e-5, tol = 1e-2, verbose = True)
cnnr.fit(B, Y)


#prediction
PTS = []                        #Predicted time sequences
P = B[[-1]]                     #Most recent time sequence
for i in range(HP // NFS + 1):  #Repeat prediction
    YH = cnnr.predict(P)
    P = np.concatenate([P[:, NFS:], YH], axis = 1)
    PTS.append(YH)
PTS = np.hstack(PTS).transpose((1, 0, 2))
A = np.vstack([A, PTS]) #Combine predictions with original data
A = np.squeeze(A) * SV  #Remove unittime dimension and rescale
C = np.squeeze(C) * SV


nt = 4
PF = cnnr.PredictFull(B[:nt])
for i in range(nt):
    fig, ax = mpl.subplots(1, 4, figsize = (16 / 1.24, 10 / 1.25))
    ax[0].plot(PF[0][i])
    ax[0].set_title('Input')
    ax[1].plot(PF[2][i])
    ax[1].set_title('Layer 1')
    ax[2].plot(PF[4][i])
    ax[2].set_title('Layer 2')
    ax[3].plot(PF[5][i])
    ax[3].set_title('Output')
    fig.text(0.5, 0.06, 'Time', ha='center')
    fig.text(0.06, 0.5, 'Activation', va='center', rotation='vertical')
    mpl.show()
    

CI = list(range(C.shape[0]))
AI = list(range(C.shape[0] + PTS.shape[0] - HP))
NDP = PTS.shape[0] #Number of days predicted
for i, cli in enumerate(cl):
    fig, ax = mpl.subplots(figsize = (16 / 1.5, 10 / 1.5))
    hind = i * len(CN) + CN.index('high')
    ax.plot(CI[-4 * HP:], C[-4 * HP:, hind], label = 'Actual')
    ax.plot(AI[-(NDP + 1):], A[-(NDP + 1):, hind], '--', label = 'Prediction')
    ax.legend(loc = 'upper left')
    ax.set_title(cli + ' (High)')
    ax.set_ylabel('USD')
    ax.set_xlabel('Time')
    #ax.axes.xaxis.
    mpl.show()