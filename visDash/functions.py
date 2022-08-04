# -*- coding: utf-8 -*-
"""
@author: AnthonyJVasquez

Summary: 
    Functions for main.py
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.utils import Bunch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def getData(Path, TargetNames, Width, Height):
    dim = (Width, Height)
    dataDict = {'filename': [],'data': [], 'target_names': TargetNames, 'target': []}
    for file in os.listdir(Path):
        image = cv2.imread(Path + file)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        dataDict['data'].append(resized)
        dataDict['target'].append(TargetNames)
        
    dataDict['data'] = np.array(dataDict['data']).resize(len(dataDict['data']), Width * Height)
    bunch = Bunch(filename = dataDict['filename'], data=dataDict['data'], 
                  target_names=dataDict['target_names'], target = dataDict['target'])
    return bunch
    

def reportTime(StartTime=None, StopTime=None):
    if not StopTime:
        StopTime = 0
    totalTime = abs(round(StopTime - StartTime, 5))
    
    if StopTime:    
        print('\nTime to Run: ', totalTime, 's')
    return totalTime

def getDescription(DF, Exclude1 = None, Exclude2 = None):
    description = DF.describe()
    return description

def preProc(Data):
    ##process data before training
    scaler = preprocessing.StandardScaler().fit(Data.data)
    XScaled = scaler.transform(Data.data)
    XNorm = preprocessing.normalize(XScaled)
    return XNorm

def trainTSNE3D(X, NumIterations, TargetNames):
    XEmbedded = TSNE(n_components=3, verbose=1, 
                     n_iter=NumIterations).fit_transform(X)
    TSNE3D = pd.DataFrame(XEmbedded, columns=['E1', 'E2', 'E3'])
    TSNE3D['Target'] = TargetNames
    return TSNE3D

def trainTSNE2D(X, NumIterations, TargetNames):
    XEmbedded = TSNE(n_components=2, verbose=1, 
                     n_iter=NumIterations).fit_transform(X)
    TSNE2D = pd.DataFrame(XEmbedded, columns=['E1', 'E2'])
    TSNE2D['Target'] = TargetNames
    return TSNE2D

def trainPCA2D(X, Labels):
   pca = PCA(n_components=2)
   principalComp = pca.fit_transform(X)
   PCA2D = pd.DataFrame(principalComp, columns=['E1', 'E2'])
   PCA2D['Target'] = Labels
   return PCA2D

def trainPCA3D(X, Labels):
   pca = PCA(n_components=3)
   principalComp = pca.fit_transform(X)
   PCA3D = pd.DataFrame(principalComp, columns=['E1', 'E2', 'E3'])
   PCA3D['Target'] = Labels
   return PCA3D

def writeToCsv(DF, WritePath):
    DF.to_csv(WritePath, index=True)
    
def correlationPlot(DF, WritePath):
    sns.set_theme(style="ticks")
    sns.pairplot(DF, hue="Target")
    plt.savefig(WritePath)


