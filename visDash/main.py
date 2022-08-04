# -*- coding: utf-8 -*-
"""
@author: AnthonyJVasquez
"""

import time, gc
import pandas as pd
import numpy as np

from functions import reportTime
from functions import preProc
from functions import trainTSNE2D
from functions import trainTSNE3D
from functions import trainPCA2D
from functions import trainPCA3D
from functions import getDescription

from dashCreator import createDash

from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    ##start timer
    start = reportTime(time.time())
    
    ##Dashboard write path
    basepath = '/home/domain/avasquez/'
    dataPath = basepath + 'data/Neuro/'
    dashWritePath = 'projects/dataVisualization/SeparationDash.html'
    

    ##set train parameters
    nIterations = 500
    
    ##instantiate dataset object
    data = load_digits()
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = pd.DataFrame(data.target)
    
    ##data exploration
    dataDescription = getDescription(df)
    
    ##preprocess data
    xNorm = preProc(data)
    
# =============================================================================
#     TSNE
# =============================================================================
    ##train
    TSNE2D = trainTSNE2D(xNorm, nIterations, data.target)
    TSNE3D = trainTSNE3D(xNorm, nIterations, data.target)
    
# =============================================================================
#     PCA
# =============================================================================
    PCA2D = trainPCA2D(xNorm, data.target)
    PCA3D = trainPCA3D(xNorm, data.target)
    
# =============================================================================
#     Classification
# =============================================================================
    X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0)
    
    ##Create Random Forest Classifier
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = np.array(regressor.predict(X_test[:])).astype(np.int32)
    RFCM = confusion_matrix(y_test, y_pred)
    
    ##Create KNN Classifier
    clf = KNeighborsClassifier(15, weights='uniform')
    clf.fit(X_train, y_train)
    preds =clf.predict(X_test[:])
    KNNCM = confusion_matrix(y_test, preds)
    
# # =============================================================================
# #     Dashboard
# # =============================================================================

    createDash(data.target_names, PCA2D, TSNE2D, PCA3D, TSNE3D, RFCM, KNNCM, dashWritePath)
    
    ##report total running time
    reportTime(start, time.time())
    gc.collect()