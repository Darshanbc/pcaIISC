#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:37:27 2019

@author: Darshan
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import eig
data = pd.read_csv("/home/pjel/ML/PCA/wineQualityReds.csv") 
import matplotlib.pyplot as plt

data.head(10)

print ("information about Dataset\n")
data.info()
print("dimension of matrix\n")
data.shape
print("separate features and class\n")
X = data.iloc[:, 1:12]
y = data.iloc[:, 12]

print("Dimension of Feature matrix\n")
X.shape
print("Dimesion of Class\n")
y.shape

print("Calculation mean of all features\n")
M=X.mean(axis=0)
print("Centered Matrix\n")
centredMatrix=X-M

print("Covariance of Central Matrix\n")
covOfCentralMat=centredMatrix.cov()

print("Calculating Eigen values and Eigen Vectors of Covariance Matrix\n")
eigVal,eigVec=eig(covOfCentralMat.transpose().values)
print("Eigen Values\n")
print(eigVal)
print("Eigen Vector\n")
print(eigVec)
print("Projected Data\n")
ManualPCA = eigVec.T.dot(centredMatrix.transpose())
ManualPCA =ManualPCA.T

significance=[np.abs(i)/np.sum(eigVal) for i in eigVal]

plt.subplot(2, 1, 1)
plt.plot(np.cumsum(significance))
plt.title("Manual Impl Variance vs No. of Components")
plt.xlabel("Number of Components")
plt.ylabel("Variance")


print( "PCA from Sklearn\n")
pca=PCA()
pca.fit(X)

skPCA=pca.transform(X)
skEigVec=pca.components_

skEigVal=pca.explained_variance_

plt.subplot(2, 1, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("sklearn Impl Variance vs No. of Components")
plt.xlabel("No. of Components")
plt.ylabel("Variance")

plt.tight_layout()

plt.show()
