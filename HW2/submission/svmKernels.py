"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''

    return np.power(np.dot(X1,X2.T)+1,_polyDegree)



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''

    # get shapes of vectors
    n, d = X1.shape
    m, e = X2.shape

    # initialize distance matrix
    dist = np.zeros((n,m))

    # find distance
    for i in range(0,m):
        dist[:,i] = np.sum(np.power(np.subtract(X1,X2[i,:]),2),axis=1)

    # return gaussian
    return np.exp((-dist/(2*(_gaussSigma**2))))



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''

    # get row-wise norm for each array
    norm1 = np.linalg.norm(X1,axis=1)
    norm2 = np.linalg.norm(X2,axis=1)

    # take dot product of arrays and divide rows by norm1 and columns by norm 2
    return (np.dot(X1,X2.T)/np.mat(norm1).T)/np.mat(norm2)