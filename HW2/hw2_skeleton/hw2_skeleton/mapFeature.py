
import numpy as np


def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''

    # get length of column
    n = len(x1)

    # count column index
    index = 0

    features = np.zeros((n,28))
    for i in range(0,7):
        for j in range(0,(7-i)):
            features[:,index] = np.multiply(np.power(x1,i),np.power(x2,j))
            index += 1

    # return 27 feature matrix by removing x0 = 1 column
    return features[:,1:]
