'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):

        self.numBoostingIters = numBoostingIters
        #self.numBoostingIters = 2
        self.maxTreeDepth = maxTreeDepth
        self.clf = [None] * numBoostingIters
        self.class_count = None
        self.beta = None
        self.classes = None

    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        # initialize betas
        self.beta = np.zeros(self.numBoostingIters)

        # get matrix dimetions
        n, d = X.shape

        # get classes
        self.classes = np.unique(y)

        # get number of classes
        self.class_count = np.unique(y).shape[0]

        # initialize vector of n uniform weights w1
        weights = np.ones(n)/float(n)

        # for T rounds of boosting
        for t in range(0,self.numBoostingIters):

            # fit
            self.clf[t] = tree.DecisionTreeClassifier(max_depth=self.maxTreeDepth).fit(X,y,sample_weight=weights)

            # predict
            predictions = self.clf[t].predict(X)

            # calculate error
            error = np.sum(np.logical_not(np.isclose(predictions,y))*weights)

            # compute beta
            self.beta[t] = 0.5*(np.log((1-error)/error)+np.log(self.class_count-1))

            # update weights
            mask = np.isclose(predictions,y)*np.exp(-1*self.beta[t])
            mask[mask==0] = 1
            weights = weights*mask

            # normalize weights
            weights = weights/np.sum(weights)            




    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        # get matrix dimetions
        n, d = X.shape

        # initialize predictions sum matrix
        beta_sum = np.zeros((n,self.class_count))

        for t in range(0,self.numBoostingIters):
            predictions = self.clf[t].predict(X)
            for k in range(0,self.class_count):
                beta_sum[:,k] = beta_sum[:,k] + (predictions == self.classes[k]) * self.beta[t]

        return self.classes[np.argmax(beta_sum,axis = 1)]