'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
        self.classes = None
        self.class_count = None
        self.class_probabilities = None
        self.cond_probabilities = None
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''

        # get matrix dimetions
        n, d = X.shape

        # setup laplace smoothing
        n_correction = 0.0
        d_correction = 0.0
        if self.useLaplaceSmoothing:
            n_correction = 1.0
            d_correction = d

        # get classes
        self.classes = np.unique(y)

        # get number of classes
        self.class_count = self.classes.shape[0]

        # calculate class probabilities
        self.class_probabilities = (np.bincount(y) + n_correction)/(np.sum(np.bincount(y)) + n_correction)

        # initialize conditional probabilities matrix
        self.cond_probabilities = np.zeros((self.class_count,d))

        # calculate probabilities from data
        for k in range(self.class_count):
            self.cond_probabilities[k,:] = (np.sum(X[np.where(y==k),:],axis=1) + n_correction)/(np.sum(X[np.where(y==k),:]) + d_correction)



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

        # initialize relative probability matrix
        relative_probabilites = np.zeros((n,self.class_count))

        # calculate relative probabilities
        for k in range(0,self.class_count):
            relative_probabilites[:,k] = np.log(self.class_probabilities[k]) + np.sum(X*np.log(self.cond_probabilities[k,:]),axis=1)

        return self.classes[np.argmax(relative_probabilites, axis=1)]



    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        
        # get matrix dimetions
        n, d = X.shape

        # initialize relative probability matrix
        probabilities = np.zeros((n,self.class_count))

        # calculate relative probabilities
        for k in range(0,self.class_count):
            probabilities[:,k] = np.log(self.class_probabilities[k]) + np.sum(X*np.log(self.cond_probabilities[k,:]),axis=1)

        # normalize
        for i in range(0,n):
            probabilities[i,:] = probabilities[i,:]/np.sum(probabilities[i,:])
        
        return probabilities
        
        
class OnlineNaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''

        self.NB = NaiveBayes(useLaplaceSmoothing)
        self.X = None
        self.y = None
        self.initial = True
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''

        # if first time being called, initialize
        if self.initial:
            self.initial = False
            self.X = X
            self.y = y

        # if not first call, append data
        else:
            self.X = np.vstack((self.X,X))
            self.y = np.append(self.y,y)

        # fit data
        self.NB.fit(self.X,self.y)


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        return self.NB.predict(X)
    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        return self.NB.predictProbs(X)
