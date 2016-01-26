'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        self.degree = degree
        self.regLambda = regLambda
        self.theta = None
        self.mean = None
        self.std = None


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        n = len(X)
        # create new matrix
        polyMatrix = np.zeros((n,degree))

        # elementwise power operation for each row
        for i in range(0,degree):
            polyMatrix[:,i] = np.power(X,(i+1))
        return polyMatrix


    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        # dementions of matrix including x0
        n = len(X)
        d = self.degree+1
        polyMatrix = np.zeros((n,d))

        # get higher order matrix
        polyMatrix[:,1:] = self.polyfeatures(X, self.degree)

        # add column of ones
        polyMatrix[:,0] = np.ones(n)

        # store mean and standard deviation for future
        self.mean = np.zeros(d)
        self.std = np.ones(d)

        # standardize data
        for i in range(1,d):
            self.mean[i] = np.mean(polyMatrix[:,i])
            self.std[i] = np.std(polyMatrix[:,i])
            polyMatrix[:,i] = np.divide(np.subtract(polyMatrix[:,i], self.mean[i]),self.std[i])

        # construct reg matrix
        regMatrix = self.regLambda * np.eye(d)
        regMatrix[0,0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(polyMatrix.T.dot(polyMatrix) + regMatrix).dot(polyMatrix.T).dot(y);

        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # dementions of matrix including x0
        n = len(X)
        d = self.degree+1
        polyMatrix = np.zeros((n,d))

        # get higher order matrix
        polyMatrix[:,1:] = self.polyfeatures(X, self.degree)

        # add column of ones
        polyMatrix[:,0] = np.ones(n)

        # standardize data
        for i in range(1,d):
            polyMatrix[:,i] = np.divide(np.subtract(polyMatrix[:,i],self.mean[i]),self.std[i])

        return polyMatrix.dot(self.theta)



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))

    # create model
    model = PolynomialRegression(degree = degree, regLambda = regLambda)

    # for each subset of the training data    
    for i in range(2,n):
        # get subset of data
        Xtrain_sub = Xtrain[0:i]
        Ytrain_sub = Ytrain[0:i]

        # train the model
        model.fit(Xtrain_sub,Ytrain_sub)

        # get predictions based on model
        XtrainOut = model.predict(Xtrain_sub)
        XtestOut = model.predict(Xtest)

        # calculate error
        errorTrain[i] = np.sum(np.power(np.subtract(XtrainOut,Ytrain_sub),2))/(i+1)
        errorTest[i] = np.sum(np.power(np.subtract(XtestOut,Ytest),2))/len(Xtest)

    return (errorTrain, errorTest)
