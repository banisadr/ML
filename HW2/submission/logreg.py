'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None

    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''

        # get matrix dimentions
        n, d = X.shape

        # initialize cost to 0
        cost = 0

        # calculate cost of each row and sum
        for i in range(0,n):
            cost += (-y[i]*np.log(self.sigmoid(theta.T.dot(X[i,:].T))) - (1-y[i])*np.log(1-self.sigmoid(theta.T.dot(X[i,:].T)))) + regLambda/2.0*np.sum(np.power(theta,2))
        return cost[0]
    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''

        # get matrix dimetions
        n, d = X.shape

        # initialize gradient
        gradient = np.zeros(d)

        # set gradient of theta0
        for i in range(0,n):
            gradient[0] += self.sigmoid(X[i,:].dot(theta))-y[i]

        # set gradient
        for i in range(0,n):
            for j in range(1,d):
                gradient[j] += (self.sigmoid(X[i,:].dot(theta))-y[i])*X[i,j] + regLambda*theta[j]

        return gradient

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''

        # get matrix dimentions
        n, d = X.shape

        # add X0 column
        X = np.c_[np.ones((n,1)), X]

        # initialize theta
        self.theta = np.random.uniform(-1,1,(d+1))

        thetaOld = self.theta

        # adjust theta until max iters is reached, or convergence
        for i in range(0,self.maxNumIters):
            thetaOld = self.theta
            self.theta = np.subtract(self.theta,np.multiply(self.computeGradient(self.theta, X, y, self.regLambda),self.alpha))
            if self.hasConverged(self.theta,thetaOld):
                return

    def hasConverged(self, thetaNew, thetaOld):
        '''
        Dedicated function used to determine if theta has converged
        '''
        return np.linalg.norm(np.subtract(thetaNew,thetaOld)) < self.epsilon


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''

        # get matrix dimentions
        n, d = X.shape

        # add X0 column
        X = np.c_[np.ones((n,1)), X]

        # make predictions
        return np.round(self.sigmoid(X.dot(self.theta)))


    def sigmoid(self, z):
        '''
        returns the sigmoid function of z
        '''

        return 1.0/(1.0+np.exp(-z))