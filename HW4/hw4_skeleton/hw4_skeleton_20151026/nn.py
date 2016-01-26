'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
import pandas as pd

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate, numEpochs=100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.s = None
        self.a_matrix = dict()
        self.theta_matrix = dict()
        self.lambda_val = 0.0001
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        # Get shape of X
        n,d = X.shape

        # Get output units
        output_units = np.unique(y).size

        # Initialize S - full layer vector
        self.s = np.concatenate((np.concatenate(([d],self.layers), axis=1),[output_units]), axis=1)

        # Initialize theta matrixes
        for i in range(self.s-1):
            self.theta_matrix[i] = np.random.uniform(-self.epsilon,self.epsilon, (self.s[i+1],self.s[i]+1))

        #self.theta_matrix[0].ravel

        # Backpropogate
        for i in range(self.numEpochs):

            # Initilaize gradient and delta storage
            delta_matrix = dict()
            grad_matrix = dict()

            # Compute a values
            self.forward_propogation(X)

            # Compute delta(L)
            delta_matrix[len(self.s)] = self.a_matrix[len(self.s)]
            delta_matrix[len(self.s)][:][y] -= 1

            # Compute errors
            j = len(self.s) - 1
            while j < 0:
                delta_matrix[j] = np.multiply(np.dot(delta_matrix[j+1],theta_matrix[j].transpose),np.multiply(self.a_matrix[j][:,1:],(1-self.a_matrix[j][:,1:])))
                j -=1
            
            # Compute gradients
            for j in range(len(self.s)-1):
                grad_matrix[j] = np.dot(delta_matrix[j+1], self.a_matrix.transpose)/n
                grad_matrix[j][:,1:] = grad_matrix[j][:,1:] + self.theta_matrix[j][;,1:]*self.lambda_val
                self.theta_matrix[j] = self.theta_matrix[j] - grad_matrix[j]*self.learningRate


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        # Forward propogate with fit thetas
        self.forward_propogation(X)

        # Return index of most probable for each instance
        return np.argmax(self.a_matrix[len(self.a_matrix)], axis=1)
    
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''


####################################################################
##################### PRIVATE FUNCTIONS ############################
####################################################################

    def sigmoid(self, z):
        '''
        Calcuate the sigmoid of an input
        '''
        return 1.0/(1.0+np.exp(-z))



    def forward_propogation(self, X):
        '''
        Used to propogate X through neural network
        Arguments:
            X is a n-by-d numpy array
        Updates:
            a_matrix with n-dimensional numpy arrays of the layer outputs
        '''
        # Get shape of X
        n,d = X.shape

        # Initialize a(1)
        self.a_matrix[0] = X

        # Propogate
        for i in range(len(self.theta_matrix)):
            # Add bias terms
            self.a_matrix[i] = np.insert(self.a_matrix[i],0,1,axis=1)
            # propogate layer forward
            self.a_matrix[i+1] = self.sigmoid(np.dot(self.a_matrix[i],theta_matrix[i+1].transpose))

