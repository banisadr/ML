"""
==================================
Script to Find Optimal Gamma and C
==================================

Simple script to explore SVM training with varying C and gamma

Example adapted from scikit_learn documentation by Eric Eaton, 2014

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, grid_search


# load the data
filename = 'data/svmTuningData.dat'
allData = np.loadtxt(filename, delimiter=',')

X = allData[:,:-1]
Y = allData[:,-1]

# set parameters
_gaussSigma = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
equivalentGamma = np.divide(1,np.multiply(np.power(_gaussSigma,2),2))
parameters = {'kernel':('rbf'), 'C':[0.01, 0.03, 0.06, 0.1, 0.3, 0.6,
    1.0, 3.0, 6.0, 10, 30, 60, 100], 'gamma':equivalentGamma}
#C = 10.0
#_gaussSigma = 1

# train the SVM
print "Training the SVM"
#equivalentGamma = 1.0 / (2 * _gaussSigma ** 2)
model = svm.SVC(C = C, kernel='rbf', gamma=equivalentGamma)
model.fit(X, Y)

print ""
print "Testing the SVM"

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.title('SVM decision surface with C = '+str(C))
plt.axis('tight')
plt.show()
