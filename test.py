from sklearn import tree
import numpy as np
X = [[0,0],[2,2]]
y = [0.5,2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,y)
#print clf.predict([[1,1]])

ones = np.ones(5)
zeros = np.zeros(3)

print ones
print zeros

print np.concatenate((ones,zeros))


# export PATH=/cygdrive/c/Anaconda/:$PATH
# which python