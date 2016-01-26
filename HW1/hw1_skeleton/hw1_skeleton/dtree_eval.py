'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance(numTrials=100):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape
    indices = np.arange(n)
    size = n/10

    # initialize statistics
    DecisionTreeAccuracy = np.zeros((1000,3))
    DecisionStumpAccuracy = np.zeros((1000,3))
    DT3Accuracy = np.zeros((1000,3))
    DT2Accuracy = np.zeros((1000,3))
    DT5Accuracy = np.zeros((1000,3))

    # repeat 100 times
    for i in range(0,numTrials):

        # shuffle the data
        idx = np.arange(n)
        np.random.seed()
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # iterate over folds
        for fold in range(0,10):
        
            # split the data
            start = fold*size+1
            end = (fold+1)*size
            Xtrain = X[np.nonzero((indices<start) | (indices>end)),:][0]  # train on 9 of 10 folds
            Xtest = X[np.nonzero((indices>=start) & (indices<=end)),:][0]
            ytrain = y[np.nonzero((indices<start) | (indices>end)),:][0]  # test on remaining fold
            ytest = y[np.nonzero((indices>=start) & (indices<=end)),:][0]

            # create 10% and 20% splits
            m,l = Xtrain.shape
            Xtrain_20 = Xtrain[:(m*2/10),:]
            ytrain_20 = ytrain[:(m*2/10),:]
            Xtrain_10 = Xtrain[:m/10,:]
            ytrain_10 = ytrain[:m/10,:]

            # train the decision tree
            clf = tree.DecisionTreeClassifier()
            clf_20 = tree.DecisionTreeClassifier()
            clf_10 = tree.DecisionTreeClassifier()
            clf_100 = clf.fit(Xtrain,ytrain)
            clf_20 = clf_20.fit(Xtrain_20,ytrain_20)
            clf_10 = clf_10.fit(Xtrain_10,ytrain_10)

            DT3 = tree.DecisionTreeClassifier(max_depth=3)
            DT3_20 = tree.DecisionTreeClassifier(max_depth=3)
            DT3_10 = tree.DecisionTreeClassifier(max_depth=3)
            DT3_100 = DT3.fit(Xtrain,ytrain)
            DT3_20 = DT3_20.fit(Xtrain_20,ytrain_20)
            DT3_10 = DT3_10.fit(Xtrain_10,ytrain_10)

            stump = tree.DecisionTreeClassifier(max_depth=1)
            stump_20 = tree.DecisionTreeClassifier(max_depth=1)
            stump_10 = tree.DecisionTreeClassifier(max_depth=1)
            stump_100 = stump.fit(Xtrain,ytrain)
            stump_20 = stump_20.fit(Xtrain_20,ytrain_20)
            stump_10 = stump_10.fit(Xtrain_10,ytrain_10)

            DT2 = tree.DecisionTreeClassifier(max_depth=2)
            DT2_20 = tree.DecisionTreeClassifier(max_depth=2)
            DT2_10 = tree.DecisionTreeClassifier(max_depth=2)
            DT2_100 = DT2.fit(Xtrain,ytrain)
            DT2_20 = DT2_20.fit(Xtrain_20,ytrain_20)
            DT2_10 = DT2_10.fit(Xtrain_10,ytrain_10)

            DT5 = tree.DecisionTreeClassifier(max_depth=5)
            DT5_20 = tree.DecisionTreeClassifier(max_depth=5)
            DT5_10 = tree.DecisionTreeClassifier(max_depth=5)
            DT5_100 = DT5.fit(Xtrain,ytrain)
            DT5_20 = DT5_20.fit(Xtrain_20,ytrain_20)
            DT5_10 = DT5_10.fit(Xtrain_10,ytrain_10)


            # output predictions on the remaining data
            y_pred = clf_100.predict(Xtest)
            DT3_pred = DT3_100.predict(Xtest)
            stump_pred = stump_100.predict(Xtest)
            DT2_pred = DT2_100.predict(Xtest)
            DT5_pred = DT5_100.predict(Xtest)

            y_pred_20 = clf_20.predict(Xtest)
            DT3_pred_20 = DT3_20.predict(Xtest)
            stump_pred_20 = stump_20.predict(Xtest)
            DT2_pred_20 = DT2_20.predict(Xtest)
            DT5_pred_20 = DT5_20.predict(Xtest)

            y_pred_10 = clf_10.predict(Xtest)
            DT3_pred_10 = DT3_10.predict(Xtest)
            stump_pred_10 = stump_10.predict(Xtest)
            DT2_pred_10 = DT2_10.predict(Xtest)
            DT5_pred_10 = DT5_10.predict(Xtest)

            # compute the training accuracy of the model
            DecisionTreeAccuracy[(fold+i*10),0] = accuracy_score(ytest, y_pred)
            DecisionStumpAccuracy[(fold+i*10),0] = accuracy_score(ytest, stump_pred)
            DT3Accuracy[(fold+i*10),0] = accuracy_score(ytest, DT3_pred)
            DT2Accuracy[(fold+i*10),0] = accuracy_score(ytest, DT2_pred)
            DT5Accuracy[(fold+i*10),0] = accuracy_score(ytest, DT5_pred)

            DecisionTreeAccuracy[(fold+i*10),1] = accuracy_score(ytest, y_pred_20)
            DecisionStumpAccuracy[(fold+i*10),1] = accuracy_score(ytest, stump_pred_20)
            DT3Accuracy[(fold+i*10),1] = accuracy_score(ytest, DT3_pred_20)
            DT2Accuracy[(fold+i*10),1] = accuracy_score(ytest, DT2_pred_20)
            DT5Accuracy[(fold+i*10),1] = accuracy_score(ytest, DT5_pred_20)

            DecisionTreeAccuracy[(fold+i*10),2] = accuracy_score(ytest, y_pred_10)
            DecisionStumpAccuracy[(fold+i*10),2] = accuracy_score(ytest, stump_pred_10)
            DT3Accuracy[(fold+i*10),2] = accuracy_score(ytest, DT3_pred_10)
            DT2Accuracy[(fold+i*10),2] = accuracy_score(ytest, DT2_pred_10)
            DT5Accuracy[(fold+i*10),2] = accuracy_score(ytest, DT5_pred_10)

    # create data for plotting
    xs = [10, 20,100]
    tree_mean = [np.mean(DecisionTreeAccuracy[:,2]),np.mean(DecisionTreeAccuracy[:,1]),np.mean(DecisionTreeAccuracy[:,0])]
    tree_stddev = [np.std(DecisionTreeAccuracy[:,2]),np.std(DecisionTreeAccuracy[:,1]),np.std(DecisionTreeAccuracy[:,0])]
    DT3_mean = [np.mean(DT3Accuracy[:,2]),np.mean(DT3Accuracy[:,1]),np.mean(DT3Accuracy[:,0])]
    DT3_stddev = [np.std(DT3Accuracy[:,2]),np.std(DT3Accuracy[:,1]),np.std(DT3Accuracy[:,0])]
    stump_mean = [np.mean(DecisionStumpAccuracy[:,2]),np.mean(DecisionStumpAccuracy[:,1]),np.mean(DecisionStumpAccuracy[:,0])]
    stump_stddev = [np.std(DecisionStumpAccuracy[:,2]),np.std(DecisionStumpAccuracy[:,1]),np.std(DecisionStumpAccuracy[:,0])]
    DT2_mean = [np.mean(DT2Accuracy[:,2]),np.mean(DT2Accuracy[:,1]),np.mean(DT2Accuracy[:,0])]
    DT2_stddev = [np.std(DT2Accuracy[:,2]),np.std(DT2Accuracy[:,1]),np.std(DT2Accuracy[:,0])]
    DT5_mean = [np.mean(DT5Accuracy[:,2]),np.mean(DT5Accuracy[:,1]),np.mean(DT5Accuracy[:,0])]
    DT5_stddev = [np.std(DT5Accuracy[:,2]),np.std(DT5Accuracy[:,1]),np.std(DT5Accuracy[:,0])]

    # plot
    plt.errorbar(xs,tree_mean,tree_stddev,fmt='-o',label="Full Tree")
    plt.errorbar(xs,DT3_mean,DT3_stddev,fmt='-o',label="Depth 3")
    plt.errorbar(xs,stump_mean,stump_stddev,fmt='-o',label="Stump")
    plt.errorbar(xs,DT2_mean,DT2_stddev,fmt='-o',label="Depth 2")
    plt.errorbar(xs,DT5_mean,DT5_stddev,fmt='-o',label="Depth 5")

    plt.legend()
    plt.title('1.4 LEARNING CURVES')
    plt.xlabel('Percent of Training Data')
    plt.ylabel('Mean Accuracy')
    plt.xlim(0,110)
    plt.savefig('learningcurve.pdf')

    # update statistics
    meanDecisionTreeAccuracy = np.mean(DecisionTreeAccuracy[:,0])
    meanDecisionStumpAccuracy = np.mean(DecisionStumpAccuracy[:,0])
    meanDT3Accuracy = np.mean(DT3Accuracy[:,0])
    stddevDecisionTreeAccuracy = np.std(DecisionTreeAccuracy[:,0])
    stddevDecisionStumpAccuracy = np.std(DecisionStumpAccuracy[:,0])
    stddevDT3Accuracy = np.std(DT3Accuracy[:,0])

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"
# ...to HERE.
