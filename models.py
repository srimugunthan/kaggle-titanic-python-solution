#!/usr/bin/python
import pandas as pd
import numpy as np
import pylab as pylab


import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def model_diagnostics1(x_validation, y_prediction):
        C = confusion_matrix(x_validation,y_prediction)
        print C
        target_names = ["Survived", "Dead"]
        print classification_report(x_validation, y_prediction, target_names=target_names)
	#print accuracy_score(x_validation, y_prediction)
        plt.matshow(C)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.show()
	return 
    
    

def logisticregressionmodel(traindata,  validationdata):
    X = np.asarray(traindata)

    #-1 to remove the intercept
    y, X = dmatrices('Survived ~ Sex + Pclass + Age + Parch + SibSp + Embarked + Fare -1', traindata, return_type="dataframe")
    print "X cols:"
    print X.columns
    print X
    y = np.ravel(y)

    model = linear_model.LogisticRegression()
    model.fit(X, y)
    print "Fitted Logistic regression model:"
    print(model)
    # check the accuracy on the training set
    print "model score:"
    print model.score(X, y)
    # make predictions
    expected, X = dmatrices('Survived ~ Sex + Pclass + Age + Parch + SibSp + Embarked + Fare -1', validationdata, return_type="dataframe")
    expected = np.ravel(expected)
    
    predicted = model.predict(X)
    print "predicted output:"
    print predicted
    model_diagnostics1(expected,predicted)
    return 



def svmmodel(traindata,  validationdata):
    X = np.asarray(traindata)

    #-1 to remove the intercept
    y, X = dmatrices('Survived ~ Sex + Pclass + Age + Parch + SibSp + Embarked  -1', traindata, return_type="dataframe")
    print "X cols:"
    print X.columns
    print X
    y = np.ravel(y)
    
    model = svm.SVC()
    model.fit(X, y)
    print "Fitted SVM  model:"
    print(model)
    # check the accuracy on the training set
    print "model score:"
    print model.score(X, y)
    # make predictions
    expected, X = dmatrices('Survived ~ Sex + Pclass + Age + Parch + SibSp + Embarked  -1', validationdata, return_type="dataframe")
    expected = np.ravel(expected)
    
    predicted = model.predict(X)
    print "predicted output:"
    print predicted
    model_diagnostics1(expected,predicted)
    return
    

def randomforestmodel(traindata,  validationdata):
    X = np.asarray(traindata)

    #-1 to remove the intercept
    y, X = dmatrices('Survived ~ Sex + Pclass + Age + Parch + SibSp + Embarked  -1', traindata, return_type="dataframe")
    print "X cols:"
    print X.columns
    print X
    y = np.ravel(y)
    
    forestmodel = RandomForestClassifier(n_estimators = 1000)
    forestmodel.fit(X, y)
    print "Fitted random forest  model:"
    print(forestmodel)
    
     # check the accuracy on the training set
    print "model score:"
    print forestmodel.score(X, y)
    # make predictions
    expected, X = dmatrices('Survived ~ Sex + Pclass + Age + Parch + SibSp + Embarked  -1', validationdata, return_type="dataframe")
    expected = np.ravel(expected)
    
    
    predicted = forestmodel.predict(X)
    print "predicted output:"
    print predicted
    model_diagnostics1(expected,predicted)
    return    
