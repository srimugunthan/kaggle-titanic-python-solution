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
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import sys

def model_diagnostics1(x_validation, y_prediction, modelname):
        filename = ""
        filename = filename + "diagnostics-"+modelname+".txt"
        orig_stdout = sys.stdout
        f = file(filename, 'w')
        sys.stdout = f
        C = confusion_matrix(x_validation,y_prediction)
        print C
        target_names = ["Survived", "Dead"]
        print classification_report(x_validation, y_prediction, target_names=target_names)
	#print accuracy_score(x_validation, y_prediction)
        
        sys.stdout = orig_stdout
	f.close()
        filename = ""
        plt.matshow(C)
        #plt.imshow(C)
        plt.title('Confusion matrix')
        plt.colorbar()
        #plt.show()
        filename = filename+"plotconfmat-"+modelname+".png"
        
        plt.savefig(filename, dpi=100)
	return 
    
    


def naivepredictionmodel(traindata,validationdata):
	print "everyone dies"
	x = 0
	columns = ['Survived']
	y = np.repeat(x, len(validationdata) , axis=0)
	indexnum = np.arange(0,len(validationdata),1)
	predictiondata = pd.DataFrame(columns=columns, index = indexnum)
	predictiondata['Survived'] = y
	#model_diagnostics1(validationdata['Survived'],predictiondata['Survived'])
	
	return 

def decisiontreemodel(traindata,validationdata):
    print "decision tree model"
    
    #http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/
    #http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    #Decision Tree Classifier
        
    print "Info train data frame"
    print "======================================="
    print traindata.info()
    
    print "Info validationdata data frame"
    print "======================================="
    print validationdata.info()    
 
    y, X = dmatrices('Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + FamilySize -1', traindata,
    return_type="dataframe")
    print "X cols:"
    print X.columns
    print X
    y = np.ravel(y)
    # fit a CART model to the data
    model = DecisionTreeClassifier()
    model.fit(X, y) 
    print(model)         
    print "validation data cols:"
    print validationdata.columns
    print validationdata.head()
    expected, X = dmatrices('Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + FamilySize  -1', validationdata,
                            return_type="dataframe")
    print "X 2 cols:"
    print X.columns
    print X.head()
    
    expected = np.ravel(expected)
    
    predicted = model.predict(X)
    print "predicted output:"
    print predicted
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    model_diagnostics1(expected,predicted,"Dtree")    
    
    return



    

'''
import sklearn svm, linear_model

def logisticregressionmodel1():
        classifier = linear_model.LogisticRegression()
        # fit the model and make predictions:
	classifier.fit(data, targets)
	predict = classifier.predict(test_data)
        return
'''

'''
    data = X[:,5]
    X = data.reshape(len(traindata), 1)
    print "this is X:"
    print X
    y = []
    for i in range(0, len(traindata)):
	y.extend([traindata['Survived'].iloc[i]])

'''





def printmodeldiagnostics():
	print "model Diagnostics"
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
