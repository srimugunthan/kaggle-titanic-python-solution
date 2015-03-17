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
