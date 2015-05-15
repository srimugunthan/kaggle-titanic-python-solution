import nltk
import re
from nltk.tokenize import RegexpTokenizer
 
def add_new_var_title(combidf):
       
    combidf['Title'] = ""
    for i in range(1, len(combidf)):
        mystring =  combidf['Name'].irow(i)
        #the_string = re.sub( '\s+', ' ', mystring ).strip()
        #tokenizer = RegexpTokenizer('[\,]+ | [\.] + | \S+')
        #tokenizer = RegexpTokenizer('[,. ]+', gaps=True)
        #words = tokenizer.tokenize(the_string)
        #the_string = ' '.join(mystring.split())
        #words = re.split(r"[,. ]",the_string)
        #words = the_string.split(' ')
        title = mystring[mystring.find(',')+1 : mystring.find('.')]
        title = title.strip()
        print title
                
        combidf['Title'].iloc[i] = title
    
       
    list1 = ['Mme', 'Mlle']
    list2 = ['Capt', 'Don', 'Major', 'Sir']
    list3 = ['Dona', 'Lady', 'the Countess', 'Jonkheer']
    '''
    for i in range(1, len(combidf)):
        mystring = combidf['Title'].irow(i)
        print mystring
        if(mystring in  s for s in list3):
            combidf['Title'].iloc[i] = 'Lady'
        if(mystring in s for s in list1):
            combidf['Title'].iloc[i] = 'Mlle'
        if(mystring in  s for s in list2):
            combidf['Title'].iloc[i] = 'Sir'
    '''    
    return



    

def add_new_var_famID(combidf):
    combidf['FamilySize'] = 1
    for i in range(1, len(combidf)):
        # Engineered variable: Family size
        combidf['FamilySize'].iloc[i] <- combidf['SibSp'].iloc[i]  + combidf['Parch'].iloc[i] + 1

    '''
    # Engineered variable: FamilyID
    combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
    combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
    combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
    # Inspect new feature
    table(combi$FamilyID)
    # Delete erroneous family IDs
    famIDs <- data.frame(table(combi$FamilyID))
    famIDs <- famIDs[famIDs$Freq <= 2,]
    combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
    # Convert to a factor
    combi$FamilyID <- factor(combi$FamilyID)
    '''
    return

def engg_new_features(traindf,testdf):
    print "construct new variables"
    # Join together the test and train sets for easier feature engineering
     
    combidf = traindf.append(testdf)
    numrowstraindf = len(traindf)
    numrowstestdf = len(testdf)

    add_new_var_title(combidf);
    add_new_var_famID(combidf);
       
       
    # Split back into test and train sets
      
    traindf =  combidf.iloc[0:numrowstraindf,]
    testdf =  combidf.iloc[(numrowstraindf):(numrowstestdf+numrowstraindf),]
    
    print "traindata len = ",len(traindf)
    print  "validation data len = ", len(testdf)
    print "Info test data frame"
    print "======================================="
    print combidf.info()
    return (traindf, testdf)
    