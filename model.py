# -*- coding: utf-8 -*-
"""

@author: Jayeola Gbenga

"""

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix, average_precision_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

def classifiers_cv(Xtrain, Xtest, ytrain, ytest):
    
    classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(max_features = 10, max_depth = 50),
    "MLP": MLPClassifier(),
    "XGB": xgb.XGBClassifier(),
    "SVC" : SVC(),
    "DCT" : DecisionTreeClassifier()
    }
    
    
    for key, classifier in classifiers.items():
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
    
        training_score = cross_val_score(classifier, Xtrain, ytrain, cv=5)
        #test_predict = cross_val_predict(classifier, pca_Xtest, pca_ytest, cv=5 )
        conf_mat = confusion_matrix(ytest, y_pred) 
        score = f1_score(ytest, y_pred, average = 'weighted')
    
        print("Classifiers: ", classifier.__class__.__name__, "Has a test score of", round(training_score.mean(), 2) * 100, "% accuracy score \n")
        print("Confusion matrix: \n " , conf_mat, "\n")
        print("f1 Score :", score , "\n")
        print("Classification report \n", classification_report(ytest, y_pred), "\n\n")


def Logisitic(Xtrain, Xtest, ytrain, ytest):
    
    lr = LogisticRegression()
    
    lr.fit(Xtrain, ytrain)
    log_reg_pred = lr.predict(Xtest)
    evaluation(ytest,log_reg_pred)
    
    return 


def Xgb(Xtrain, Xtest, ytrain, ytest):
    
    xg = xgb.XGBClassifier()
    xg.fit(Xtrain, ytrain)
    xg_pred = xg.predict(Xtest)
    evaluation(ytest,xg_pred)
    
 
def randfor(Xtrain, Xtest, ytrain, ytest):    
    randf = RandomForestClassifier(max_features = 10, max_depth = 50)
    randf.fit(Xtrain, ytrain)
    randf_pred = randf.predict(Xtest)
    evaluation(ytest,randf_pred)


def mlp(Xtrain, Xtest, ytrain, ytest):
    mlp = MLPClassifier()
    mlp.fit(Xtrain, ytrain)
    mlp_pred = mlp.predict(Xtest)
    evaluation(ytest,mlp_pred)
    
    
def svc(Xtrain, Xtest, ytrain, ytest):
    svc = SVC()
    svc.fit(Xtrain, ytrain)
    svc_pred = svc.predict(Xtest)
    evaluation(ytest,svc_pred)
    
def dtc(Xtrain, Xtest, ytrain, ytest):
    dtc = DecisionTreeClassifier()
    dtc.fit(Xtrain, ytrain)
    dtc_pred = dtc.predict(Xtest) 
    evaluation(ytest,dtc_pred)
    
def evaluation(y_test,y_pred):
    
    print(roc_auc_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred)) 