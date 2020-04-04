import sklearn
from sklearn import tree
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
path ="/home/wikus/IQA/MAchine_Learning/ML_Classifiers/Data.csv"
def readCSV (path):
    file=open( path, "r")
    df = pd.read_csv(file)
    x = df.iloc[:,1:6]
    y =df.iloc[:,6:]
    print(y)
    print(x)
    clf = tree.DecisionTreeClassifier()

    X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=4)
    print(len(X_train))
    print(len(y_train))
    clf = clf.fit(X_train, y_train)
    y_pred =clf.predict(X_test)
    conf = confusion_matrix(y_test, y_pred)
    print(conf)
    #fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
    #metrics.auc(fpr, tpr)
    print(roc_auc_score(y_test, y_pred))
    #for i,row in reader.iterrows():
readCSV(path)
