import sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
''' die program word gebruik om te kyk hoe accurate die model is
path is die path na die csv file wat feature.py vir ons gee met al die features en die "good" value
die model kan dan net ge-export word en gesave word
'''
path ="/home/wmk/IQA/Features/Data.csv"

file=open( path, "r")
df = pd.read_csv(file)
x = df.iloc[:,1:6]
y =df.iloc[:,6:]
    #clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=50)
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.33, random_state=4)
'''Param tuning'''
tuned_parameters = [{'n_estimators': [500,800,1000], 'max_depth': [10],
                     'random_state':[20]}]

scores = ['recall','f1']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=6,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train.values.ravel())

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


    #clf = clf.fit(X_train, y_train)
    #y_pred =clf.predict(X_test)
    conf = confusion_matrix(y_test, y_pred)
    print(conf)
    #fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
    #metrics.auc(fpr, tpr)
    print(roc_auc_score(y_test, y_pred))
