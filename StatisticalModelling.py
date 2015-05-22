from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import StandardScaler
import Orange
import Orange.data
import pandas as pd
import csv
import numpy as np


def prepare_data(learn_file, labels_file):
    """
    Open learning set, cluster labels
    and change ZIP codes with corresponding
    cluster label. Return X and y for learning.
    """
    print "---preparing data...",
    l_set = pd.read_csv(learn_file, sep='\t')
    l_set = l_set.iloc[::20, :]

    #replace NaN values with zero.
    l_set = l_set.fillna(0)

    with open(labels_file, mode='r') as file_in:
        reader = csv.reader(file_in)
        c_labels = {float(rows[0]): rows[1] for rows in reader}
    #change ZIP with label
    l_set['zip'] = l_set['zip'].convert_objects(convert_numeric=True).dropna()
    l_set['zip'] = l_set['zip'].map(c_labels.get)

    print "done---"

    #remove where ZIP None - for testing on part data
    l_set = l_set[l_set.zip.notnull()]

    #X for learning features, y for click
    X = l_set[['creative_id', 'zip', 'domain']]
    y = l_set['click']

    #Replace domain with numeric
    unique_d = set(X['domain'])
    dict_d = {}
    for c, d in enumerate(unique_d):
        dict_d[d] = c

    X['domain'] = X['domain'].map(dict_d.get)
    data = pd.concat([X, y],  axis=1)
    data.to_csv("data.csv", sep=',')
    return X, y


def random_forest(X, y):
    """
    Scikit Random Forest implementation
    with 100 trees, testing on 0.4 part
    of the data, and train on 0.6.
    """

    #Scale data
    X = StandardScaler().fit_transform(X)
    #split data to train and test
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)

    #create rfc object
    forest = rfc(n_estimators=100)
    #fit training data
    prob = forest.fit(X_train, y_train, ).predict_proba(X_test)
    #compute ROC
    fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    #print fpr, tpr, thresholds
    print "AUC Random Forest:  " + str(roc_auc)
    #score on test part
    #print forest.score(X_test, y_test)
    #fold =
    #K Fold Cross Validation
    """
    cv = cross_validation.KFold(len(X), n_folds=5)
    AUC = []
    for train, test in cv:
        prob = forest.fit(X[train], y[train]).predict_proba(X[test])
        #compute ROC
        fpr, tpr, thresholds = roc_curve(y[test], prob[:, 1])
        AUC.append(auc(fpr, tpr))
    print AUC
    """


def logistic_regression(X, y):
    """
    Scikit logistic regression implementation,
    testing only, beacuse of poor AUC.
    """
    #Scale data
    X = StandardScaler().fit_transform(X)
    #split data to train and test
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)

    log_r = LogisticRegression()
    prob = log_r.fit(X_train, y_train).predict_proba(X_test)
    #compute ROC
    fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    #print fpr, tpr, thresholds
    print "AUC Logistic Regression:  " + str(roc_auc)


def orange_random_forest():
    """
    Orange random forest implementation
    with K-fold cross validation.
    """
    #Scale data
    #X = StandardScaler().fit_transform(X)
    #split data to train and test
    data = Orange.data.Table("data.csv")


    forest = Orange.ensemble.forest.RandomForestLearner(trees=100, name="forest")
    res = Orange.evaluation.testing.cross_validation([forest], data, folds=5)
    print "Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[0]
    print "AUC:      %.2f" % Orange.evaluation.scoring.AUC(res)[0]


def orange_stacking():
    """
    Orange stacking implementation based on
    example code found in orange documentation:
    http://orange.biolab.si/docs/latest/reference/rst/Orange.ensemble.html#id1
    """
    data = Orange.data.Table("data.csv")
    #prepare 0 level learners
    forest = Orange.ensemble.forest.RandomForestLearner(trees=100, name="forest")
    bayes = Orange.classification.bayes.NaiveLearner(name="bayes")
    log = Orange.classification.logreg.LogRegLearner()
    knn = Orange.classification.knn.kNNLearner()
    #lin = Orange.classification.svm.LinearLearner(name="lr")
    zero_level_clf = [knn, bayes, log]
    stack = Orange.ensemble.stacking.StackedClassificationLearner(zero_level_clf)
    clf = [stack, knn, bayes, log]
    res = Orange.evaluation.testing.cross_validation(clf, data, folds=5)
    print "\n".join(["%8s: %5.3f" % (l.name, r) for r, l in zip(Orange.evaluation.scoring.AUC(res), clf)])
    #print "AUC:      %.2f" % Orange.evaluation.scoring.AUC(res)[0]


def stacking(X, y):
    """
    Stacking with scikit, implemented
    based on example found in
    https://github.com/log0/vertebral/blob/master/stacked_generalization.py
    """

    classifiers = [rfc(n_estimators=50), etc(n_estimators=100), LogisticRegression()]
    #Scale data
    X = StandardScaler().fit_transform(X)
    #split data to train and test
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)
    #0 level classifier probabilities
    X_z_train, X_z_test, y_z_train, y_z_test = cross_validation.train_test_split(X_train, y_train)
    data = np.zeros((X_z_test.shape[0], len(classifiers)))

    for i, clf in enumerate(classifiers):
        data[:, i] = clf.fit(X_z_train, y_z_train).predict_proba(X_z_test)[:, 1]


    stack_clf = rfc(n_estimators=100)
    prob = stack_clf.fit(data, y_z_test).predict_proba(X_test)
    #compute ROC
    fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    #print fpr, tpr, thresholds
    print "AUC Stacking:  " + str(roc_auc)


learning_set = "C:\BigData\Zemanta_challenge_1_data/training_set.tsv"
learning_part = "C:\BigData\Zemanta_challenge_1_data/training_part.tsv"
labels = "hc_results.csv"
X, y = prepare_data(learning_part, labels)
#random_forest(X, y)
#logistic_regression(X, y)
#orange_random_forest()
#stacking(X, y)
orange_stacking()