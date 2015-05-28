#The code is published under MIT license.
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
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
    # t_set = pd.read_csv(test_file, sep='\t', header=None, names=['click', 'creative_id', 'zip', 'domain', 'page'])
    # t_set = pd.read_csv(test_file, sep='\t', header=None, names=['creative_id', 'zip', 'domain', 'page'])

    l_set = l_set.iloc[::5, :]
    # t_set = t_set.iloc[::5, :]
    #replace NaN values with zero.
    l_set = l_set.fillna(0)
    # t_set = t_set.fillna(0)
    with open(labels_file, mode='r') as file_in:
        reader = csv.reader(file_in)
        c_labels = {float(rows[0]): rows[1] for rows in reader}
    #change ZIP with label
    l_set['zip'] = l_set['zip'].convert_objects(convert_numeric=True).dropna()
    l_set['zip'] = l_set['zip'].map(c_labels.get)

    # Change ZIP with label
    # t_set['zip'] = t_set['zip'].convert_objects(convert_numeric=True).dropna()
    # t_set['zip'] = t_set['zip'].map(c_labels.get)

    l_set = l_set.reindex(np.random.permutation(l_set.index))

    print "done---"

    #remove where ZIP None - for testing on part data
    # l_set = l_set[l_set.zip.notnull()]
    # t_set = t_set[t_set.zip.notnull()]

    #X for learning features, y for click
    X = l_set[['creative_id', 'zip', 'domain']]
    y = l_set['click']
    # X_sub = t_set[['creative_id', 'zip', 'domain']]
    # y_sub = t_set['click']


    #Replace domain with numeric
    unique_d = set(X['domain'])
    # print len(unique_d)
    # unique_d |= set(X_sub['domain'])
    dict_d = {}
    for c, d in enumerate(unique_d):
        dict_d[d] = c

    X['domain'] = X['domain'].map(dict_d.get)

    X = X.fillna(0)
    # X_sub['domain'] = X_sub['domain'].map(dict_d.get)
    # X_sub = X_sub.fillna(0)

    return X, y, # X_sub, y_sub


def random_forest(X, y, n_estimators):
    """
    Scikit Random Forest implementation
    with 100 trees, testing on 0.4 part
    of the data, and train on 0.6.
    """
    #Scale data
    #X = StandardScaler().fit_transform(X)
    #split data to train and test
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)


    # print X_train
    # print y_train
    # create rfc object
    forest = RandomForestClassifier(n_estimators=n_estimators)
    #fit training data
    prob = forest.fit(X_train, y_train, ).predict_proba(X_test)

    #compute ROC
    fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    #print fpr, tpr, thresholds
    print "AUC Random Forest:  " + str(roc_auc)


def stacking_scikit(X, y, n_estimators):
    """
    Stacking with classifiers from scikit-learn
    library. Based on example
    https://github.com/log0/vertebral/blob/master/stacked_generalization.py
    """
    X = X.as_matrix()
    y = y.as_matrix()
    base_classifiers = [RandomForestClassifier(n_estimators=n_estimators),
                        ExtraTreesClassifier(n_estimators=n_estimators),
                        GradientBoostingClassifier(n_estimators=n_estimators)]
    clf_names = ["Random Forest", "Extra Trees Classifier", "Gradient Boosting Classifier"]
    # Divide data on training and test set
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    # Arrays for classifier results
    out_train = np.zeros((X_train.shape[0], len(base_classifiers)))
    out_test = np.zeros((X_test.shape[0], len(base_classifiers)))

    t_cv = list(StratifiedKFold(y_train, n_folds=5))
    for i, clf in enumerate(base_classifiers):
        print "Training classifier " + clf_names[i]
        cv_probabilities = np.zeros((X_test.shape[0], len(t_cv)))
        # cross validation train
        for j, (train_i, test_i) in enumerate(t_cv):
            X_train_0 = X_train[train_i]
            y_train_0 = y_train[train_i]
            X_test_0 = X_train[test_i]
            # train each classifier
            clf.fit(X_train_0, y_train_0)
            # Get probabilities for click on internal test data
            proba = clf.predict_proba(X_test_0)
            out_train[test_i, i] = proba[:, 1]
            # Probabilities for test data
            proba_test = clf.predict_proba(X_test)
            cv_probabilities[:, j] = proba_test[:, 1]
        # Average of predictions
        out_test[:, i] = cv_probabilities.mean(1)

    print "Stacking with Logistic regression"
    stack_clf = LogisticRegression(C=10)
    stack_clf.fit(out_train, y_train)

    stack_prediction = stack_clf.predict_proba(out_test)

    #compute ROC
    fpr, tpr, thresholds = roc_curve(y_test, stack_prediction[:, 1])
    roc_auc = auc(fpr, tpr)
    print "AUC Stacking:  " + str(roc_auc)
    #write to file
    np.savetxt(fname="results.txt", X=stack_prediction[:, 1], fmt="%0.6f")


learning_set = "C:\BigData\Zemanta_challenge_1_data/training_set.tsv"
learning_part = "C:\BigData\Zemanta_challenge_1_data/training_part.tsv"
test_set = "C:\BigData\Zemanta_challenge_1_data/test_set.tsv"

labels = "hc_results.csv"
X, y = prepare_data(learning_set, labels)

random_forest(X, y, 10)
stacking_scikit(X, y, 10)
