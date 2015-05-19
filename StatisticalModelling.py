from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import csv



def prepare_data(learn_file, labels_file):
    """
    Open learning set, cluster labels
    and change ZIP codes with corresponding
    cluster label. Return X and y for learning.
    """
    print "---preparing data...",
    l_set = pd.read_csv(learn_file, sep='\t')
    #l_set = l_set.iloc[::3, :]

    #replace NaN values with zero.
    l_set = l_set.fillna(0)

    with open(labels_file, mode='r') as file_in:
        reader = csv.reader(file_in)
        c_labels = {float(rows[0]): rows[1] for rows in reader}
    #change ZIP with label
    l_set['zip'] = l_set['zip'].convert_objects(convert_numeric=True).dropna()
    l_set['zip'] = l_set['zip'].map(c_labels.get)

    print "done---"

    #remove None - for testing on part data
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

    return X, y


def random_forest(X, y):
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



learning_set = "C:\BigData\Zemanta_challenge_1_data/training_set.tsv"
learning_part = "C:\BigData\Zemanta_challenge_1_data/training_set.tsv"
labels = "hc_results.csv"
X, y = prepare_data(learning_part, labels)
random_forest(X, y)
logistic_regression(X, y)