from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.cross_validation import StratifiedKFold as skf
from sklearn.metrics import roc_curve, auc
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
    #replace NaN values with zero.
    l_set = l_set.fillna(0)
    with open(labels_file, mode='r') as file_in:
        reader = csv.reader(file_in)
        c_labels = {float(rows[0]): rows[1] for rows in reader}
    #change ZIP with label
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


def decide(X, y):
    #Scale data
    X = StandardScaler().fit_transform(X)
    rfc.fit(X, y)
    print rfc.predict_proba(X)
    #fold =


learning_set = "C:\BigData\Zemanta_challenge_1_data/training_set.tsv"
learning_part = "C:\BigData\Zemanta_challenge_1_data/training_part.tsv"
labels = "hc_results.csv"
X, y = prepare_data(learning_part, labels)
#decide(X, y)