import pandas as pd
from sklearn import svm
from sklearn.cluster import DBSCAN
from scipy import stats
from sklearn.decomposition import IncrementalPCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


def process_data(dataset):
    """
    Open csv and read data
    to numpy array
    """
    df_data = pd.read_csv(dataset, sep=',', header=0)
    df_data = df_data[:10000]
    df_data = df_data.fillna(0)
    print "---procesing data done---"
    return df_data


def svm_outliers_detect(data):
    """
    Perform single class svm
    to detect outliers
    """
    clf_svm = svm.OneClassSVM(nu=0.25, gamma=0.05)
    clf_svm.fit(data)
    print "---fitting done---"
    clf_svm.fit
    results = clf_svm.decision_function(data)
    print "---detecting outliers done---"
    return results


def plot_outliers(data, outliers, thresh):
    """
    Draw inliers(blue) and
    outliers(red) using PCA
    """
    i_pca = IncrementalPCA(n_components=2, batch_size=10000)
    X = i_pca.fit(data).transform(data)
    plt.figure()
    for x, o in zip(X, outliers):
        if o < thresh:
            color = "red"
        else:
            color = "blue"
        plt.scatter(x[0], x[1], color=color)
    plt.show()
    print "---PCA done---"


def db_scan(data):
    """
    Perform DB Scan on data.
    """
    X = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=0.8, min_samples=40).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

data_file = "C:\BigData\Zemanta_challenge_1_data/output_test.csv"
data = process_data(data_file)
#print data
out_in = svm_outliers_detect(data)

threshold = stats.scoreatpercentile(out_in, 10)
#print threshold
outliers = sum(1 for zip in out_in if zip < threshold)
#print outliers
#out_in = [item for sublist in out_in for item in sublist]
#print out_in
#plot_outliers(data, out_in, threshold)
#db_scan(data)