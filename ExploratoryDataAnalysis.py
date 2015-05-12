import pandas as pd
from sklearn import svm
from sklearn.cluster import DBSCAN
from scipy import stats
from sklearn.decomposition import IncrementalPCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
import csv


def process_data(dataset):
    """
    Open csv and read data
    to numpy array
    """
    print "---procesing data...",
    df_data = pd.read_csv(dataset, sep=',', header=0)
    #testing on part of data
    df_data = df_data.iloc[::20, :]
    df_data = df_data.replace('(X)', 0)
    df_data = df_data.astype(float)
    df_data = df_data[df_data.MalesPerFemales != 0]

    l_zip = df_data['ZIP']
    del df_data['ZIP']
    print "done---"
    #normalize
    print "---normalizing data...",
    df_data = preprocessing.normalize(df_data, norm="l1")
    #x = df_data.values #returns a numpy array
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(x)
    #df_data = pd.DataFrame(x_scaled)
    #df_data = preprocessing.scale(df_data)

    print "done---"
    return df_data, l_zip


def svm_outliers_detect(X):
    """
    Perform single class svm
    to detect outliers
    """
    clf_svm = svm.OneClassSVM(nu=0.25, gamma=0.05)
    clf_svm.fit(X)
    print "---fitting done---"
    y_pred = clf_svm.decision_function(X).ravel()

    print len(y_pred)

    print "---detecting outliers done---"
    i_pca = IncrementalPCA(n_components=2, batch_size=10000)
    X = i_pca.fit(data).transform(data)
    plt.figure()

    n_outliers = 800
    b = plt.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='red')
    c = plt.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='blue')

    plt.show()
    return X

def plot_outliers(data, outliers, thresh):
    """
    Draw inliers(blue) and
    outliers(red) using PCA
    """
    i_pca = IncrementalPCA(n_components=2, batch_size=10000)
    X = i_pca.fit(data).transform(data)
    plt.figure()
    #for x, o in zip(X, outliers):
    #    if o < thresh:
    #        color = "red"
    #    else:
    #        color = "blue"
    #    plt.scatter(x[0], x[1], color=color)
    n_outliers = 800
    b = plt.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='red')
    c = plt.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='blue')

    plt.show()
    print "---PCA done---"


def db_scan(data):
    """
    Perform DB Scan on data.
    """
    print "---DBScan...",
    #X = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=10, min_samples=10).fit(data)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers_ = sum(1 for outlier in labels if outlier == -1)
    unique_labels = set(labels)
    print "done---"
    print "Clusters found: ", n_clusters_, " Outliers: ", n_outliers_

    #plot clusters and outliers

    #PCA
    i_pca = IncrementalPCA(n_components=2, batch_size=10000)
    X = i_pca.fit(data).transform(data)
    #MDS
    #mds = MDS(n_components=2)
    #X = mds.fit(data)

    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
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


def hierarhical_clustering(data):
    """
    Perform Hierarhical
    clustering on data
    """
    print "---Hierarhical clustering...",
    hc = AgglomerativeClustering(n_clusters=20, linkage='ward').fit(data)
    labels = hc.labels_
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))


    #PCA
    i_pca = IncrementalPCA(n_components=2, batch_size=10000)
    X = i_pca.fit(data).transform(data)
    print "done---"
    #print "Clusters found: ", n_clusters_, " Outliers: ", n_outliers_
    #print labels
    for i in range(len(X)):
        plt.scatter(X[i, 0], X[i, 1], c=colors[labels[i]])



    #plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    return labels


def results_to_csv(zips, labels, output):

    with open(output, "wb") as csv_file:

        w = csv.writer(csv_file, delimiter=',')
        data = zip(map(int, zips), labels)
        w.writerows(data)



data_file = "C:\BigData\Zemanta_challenge_1_data/FINAL_nan.csv"
data, zips = process_data(data_file)
#print data
#out_in = svm_outliers_detect(data)

#threshold = stats.scoreatpercentile(out_in, 10)
#print out_in
#plot_outliers(data, out_in, threshold)
#db_scan(data)
labels = hierarhical_clustering(data)
results_to_csv(zips, labels, 'hc_results.csv')