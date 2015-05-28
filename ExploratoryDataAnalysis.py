#The code is published under MIT license.

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
    # df_data = df_data.iloc[::10, :]
    df_data = df_data.replace('(X)', 0)
    df_data = df_data.astype(float)
    df_data = df_data[df_data.MalesPerFemales != 0]

    l_zip = df_data['ZIP']
    del df_data['ZIP']
    print "done---"
    #normalize
    print "---normalizing data...",
    df_data_n = preprocessing.normalize(df_data, norm="l1")
    # df_data_n = preprocessing.scale(df_data)

    print "done---"
    return df_data_n, l_zip, df_data


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


def db_scan(data):
    """
    Perform DB Scan on data.
    """
    print "---DBScan...",
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
    return labels


def hierarhical_clustering(data):
    """
    Perform Hierarhical
    clustering on data
    """
    print "---Hierarhical clustering...",
    hc = AgglomerativeClustering(n_clusters=40, linkage='ward').fit(data)
    labels = hc.labels_
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    #PCA
    i_pca = IncrementalPCA(n_components=2, batch_size=10000)
    X = i_pca.fit(data).transform(data)
    print "done---"
    for i in range(len(X)):
        plt.scatter(X[i, 0], X[i, 1], c=colors[labels[i]])

    plt.show()
    return labels


def results_to_csv(zips, labels, output):
    """
    Save results to csv
    """
    with open(output, "wb") as csv_file:
        w = csv.writer(csv_file, delimiter=',')
        data = zip(map(int, zips), labels)
        w.writerows(data)


def averages_fix(labels, all_data):
    """
    Calculate  white %, income,
    unemployment, density average
    """
    column_names = ['Labels', 'White %', 'Black %', 'Unemp. Rate', 'Density Per Sq Mile', 'Land-Sq-Mi']
    df_average = pd.DataFrame(data=np.zeros((0, len(column_names))), columns=column_names)
    print "---Computing averages...",
    n_clusters = len(set(labels))
    # Allocate list of zeros
    white = [0] * n_clusters
    black = [0] * n_clusters
    unemp = [0] * n_clusters
    density = [0] * n_clusters
    land = [0] * n_clusters
    counters = [0] * n_clusters
    for row, label in zip(all_data.iterrows(), labels):
        white[label] += row[1]['White']
        black[label] += row[1]['Black']
        unemp[label] += row[1]['Unemp. Rate']
        density[label] += row[1]['Density Per Sq Mile']
        land[label] += row[1]['Land-Sq-Mi']
        counters[label] += 1
    # divide with counter to get average
    for i, counter in enumerate(counters):
        white[i] /= counters[i]
        black[i] /= counters[i]
        unemp[i] /= counters[i]
        density[i] /= counters[i]
        land[i] /= counters[i]

    for i in range(0, n_clusters):
        df_average = df_average.append({'Labels': i, 'White %': white[i], 'Black %': black[i], 'Unemp. Rate': unemp[i],
                                        'Density Per Sq Mile': density[i], 'Land-Sq-Mi': land[i]}, ignore_index='true')
    print "done---"
    df_average.to_csv('averages_per_cluster_final.csv')
    return df_average


data_file = "C:\BigData\Zemanta_challenge_1_data/FINAL.csv"
data, zips, raw_data = process_data(data_file)

labels = db_scan(data)
labels = hierarhical_clustering(data)
results_to_csv(zips, labels, 'hc_results_db.csv')
# labels = []


# with open("hc_results.csv", "rb") as csv_file:
#         data = csv.reader(csv_file, delimiter=',')
#         for row in data:
#             labels.append(row[1])

#print labels
# labels = [int(i) for i in labels]
# averages_fix(labels, raw_data)