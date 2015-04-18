import pandas as pd
from sklearn import svm



def preprocess_data(dataset):
    """
    Open csv and read data
    to numpy array
    """
    df_data = pd.read_csv(dataset, sep=',', header=0)
    df_data = df_data[:1000]
    df_data = df_data.fillna(0)
    print "---procesing data done---"
    return df_data



def svm_outliers_detect(data):
    """
    Perform single class svm
    to detect outliers
    """
    clf_svm = svm.OneClassSVM(nu=0.1, gamma=0.1)
    clf_svm.fit(data)
    print "---fitting done---"
    clf_svm.fit
    results = clf_svm.decision_function(data).ravel()
    print "---detecting outliers done---"
    return results


data_file = "C:\BigData\Zemanta_challenge_1_data/output_test.csv"
data = preprocess_data(data_file)
#print data
out_in = svm_outliers_detect(data)
outliers = sum(1 for zip in out_in if zip < 0)
print outliers