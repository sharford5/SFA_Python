import pandas as pd
from src.timeseries.TimeSeries import TimeSeries
import os
import numpy as np

uv_dir = os.getcwd() + "\\datasets\\univariate\\"
mv_dir = os.getcwd() + "\\datasets\\multivariate\\"


def uv_load(dataset_name, APPLY_Z_NORM = True, logger = None):
    try:
        train = {}
        test = {}

        logger.Log("Loading from: %s%s" % (uv_dir, dataset_name))
        train_raw = pd.read_csv((uv_dir + dataset_name + "\\" + dataset_name + "_TRAIN"), sep=",", header=None)
        test_raw = pd.read_csv((uv_dir + dataset_name + "\\" + dataset_name + "_TEST"), sep=",", header=None)

        train["Type"] = "UV"
        train["Samples"] = train_raw.shape[0]
        train["Size"] = train_raw.shape[1]-1
        train["Labels"] = []

        test["Type"] = "UV"
        test["Samples"] = test_raw.shape[0]
        test["Size"] = test_raw.shape[1]-1
        test["Labels"] = []

        for i in range(train["Samples"]):
            label = int(train_raw.iloc[i, 0])
            train["Labels"].append(label)
            series = train_raw.iloc[i,1:].tolist()
            train[i] = TimeSeries(series, label, APPLY_Z_NORM=APPLY_Z_NORM)
            if APPLY_Z_NORM:
                train[i].NORM(True)

        for i in range(test["Samples"]):
            label = int(test_raw.iloc[i, 0])
            test["Labels"].append(label)
            series = test_raw.iloc[i, 1:].tolist()
            test[i] = TimeSeries(series, label, APPLY_Z_NORM=APPLY_Z_NORM)
            if APPLY_Z_NORM:
                test[i].NORM(True)

        logger.Log("Done reading %s Training Data...  Samples: %s  Length: %s"   % (dataset_name, str(train["Samples"]), str(train["Size"])))
        logger.Log("Done reading %s Testing Data...  Samples: %s  Length: %s"   % (dataset_name, str(test["Samples"]), str(test["Size"])))
        logger.Log("Classes: %s" % str(np.unique(train['Labels'])))

        return train, test

    except:
        logger.Log("Data not loaded. Checking the Multivariate Data Path...")



def mv_load(dataset_name, useDerivatives,  APPLY_Z_NORM = False, logger = None):
    try:
        train = {}
        test = {}

        logger.Log("Loading from: %s%s" % (mv_dir, dataset_name))
        train_raw = pd.read_csv((mv_dir + dataset_name + "\\" + dataset_name + "_TRAIN3"), sep=" ", header=None)
        test_raw = pd.read_csv((mv_dir + dataset_name + "\\" + dataset_name + "_TEST3"), sep=" ", header=None)

        train["Type"] = "MV"
        train["Samples"] = int(train_raw.iloc[-1,0])
        train["Dimensions"] = 2*(train_raw.shape[1]-3) if useDerivatives else train_raw.shape[1]-3
        train["Labels"] = []

        test["Type"] = "MV"
        test["Samples"] = int(test_raw.iloc[-1,0])
        test["Dimensions"] = 2*(test_raw.shape[1]-3) if useDerivatives else test_raw.shape[1]-3
        test["Labels"] = []

        for i in range(int(train_raw.iloc[-1,0])):
            row_info = train_raw[train_raw[0] == i+1]
            label = row_info.iloc[0,2]
            train["Labels"].append(label)
            channel = 0
            train[i] = {}
            for j in range(3, row_info.shape[1]):
                series = row_info.iloc[:,j].tolist()
                train[i][channel] = TimeSeries(series, label, APPLY_Z_NORM=APPLY_Z_NORM)
                channel += 1

            if useDerivatives:
                for j in range(channel):
                    series = train[i][j].data
                    series2 = [0. for _ in range(len(series))]
                    for u in range(1,len(series)):
                        series2[u-1] = abs(series[u] - series[u-1])
                    train[i][channel+j] = TimeSeries(series2, label, APPLY_Z_NORM=APPLY_Z_NORM)

        for i in range(int(test_raw.iloc[-1,0])):
            row_info = test_raw[test_raw[0] == i+1]
            label = row_info.iloc[0,2]
            test["Labels"].append(label)
            channel = 0
            test[i] = {}
            for j in range(3, row_info.shape[1]):
                series = row_info.iloc[:,j].tolist()
                test[i][channel] = TimeSeries(series, label, APPLY_Z_NORM=APPLY_Z_NORM)
                channel += 1

            if useDerivatives:
                for j in range(channel):
                    series = test[i][j].data
                    series2 = [0. for _ in range(len(series))]
                    for u in range(1,len(series)):
                        series2[u-1] = abs(series[u] - series[u-1])
                    test[i][channel+j] = TimeSeries(series2, label, APPLY_Z_NORM=APPLY_Z_NORM)


        logger.Log("Done reading %s Training Data...  Samples: %s  Dimensions: %s"   % (dataset_name, str(train["Samples"]), str(train["Dimensions"])))
        logger.Log("Done reading %s Testing Data...  Samples: %s  Dimensions: %s"   % (dataset_name, str(test["Samples"]), str(test["Dimensions"])))
        logger.Log("Classes: %s" % str(np.unique(train['Labels'])))

        return train, test
    except:
        logger.Log("Data not loaded. Check Data name and path")
