import pandas as pd
from src.timeseries.TimeSeries import TimeSeries
import os

uv_dir = os.getcwd()[:-5] + "\\datasets\\univariate\\"
mv_dir = os.getcwd()[:-5] + "\\datasets\\multivariate\\"


def uv_load(dataset_name):
    try:
        train = {}
        test = {}

        train_raw = pd.read_csv((uv_dir + dataset_name + "\\" + dataset_name + "_TRAIN"), sep=" ", header=None)
        test_raw = pd.read_csv((uv_dir + dataset_name + "\\" + dataset_name + "_TEST"), sep=" ", header=None)

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
            train[i] = TimeSeries(series, label)
            train[i].NORM(True)

        for i in range(test["Samples"]):
            label = int(test_raw.iloc[i, 0])
            test["Labels"].append(label)
            series = test_raw.iloc[i, 1:].tolist()
            test[i] = TimeSeries(series, label)
            test[i].NORM(True)


        print("Done reading " + dataset_name + " Training Data...  Samples: " + str(train["Samples"]) + "   Length: " + str(train["Size"]))
        print("Done reading " + dataset_name + " Testing Data...  Samples: " + str(test["Samples"]) + "   Length: " + str(test["Size"]))
        print()

        return train, test

    except:
        print("Data not loaded Try changing the data path in the TimeSeriesLoader file")



def mv_load(dataset_name, useDerivatives):
    try:
        train = {}
        test = {}

        train_raw = pd.read_csv((mv_dir + dataset_name + "\\" + dataset_name + "_TRAIN3"), sep = " ", header=None)
        test_raw = pd.read_csv((mv_dir + dataset_name + "\\" + dataset_name + "_TEST3"), sep = " ", header=None)

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
                train[i][channel] = TimeSeries(series, label)
                channel += 1
                if useDerivatives:
                    series2= [0. for _ in range(len(series)-1)]
                    for u in range(1,len(series)):
                        series2[u-1] = series[u] - series[u-1]
                    train[i][channel] = TimeSeries(series2, label)
                    channel += 1

        for i in range(int(test_raw.iloc[-1,0])):
            row_info = test_raw[test_raw[0] == i+1]
            label = row_info.iloc[0,2]
            test["Labels"].append(label)
            channel = 0
            test[i] = {}
            for j in range(3, row_info.shape[1]):
                series = row_info.iloc[:,j].tolist()
                test[i][channel] = TimeSeries(series, label)
                channel += 1
                if useDerivatives:
                    series2= [0. for _ in range(len(series)-1)]
                    for u in range(1,len(series)):
                        series2[u-1] = series[u] - series[u-1]
                    test[i][channel] = TimeSeries(series2, label)
                    channel += 1


        print("Done reading "+dataset_name+" Training Data...  Samples: " + str(train["Samples"])+ "   Dimensions: "+str(train["Dimensions"]))
        print("Done reading "+dataset_name+" Testing Data...  Samples: " + str(test["Samples"])+ "   Dimensions: "+str(test["Dimensions"]))
        print()

        return train, test
    except:
        print("Data not loaded Try changing the data path in the TimeSeriesLoader file")


