import pandas as pd
from src.timeseries.TimeSeries import NORM
import os

data_dir = os.getcwd()[:-5] + "\\datasets\\"

def load(dataset_name, seperate):
    try:
        train = pd.read_csv((data_dir + dataset_name + "\\" + dataset_name + "_TRAIN"), sep = seperate, header=None).iloc[:, 1:]
        test = pd.read_csv((data_dir + dataset_name + "\\" + dataset_name + "_TEST"), sep = seperate, header=None).iloc[:, 1:]
        train_labels = pd.read_csv((data_dir + dataset_name + "/" + dataset_name + "_TRAIN"), sep = seperate, header=None).iloc[:, 0]
        test_labels = pd.read_csv((data_dir + dataset_name + "/" + dataset_name + "_TEST"), sep = seperate, header=None).iloc[:, 0]

        print("Done reading "+dataset_name+" Training Data...  Samples: " + str(train.shape[0])+ "   Series Length: "+str(train.shape[1]))
        print("Done reading "+dataset_name+" Testing Data...  Samples: " + str(test.shape[0])+ "   Series Length: "+str(test.shape[1]))
        print()

        # Normalize train and test
        for i in range(train.shape[0]):
            train.iloc[i, :] = NORM(train.iloc[i, :].tolist())
        for i in range(test.shape[0]):
            test.iloc[i, :] = NORM(test.iloc[i, :].tolist())

        train_labels = [int(t) for t in train_labels]
        test_labels = [int(t) for t in test_labels]

        return train, test, train_labels, test_labels
    except:
        print("Data not loaded Try changing the data path in the TimeSeriesLoader file")


