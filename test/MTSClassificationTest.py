import sys
import os
sys.path.append(os.getcwd()[:-5])

from src.timeseries.TimeSeriesLoader import mv_load
from  src.classification.MUSEClassifier import *


Datasets = [#"PenDigits",
              # "ShapesRandom",
              "DigitShapeRandom",
              # "ECG",
              # "JapaneseVowels",
              # "Libras"
]



for data in Datasets:
    train, test = mv_load(data, True)


    #The MUSE Classifier
    muse = MUSEClassifier(data)
    scoreMUSE = muse.eval(train, test)[0]
    print(data+"; "+scoreMUSE)


