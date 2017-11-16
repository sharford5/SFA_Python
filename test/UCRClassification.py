import sys
import os
sys.path.append(os.getcwd()[:-5])

from src.timeseries.TimeSeriesLoader import load
from  src.classification.BOSSEnsembleClassifier import *
from  src.classification.BOSSVSClassifier import *


Datasets = {#"Coffee":" ",
            # "CBF":"\t",
            # "Beef":" ",
            "ECG200":" ",
            # "Gun_Point":" ",
            # "BeetleFly":" "
            }

for data, sep in Datasets.items():
    train, test, train_labels, test_labels = load(data, sep)

    #The BOSS Ensemble Classifier
    boss = BOSSEnsembleClassifier(data)
    scoreBOSS = boss.eval(train, test, train_labels, test_labels)
    print(data+"; "+scoreBOSS)

    #The BOSS VS Classifier
    bossVS = BOSSVSClassifier(data)
    scoreBOSSVS = bossVS.eval(train, test, train_labels, test_labels)
    print(data+"; "+scoreBOSSVS)



