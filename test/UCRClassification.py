import sys
import os
sys.path.append(os.getcwd()[:-5])

from src.timeseries.TimeSeriesLoader import load
from  src.classification.BOSSEnsembleClassifier import *
from  src.classification.BOSSVSClassifier import *
from  src.classification.ShotgunEnsembleClassifier import *
from  src.classification.ShotgunClassifier import *


Datasets = {#"Coffee":" ",
            # "CBF":"\t",
            # "Beef":" ",
            # "ECG200":" ",
            "Gun_Point":" ",
            # "BeetleFly":" "
            }


for data, sep in Datasets.items():
    train, test, train_labels, test_labels = load(data, sep)

    # #The BOSS Ensemble Classifier
    # boss = BOSSEnsembleClassifier(data)
    # scoreBOSS = boss.eval(train, test, train_labels, test_labels)[0]
    # print(data+"; "+scoreBOSS)

    # #The BOSS VS Classifier
    # bossVS = BOSSVSClassifier(data)
    # scoreBOSSVS = bossVS.eval(train, test, train_labels, test_labels)[0]
    # print(data+"; "+scoreBOSSVS)

    #The Shotgun Ensemble Classifier
    shotgunEnsemble = ShotgunEnsembleClassifier(data)
    scoreShotgunEnsemble = shotgunEnsemble.eval(train, test, train_labels, test_labels)[0]
    print(data+"; "+scoreShotgunEnsemble)

    # #The Shotgun Classifier
    # shotgun = ShotgunClassifier(data)
    # scoreShotgun = shotgun.eval(train, test, train_labels, test_labels)[0]
    # print(data+"; "+scoreShotgun)


