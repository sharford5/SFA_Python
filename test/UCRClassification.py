import sys
import os
sys.path.append(os.getcwd()[:-5])

from src.timeseries.TimeSeriesLoader import uv_load

from  src.classification.WEASELClassifier import *
from  src.classification.BOSSEnsembleClassifier import *
from  src.classification.BOSSVSClassifier import *
from  src.classification.ShotgunEnsembleClassifier import *
from  src.classification.ShotgunClassifier import *


Datasets = [#"Coffee",
            # "Beef",
            # "ECG200",
            "Gun_Point",
            # "BeetleFly"
            ]


for data in Datasets:
    train, test = uv_load(data)

    #The WEASEL Classifier
    weasel = WEASELClassifier(data)
    scoreWEASEL = weasel.eval(train, test)[0]
    print(data+"; "+scoreWEASEL)

    #The BOSS Ensemble Classifier
    boss = BOSSEnsembleClassifier(data)
    scoreBOSS = boss.eval(train, test)[0]
    print(data+"; "+scoreBOSS)

    #The BOSS VS Classifier
    bossVS = BOSSVSClassifier(data)
    scoreBOSSVS = bossVS.eval(train, test)[0]
    print(data+"; "+scoreBOSSVS)

    #The Shotgun Ensemble Classifier
    shotgunEnsemble = ShotgunEnsembleClassifier(data)
    scoreShotgunEnsemble = shotgunEnsemble.eval(train, test)[0]
    print(data+"; "+scoreShotgunEnsemble)

    #The Shotgun Classifier
    shotgun = ShotgunClassifier(data)
    scoreShotgun = shotgun.eval(train, test)[0]
    print(data+"; "+scoreShotgun)


