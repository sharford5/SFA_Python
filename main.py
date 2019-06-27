from src.timeseries.TimeSeriesLoader import uv_load
from src.timeseries.TimeSeriesLoader import mv_load
from src.utils import logger
import src.utils.parameters as params

FIXED_PARAMETERS = params.load_parameters()
logpath = FIXED_PARAMETERS["log_path"] + FIXED_PARAMETERS['test'] +"_"+ FIXED_PARAMETERS['dataset'] + ".log"
logger = logger.Logger(logpath)
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)

try:
    train, test = uv_load(FIXED_PARAMETERS['dataset'], logger = logger)
except:
    train, test = mv_load(FIXED_PARAMETERS['dataset'], useDerivatives = True, logger = logger)

try:
    ##=========================================================================================
    ## Multivariate Classifier Tests
    ##=========================================================================================
    if FIXED_PARAMETERS['test'] == 'MUSE':
        logger.Log("Test: MUSE")
        from src.classification.MUSEClassifier import *
        muse = MUSEClassifier(FIXED_PARAMETERS, logger)
        scoreMUSE = muse.eval(train, test)[0]
        logger.Log("%s: %s" % (FIXED_PARAMETERS['dataset'], scoreMUSE))

    ##=========================================================================================
    ## Univariate Classifier Tests
    ##=========================================================================================
    if FIXED_PARAMETERS['test'] == 'WEASEL':
        logger.Log("Test: WEASEL")
        from src.classification.WEASELClassifier import *
        weasel = WEASELClassifier(FIXED_PARAMETERS, logger)
        scoreWEASEL = weasel.eval(train, test)
        logger.Log("%s: %s" % (FIXED_PARAMETERS['dataset'], scoreWEASEL))


    if FIXED_PARAMETERS['test'] == 'BOSSEnsemble':
        logger.Log("Test: BOSSEnsemble")
        from src.classification.BOSSEnsembleClassifier import *
        boss = BOSSEnsembleClassifier(FIXED_PARAMETERS, logger)
        scoreBOSS = boss.eval(train, test)[0]
        logger.Log("%s: %s" % (FIXED_PARAMETERS['dataset'], scoreBOSS))


    if FIXED_PARAMETERS['test'] == 'BOSSVS':
        logger.Log("Test: BOSSVS")
        from src.classification.BOSSVSClassifier import *
        bossVS = BOSSVSClassifier(FIXED_PARAMETERS, logger)
        scoreBOSSVS = bossVS.eval(train, test)[0]
        logger.Log("%s: %s" % (FIXED_PARAMETERS['dataset'], scoreBOSSVS))


    if FIXED_PARAMETERS['test'] == 'ShotgunEnsemble':
        logger.Log("Test: ShotgunEnsemble")
        from src.classification.ShotgunEnsembleClassifier import *
        shotgunEnsemble = ShotgunEnsembleClassifier(FIXED_PARAMETERS, logger)
        scoreShotgunEnsemble = shotgunEnsemble.eval(train, test)[0]
        logger.Log("%s: %s" % (FIXED_PARAMETERS['dataset'], scoreShotgunEnsemble))


    if FIXED_PARAMETERS['test'] == 'Shotgun':
        logger.Log("Test: Shotgun")
        from src.classification.ShotgunClassifier import *
        shotgun = ShotgunClassifier(FIXED_PARAMETERS, logger)
        scoreShotgun = shotgun.eval(train, test)[0]
        logger.Log("%s: %s" % (FIXED_PARAMETERS['dataset'], scoreShotgun))

    ##=========================================================================================
    ## SFA Word Tests
    ##=========================================================================================
    if FIXED_PARAMETERS['test'] == 'SFAWordTest':
        logger.Log("Test: SFAWordTest")
        from src.transformation.SFA import *
        sfa = SFA(FIXED_PARAMETERS["histogram_type"], logger = logger)
        sfa.fitTransform(train, FIXED_PARAMETERS['wordLength'], FIXED_PARAMETERS['symbols'], FIXED_PARAMETERS['normMean'])
        logger.Log(sfa.__dict__)


        for i in range(test["Samples"]):
            wordList = sfa.transform2(test[i].data, "null", str_return = True)
            logger.Log("%s-th transformed TEST time series SFA word \t %s " % (i, wordList))


    if FIXED_PARAMETERS['test'] == 'SFAWordWindowingTest':
        logger.Log("Test: SFAWordWindowingTest")
        from src.transformation.SFA import *

        sfa = SFA(FIXED_PARAMETERS["histogram_type"], logger = logger)
        sfa.fitWindowing(train, FIXED_PARAMETERS['windowLength'], FIXED_PARAMETERS['wordLength'], FIXED_PARAMETERS['symbols'], FIXED_PARAMETERS['normMean'], True)
        logger.Log(sfa.__dict__)

        for i in range(test["Samples"]):
            wordList = sfa.transformWindowing(test[i], str_return = True)
            logger.Log("%s-th transformed time series SFA word \t %s " % (i, wordList))
except:
    logger.Log("Test and Dataset combo entered is not available")


