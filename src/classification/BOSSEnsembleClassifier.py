from  src.transformation.BOSS import *
from joblib import Parallel, delayed
import pickle

'''
The Bag-of-SFA-Symbols Ensemble Classifier as published in
 Schäfer, P.: The boss is concerned with time series classification
 in the presence of noise. DMKD (2015)
'''

class BOSSEnsembleClassifier():

    def __init__(self, FIXED_PARAMETERS, logger):
        self.NAME = FIXED_PARAMETERS['dataset']
        self.train_bool = FIXED_PARAMETERS['train_bool']
        self.score_path = './stored_models/BOSSEnsemble_%s_score.p' % (self.NAME)
        self.model_path = './stored_models/BOSSEnsemble_%s_model.p' % (self.NAME)

        self.factor = 0.92
        self.maxF = 16
        self.minF = 6
        self.maxS = 4
        self.MAX_WINDOW_LENGTH = FIXED_PARAMETERS['MAX_WINDOW_LENGTH']
        self.NORMALIZATION = [True, False]
        self.ENSEMBLE_WEIGHTS = True
        logger.Log(self.__dict__, level = 0)
        self.logger = logger


    def eval(self, train, test):
        if self.train_bool:
            scores = self.fit(train)
            pickle.dump(scores, open(self.score_path, 'wb'))
            pickle.dump(self.model, open(self.model_path, 'wb'))
        else:
            scores = pickle.load(open(self.score_path, 'rb'))
            self.model = pickle.load(open(self.model_path, 'rb'))

        self.logger.Log("Final Ensembled Models...")
        for m in self.model:
            self.logger.Log("Norm:%s  WindowLength:%s  Features:%s  TrainScore:%s" % (m.norm, m.windowLength, m.features, m.score))

        p = self.predict(self.model, test, testing=True)
        test_acc = p.correct/test["Samples"]
        return "BOSS Ensemble; "+str(round(scores/train['Samples'],3))+"; "+str(round(test_acc,3)), p.labels


    def fit(self, train):
        self.minWindowLength = 10
        maxWindowLength = getMax(train, self.MAX_WINDOW_LENGTH)
        self.windows = self.getWindowsBetween(self.minWindowLength, maxWindowLength)
        self.logger.Log("Windows: %s" % self.windows)

        bestCorrectTraining = 0.
        bestScore = None

        for norm in self.NORMALIZATION:
            models, correctTraining = self.fitEnsemble(norm, train)
            p = self.predict(models, train, testing=True)
            if bestCorrectTraining < p.correct:
                bestCorrectTraining = p.correct
                bestScore = p.correct
                currentMax = max([i.score for i in models])
                if currentMax < bestScore:
                    for i in range(len(models)):
                        if models[i].score == currentMax:
                            models[i].score = bestScore
                            break
                self.model = models

        return bestScore


    def fitIndividual(self, NormMean, samples, i):
        model = BOSSModel(NormMean, self.windows[i])
        boss = BOSS(self.maxF, self.maxS, self.windows[i], NormMean, logger = self.logger)
        train_words = boss.createWords(samples)

        f = self.minF
        keep_going = True
        while (f <= self.maxF) & (keep_going == True):
            bag = boss.createBagOfPattern(train_words, samples, f)
            s = self.prediction(bag, bag)
            if s.correct > model.score:
                model.score = s.correct
                model.features = f
                model.boss = boss
                model.bag = bag
            if s.correct == samples["Samples"]:
                keep_going = False
            f += 2

        self.logger.Log("Correct for Norm=%s & Window=%s: %s @ f=%s" % (NormMean, self.windows[i], model.score, model.features))
        self.results.append(model)


    def fitEnsemble(self, NormMean, samples):
        correctTraining = 0
        self.results = []
        self.logger.Log("%s  Fitting for a norm of %s" % (self.NAME, str(NormMean)))

        Parallel(n_jobs=1, backend="threading")(delayed(self.fitIndividual)(NormMean, samples, i) for i in range(len(self.windows)))

        #Find best correctTraining
        for i in range(len(self.results)):
            if self.results[i].score > correctTraining:
                correctTraining = self.results[i].score

        self.logger.Log("CorrectTrain for a norm of %s" % (correctTraining))
        # Remove Results that are no longer satisfactory
        new_results = []
        self.logger.Log("Stored Models for Norm=%s" % NormMean)
        for i in range(len(self.results)):
            if self.results[i].score >= (correctTraining * self.factor):
                self.logger.Log("WindowLength:%s  Features:%s  TrainScore:%s" % (self.results[i].windowLength, self.results[i].features, self.results[i].score))
                new_results.append(self.results[i])

        return new_results, correctTraining


    def prediction(self, bag_test, bag_train, testing = False):
        p_labels = [None for i in range(len(bag_test))]
        p_correct = 0

        for i in range(len(bag_test)):
            minDistance = 0x7fffffff

            noMatchDistance = 0
            for key in bag_test[i].bob.keys():
                noMatchDistance += bag_test[i].bob[key] ** 2

            for j in range(len(bag_train)):
                if (j != i) or testing:  #Second condition is to avoid direct match of train vs train (itself)
                    distance = 0
                    for key in bag_test[i].bob.keys():
                        buf = bag_test[i].bob[key] - bag_train[j].bob[key] if key in bag_train[j].bob.keys() else bag_test[i].bob[key]
                        distance += buf ** 2

                        if distance >= minDistance:
                            continue

                    if (distance != noMatchDistance) & (distance < minDistance):
                        minDistance = distance
                        p_labels[i] = bag_train[j].label

            if bag_test[i].label == p_labels[i]:
                p_correct += 1
        print(p_correct)

        return Predictions(p_correct, p_labels)


    def predictIndividual(self, models, samples, testing, i):
        score = models[i]
        model = score.boss
        wordsTest = model.createWords(samples)
        test_bag = model.createBagOfPattern(wordsTest, samples, score.features)
        p = self.prediction(test_bag, score.bag, testing)
        for j in range(len(p.labels)):
            self.Label_Matrix[j][i] = {p.labels[j] : score.score}

        self.logger.Log("Predicting for WindowLength:%s" % (model.windowLength))



    def predict(self, models, samples, testing=False):
        uniqueLabels = np.unique(samples["Labels"])
        self.Label_Matrix = [[None for _ in range(len(models))] for _ in range(samples['Samples'])]
        predictedLabels = [None for _ in range(samples['Samples'])]

        Parallel(n_jobs=1, backend="threading")(delayed(self.predictIndividual)(models, samples, testing, i) for i in range(len(models)))

        maxCounts = [None for _ in range(samples['Samples'])]
        for i in range(len(self.Label_Matrix)):
            counts = {l:0 for l in uniqueLabels}
            for k in self.Label_Matrix[i]:
                if (k != None) and (list(k.keys())[0] != None):
                    label = list(k.keys())[0]
                    count = counts[label] if label in counts.keys() else 0
                    increment = list(k.values())[0] if self.ENSEMBLE_WEIGHTS else 1
                    count = increment if (count == None) else count + increment
                    counts[label] = count

            maxCount = -1
            for e in uniqueLabels: # counts.keys():
                if (predictedLabels[i] == None) or (maxCount < counts[e]) or (maxCount == counts[e]) and (predictedLabels[i] <= e):
                    maxCount = counts[e]
                    predictedLabels[i] = e

        correctTesting = 0
        for i in range(samples["Samples"]):
            correctTesting += 1 if samples[i].label == predictedLabels[i] else 0

        return Predictions(correctTesting, predictedLabels)


    def getWindowsBetween(self, minWindowLength, maxWindowLength):
        windows = []
        for windowLength in range(maxWindowLength, minWindowLength-1, -1):
            windows.append(windowLength);
        return windows



def getMax(samples, maxWindowSize):
    m = 0
    for i in range(samples['Samples']):
        m = max(len(samples[i].data), m)

    return min(maxWindowSize, m)



class BOSSModel():
    def __init__(self, normed, windowLength):
        self.NAME = "BOSS Ensemble"
        self.score = 0
        self.features = 0
        self.norm = normed
        self.windowLength = windowLength
        self.boss = None
        self.bag = None

class Predictions():
    def __init__(self, correct, labels):
        self.correct = correct
        self.labels = labels
