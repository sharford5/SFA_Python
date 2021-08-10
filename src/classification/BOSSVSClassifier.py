from  src.transformation.BOSSVS import *
import random
from statistics import mode
from joblib import Parallel, delayed
import pickle

'''
The Bag-of-SFA-Symbols in Vector Space classifier as published in
 Schäfer, P.: Scalable time series classification. DMKD (2016)
'''

class BOSSVSClassifier():

    def __init__(self, FIXED_PARAMETERS, logger):
        self.NAME = FIXED_PARAMETERS['dataset']
        self.train_bool = FIXED_PARAMETERS['train_bool']
        self.score_path = './stored_models/BOSSVS_%s_score.p' % (self.NAME)
        self.model_path = './stored_models/BOSSVS_%s_model.p' % (self.NAME)

        self.factor = 0.95
        self.maxF = 16
        self.minF = 4
        self.maxS = 4
        self.MAX_WINDOW_LENGTH = 250
        self.folds = 10
        self.NORMALIZATION = [True, False]
        self.ENSEMBLE_WEIGHTS = True
        logger.Log(self.__dict__, level = 0)
        self.logger = logger

    def getStratifiedTrainTestSplitIndices(self, samples, splits): #New Cross Validation Method
        elements = {}
        for i in range(samples['Samples']):
            l = samples[i].label
            if l in elements.keys():
                elements[l].append(i)
            else:
                elements[l] = [i]

        sets = [[] for _ in range(splits)]

        for k in elements.keys():
            v = elements[k]
            keep_going = True
            while keep_going:
                for s in range(splits):
                    if len(v) != 0:
                        dd = v.pop(0)
                        sets[s].append(dd)
                    else:
                        keep_going = False


        self.train_indices = {i:[] for i in range(splits)}
        self.test_indices = {i:[] for i in range(splits)}
        for i in range(splits):
            for j in range(splits):
                if i == j:
                    self.test_indices[i] = sets[j]
                else:
                    self.train_indices[i] += sets[j]


    def createFoldIndex(self, l, n_folds):  #Old Cross Validation Method
        random.seed(0)
        fold_index = [0]
        perm = [i for i in range(l)]
        # random.shuffle(perm)

        for i in range(1, n_folds):
            fold_index.append(int(math.floor(i * l / n_folds)))
        fold_index.append(l)
        self.train_indices = {i:[] for i in range(n_folds)}
        self.test_indices = {i:[] for i in range(n_folds)}
        for i in range(n_folds):
            for j in range(l):
                if (j < fold_index[i]) | (j >= fold_index[i+1]):
                    self.train_indices[i].append(perm[j])
                else:
                    self.test_indices[i].append(perm[j])


    def eval(self, train, test):
        # self.createFoldIndex(train["Samples"], self.folds)
        self.getStratifiedTrainTestSplitIndices(train, self.folds)

        if self.train_bool:
            correctTraining = self.fit(train)
            pickle.dump(correctTraining, open(self.score_path, 'wb'))
            pickle.dump(self.model, open(self.model_path, 'wb'))
        else:
            correctTraining = pickle.load(open(self.score_path, 'rb'))
            self.model = pickle.load(open(self.model_path, 'rb'))


        train_acc = correctTraining/train["Samples"]

        self.logger.Log("Final Ensembled Models...")
        for m in self.model:
            self.logger.Log("Norm:%s  WindowLength:%s  Features:%s  TrainScore:%s" % (m.norm, m.windowLength, m.features, m.score))

        p = self.prediction(self.model, test)
        test_acc = p.correct/test["Samples"]
        return ("BOSSVS; "+str(round(train_acc,3))+"; "+str(round(test_acc,3))), p.labels


    def fit(self, train, ):
        maxCorrect = -1
        self.minWindowLength = 10
        maxWindowLength = getMax(train, self.MAX_WINDOW_LENGTH)

        count = math.sqrt(maxWindowLength)
        distance = (maxWindowLength - self.minWindowLength) / count
        windows = []
        c = self.minWindowLength
        while c <= maxWindowLength:
            windows.append(int(c))
            c += math.floor(distance)
        # windows = [22]

        for normMean in self.NORMALIZATION:
            models = self.fitEnsemble(windows, normMean, train)
            p = self.prediction(models, train)

            if maxCorrect <= p.correct:
                maxCorrect = p.correct
                currentMax = max([i.score for i in models])
                if currentMax < maxCorrect:
                    for i in range(len(models)):
                        if models[i].score == currentMax:
                            models[i].score = maxCorrect
                            break

                self.model = models

        return maxCorrect


    def fitIndividual(self, NormMean, samples, windows, i):
        uniqueLabels = np.unique(samples["Labels"])
        model = BossVSModel(NormMean, windows[i])
        bossvs = BOSSVS(self.maxF, self.maxS, windows[i], NormMean, logger = self.logger)
        words = bossvs.createWords(samples)

        f = self.minF
        keep_going = True
        while (keep_going) & (f <= min(windows[i], self.maxF)):
            bag = bossvs.createBagOfPattern(words, samples, f)
            correct = 0
            for s in range(self.folds):
                idf = bossvs.createTfIdf(bag, self.train_indices[s], uniqueLabels, samples["Labels"])
                correct += self.predict(self.test_indices[s], bag, idf, samples["Labels"]).correct

            if correct > model.score:
                model.score = correct
                model.features = f
            if correct == samples["Samples"]:
                keep_going = False
            f += 2

        bag = bossvs.createBagOfPattern(words, samples, model.features)
        model.idf = bossvs.createTfIdf(bag, [i for i in range(samples["Samples"])], uniqueLabels, samples["Labels"])
        model.bossvs = bossvs

        self.logger.Log("Correct for Norm=%s & Window=%s: %s @ f=%s" % (NormMean, windows[i], model.score, model.features))
        self.results.append(model)


    def fitEnsemble(self, windows, normMean, samples):
        correctTraining = 0
        self.results = []

        self.logger.Log("%s  Fitting for a norm of %s" % (self.NAME, str(normMean)))
        Parallel(n_jobs=1, backend="threading")(delayed(self.fitIndividual)(normMean, samples, windows, i) for i in range(len(windows)))

        # Find best correctTraining
        for i in range(len(self.results)):
            if self.results[i].score > correctTraining:
                correctTraining = self.results[i].score

        # Remove Results that are no longer satisfactory
        new_results = []
        self.logger.Log("Stored Models for Norm=%s" % normMean)
        for i in range(len(self.results)):
            if self.results[i].score >= (correctTraining * self.factor):
                self.logger.Log("WindowLength:%s  Features:%s  TrainScore:%s" % (self.results[i].windowLength, self.results[i].features, self.results[i].score))
                new_results.append(self.results[i])

        return new_results


    def predict(self, indices, bagOfPatternsTestSamples, matrixTrain, labels):
        unique_labels = list(np.unique(labels))
        unique_labels.sort()
        pred_labels = [None for _ in range(len(indices))]
        correct = 0
        for x, i in enumerate(indices):
            bestDistance = 0.0
            for key in unique_labels: # matrixTrain.keys():
                label = key
                stat = matrixTrain[key]
                distance = 0.0
                for key2, value2 in bagOfPatternsTestSamples[i].bob.items():
                    Value = stat[key2] if key2 in stat.keys() else 0.
                    distance += value2 * (Value + 1.0)

                #No mag normal option
                if distance > bestDistance:
                    # print(i, " ", label, " ", distance)
                    bestDistance = distance
                    pred_labels[x] = label

            if pred_labels[x] == bagOfPatternsTestSamples[i].label:
                correct += 1

        return Predictions(correct, pred_labels)


    def prediction(self, model, samples):
        uniqueLabels = np.unique(samples["Labels"])
        pred_labels = [[None for _ in range(len(model))] for _ in range(samples['Samples'])]
        predictedLabels = [None for _ in range(samples["Samples"])]
        indicesTest = [i for i in range(samples["Samples"])]

        for i, score in enumerate(model):
            bossvs = score.bossvs
            wordsTest = bossvs.createWords(samples)
            bagTest = bossvs.createBagOfPattern(wordsTest, samples, score.features)

            p = self.predict(indicesTest, bagTest, score.idf, samples["Labels"])
            for j in range(len(p.labels)):
                pred_labels[j][i] = {p.labels[j] : score.score}


        maxCounts = [None for _ in range(samples['Samples'])]
        for i in range(len(pred_labels)):
            counts = {l:0 for l in uniqueLabels}
            for k in pred_labels[i]:
                if (k != None) and (list(k.keys())[0] != None):
                    label = list(k.keys())[0]
                    count = counts[label] if label in counts.keys() else 0
                    increment = list(k.values())[0] if self.ENSEMBLE_WEIGHTS else 1
                    count = increment if (count == None) else count + increment
                    counts[label] = count


            maxCount = -1
            for e in uniqueLabels:# counts.keys():
                if (predictedLabels[i] == None) or (maxCount < counts[e]) or (maxCount == counts[e]) and (predictedLabels[i] <= e):
                    maxCount = counts[e]
                    predictedLabels[i] = e

        correct = sum([predictedLabels[i] == samples[i].label for i in range(samples["Samples"])])
        return Predictions(correct, predictedLabels)



def getMax(samples, maxWindowSize):
    m = 0
    for i in range(samples['Samples']):
        m = max(len(samples[i].data), m)

    return min(maxWindowSize, m)


class BossVSModel():
    def __init__(self, normed, windowLength):
        self.NAME = "BOSSVS"
        self.score = 0
        self.features = 0
        self.norm = normed
        self.windowLength = windowLength
        self.bossvs = None
        self.idf = None


class Predictions():
    def __init__(self, correct, labels):
        self.correct = correct
        self.labels = labels
