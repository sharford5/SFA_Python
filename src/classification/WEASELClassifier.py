from  src.transformation.WEASEL import *
import pandas as pd
import numpy as np
import random
import pickle

from src.LibLinear.Feature import *
from src.LibLinear.FeatureNode import *
from src.LibLinear.Linear import *
from src.LibLinear.Parameter import *
from src.LibLinear.Problem import *
from src.LibLinear.SolverType import *


'''
The WEASEL (Word ExtrAction for time SEries cLassification) classifier as published in
 Schäfer, P., Leser, U.: Fast and Accurate Time Series
 Classification with WEASEL." CIKM 2017
'''

class WEASELClassifier():

    def __init__(self, FIXED_PARAMETERS, logger):
        self.NAME = FIXED_PARAMETERS['dataset']
        self.train_bool = FIXED_PARAMETERS['train_bool']
        self.score_path = './stored_models/WEASEL_%s_score.p' % (self.NAME)
        self.model_path = './stored_models/WEASEL_%s_model.p' % (self.NAME)
        self.linearmodel_path = './stored_models/WEASEL_%s_linearmodel.p' % (self.NAME)
        self.wordmodel_path = './stored_models/WEASEL_%s_wordmodel.p' % (self.NAME)

        self.maxF = 6
        self.minF = 4
        self.maxS = 4
        self.chi = 2
        self.bias = 1
        self.p = 0.1
        self.iter = 5000
        self.c = 1
        self.solverType = SolverType('L2R_LR_DUAL')
        self.word_model = None
        self.linear_model = None

        self.lowerBounding = False
        self.MIN_WINDOW_LENGTH = 2
        self.MAX_WINDOW_LENGTH = FIXED_PARAMETERS['MAX_WINDOW_LENGTH']
        self.NORMALIZATION = [False, True]
        logger.Log(self.__dict__, level = 0)
        self.logger = logger


    def eval(self, train, test):
        if self.train_bool:
            scores = self.fitWeasel(train)
            pickle.dump(scores, open(self.score_path, 'wb'))
            pickle.dump(self.model, open(self.model_path, 'wb'))
            pickle.dump(self.linear_model, open(self.linearmodel_path, 'wb'))
            pickle.dump(self.word_model, open(self.wordmodel_path, 'wb'))
        else:
            scores = pickle.load(open(self.score_path, 'rb'))
            self.model = pickle.load(open(self.model_path, 'rb'))
            self.linear_model = pickle.load(open(self.linearmodel_path, 'rb'))
            self.word_model = pickle.load(open(self.wordmodel_path, 'rb'))

        scoreTest = self.predictProbabilities(test)
        # acc, labels = self.predict(scores, test)
        return "WEASEL; "+str(round(scores.train_correct/train['Samples'], 3))+"; "+str(round(scoreTest.acc, 3))


    def fitWeasel(self, samples):
        maxCorrect = -1
        bestF = -1
        bestNorm = False
        keep_going = True
        for normMean in self.NORMALIZATION:
            if keep_going:
                self.windows = self.getWindowLengths(samples, normMean)
                self.logger.Log("Windows: %s" % self.windows)
                model = WEASEL(self.maxF, self.maxS, self.windows, normMean, self.lowerBounding, logger = self.logger)
                words = model.createWORDS(samples)

                f = self.minF
                while (f <= self.maxF) & (keep_going == True):
                    model.dict.reset()
                    bop = model.createBagOfPatterns(words, samples, f)
                    bop = model.filterChiSquared(bop, self.chi)
                    problem = self.initLibLinearProblem(bop, model.dict, self.bias)
                    correct = self.trainLibLinear(problem, 10)
                    print(correct)

                    if correct > maxCorrect:
                        self.logger.Log("New Best Correct at Norm=%s and F=%s of: %s" % (normMean, f, correct))
                        maxCorrect = correct
                        bestF = f
                        bestNorm = normMean
                    if correct == samples["Samples"]:
                        keep_going = False

                    f += 2

        self.logger.Log("Best Model: Norm=%s  Features=%s  Correct=%s/%s" % (bestNorm, bestF, maxCorrect, samples['Samples']))
        self.logger.Log("Final Fitting...")
        self.windows = self.getWindowLengths(samples, bestNorm)
        self.word_model = WEASEL(self.maxF, self.maxS, self.windows, bestNorm, self.lowerBounding, logger = self.logger)
        words = self.word_model.createWORDS(samples)
        bop = self.word_model.createBagOfPatterns(words, samples, bestF)
        bop = self.word_model.filterChiSquared(bop, self.chi)
        problem = self.initLibLinearProblem(bop, self.word_model.dict, self.bias)
        param = Parameter(self.solverType, self.c, self.iter, self.p)
        self.model = Linear()
        self.linear_model = self.model.train(problem, param)

        self.bestF = bestF ##
        return WEASELMODEL(bestNorm, bestF, maxCorrect, samples["Samples"], problem.n)


    def predict(self, scores, test):
        self.logger.Log("Test Word Creation...")
        words = self.word_model.createWORDS(test, data = 'Test')
        self.logger.Log("Test Bag Creation...")
        bag = self.word_model.createBagOfPatterns(words, test, scores.f)
        bag = self.word_model.dict.Remap(bag)
        self.logger.Log("Test Prediction...")
        self.features = self.initLibLinear(bag, scores.n_features)

        pred_labels = []
        for f in self.features:
            pred_labels.append(self.model.predict(self.linear_model, f))

        self.logger.Log("Predicted Correct for %s/%s" % (sum([pred_labels[i] == test[i].label for i in range(test["Samples"])]), test["Samples"]))
        acc = sum([pred_labels[i] == test[i].label for i in range(test["Samples"])])/test["Samples"]
        return acc, pred_labels


    def initLibLinearProblem(self, bob, dict, bias):
        problem = Problem()
        problem.bias = bias
        problem.n = dict.size()
        problem.y = [bob[j].label for j in range(len(bob))]
        features = self.initLibLinear(bob, problem.n)
        problem.l = len(features)
        problem.x = features
        return problem


    def initLibLinear(self, bob, max_feature):
        featuresTrain = [None for _ in range(len(bob))]
        for j in range(len(bob)):
            features = []
            bop = bob[j]
            for word_key, word_value in bop.bob.items():
                if (word_value > 0) & (word_key <= max_feature):
                    features.append(FeatureNode(word_key, word_value))

            LIST = [[f.index, f.value] for f in features]
            FRAME = pd.DataFrame(LIST)
            if len(LIST) > 0:
                FRAME = FRAME.sort_values(FRAME.columns[0])

            new_feature = []
            for i in range(FRAME.shape[0]):
                new_feature.append(FeatureNode(FRAME.iloc[i,0], FRAME.iloc[i,1]))

            featuresTrain[j] = new_feature
        return featuresTrain


    def trainLibLinear(self, prob, n_folds = 10):
        param = Parameter(self.solverType, self.c, self.iter, self.p)

        random.seed(1234)
        l = prob.l
        n_folds = l  if n_folds > l else n_folds

        fold_start = [0]
        perm = [i for i in range(l)]
        # random.shuffle(perm)

        for i in range(1, n_folds):
            fold_start.append(int(math.floor(i*l/n_folds)))

        fold_start.append(l)
        correct = 0
        count = 0
        ## 10 fold cross validation of training set
        for i in range(n_folds):
            model = Linear()
            b = fold_start[i]
            e = fold_start[i + 1]

            subprob = Problem
            subprob.bias = prob.bias
            subprob.n = prob.n
            subprob.l = l - (e - b)
            subprob.y = []

            rows = []
            for j in range(b):
                print(j)
                rows.append(perm[j])
                subprob.y.append(prob.y[perm[j]])

            for j in range(e, l):
                print(j)
                rows.append(perm[j])
                subprob.y.append(prob.y[perm[j]])

            subprob.x = [prob.x[j] for j in rows]
            fold_model = model.train(subprob, param)

            fold_x = []
            fold_y = []
            for u in range(b,e):
                print(u)

                fold_x.append(prob.x[perm[u]])
                fold_y.append(prob.y[perm[u]])

            fold_labels = []
            for h in range(len(fold_y)):
                fold_labels.append(model.predict(fold_model, fold_x[h]))

            for u in range(len(fold_y)):
                count += 1
                correct += 1 if fold_y[u] == fold_labels[u] else 0
            print(count, " ", correct)

        return correct


    def predictProbabilities(self, samples):  #REmoved scores as argument
        labels = [None for _ in range(samples['Samples'])]
        probabilities = [[] for _ in range(samples['Samples'])]

        self.logger.Log("Test Word Creation...")
        wordsTest = self.word_model.createWORDS(samples, data = 'Test')
        self.logger.Log("Test Bag Creation...")
        bagTest = self.word_model.createBagOfPatterns(wordsTest, samples, self.bestF) ##, scores.f)

        self.word_model.dict.Remap(bagTest)
        self.logger.Log("Test Prediction...")
        features = self.initLibLinear(bagTest, self.linear_model.nr_feature)

        correct = 0.
        for ind in range(len(features)):
            probabilities[ind] = [0.0 for _ in range(self.linear_model.nr_class)]
            labels[ind] = self.model.predictProbability(self.linear_model, features[ind], probabilities[ind])
            if labels[ind] == samples[ind].label:
                correct += 1

        self.logger.Log("Predicted Correct for %s/%s" % (sum([labels[i] == samples[i].label for i in range(samples["Samples"])]), samples["Samples"]))
        acc = correct/samples['Samples']
        return Predictions(acc, labels, probabilities)


    def getWindowLengths(self, samples, norm):
        mi = max(3,self.MIN_WINDOW_LENGTH) if (norm) and (self.MIN_WINDOW_LENGTH<=2) else self.MIN_WINDOW_LENGTH
        ma = getMax(samples, self.MAX_WINDOW_LENGTH)
        wLengths = [0 for _ in range(ma - mi + 1)]
        a = 0
        for w in range(mi, ma+1):
            wLengths[a] = w
            a += 1

        return wLengths



class WEASELMODEL():
    def __init__(self, norm, f, correct, size, n_features):
        self.norm = norm
        self.f = f
        self.train_correct = correct
        self.train_size = size
        self.n_features = n_features


class Predictions():
    def __init__(self, acc, labels, probabilities = None):
        self.acc = acc
        self.labels = labels
        self.probabilities = probabilities
        self.realLabels = list(np.unique(labels))


def getMax(samples, maxWindowSize):
    m = 0
    for i in range(samples['Samples']):
        m = max(len(samples[i].data), m)

    return min(maxWindowSize, m)
