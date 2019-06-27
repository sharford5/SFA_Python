from  src.transformation.MUSE import *
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
The WEASEL+MUSE classifier as published in

 Schäfer, P., Leser, U.: Multivariate Time Series Classification with WEASEL+MUSE. arXiv 2017
 http://arxiv.org/abs/1711.11343
'''

class MUSEClassifier():

    def __init__(self, FIXED_PARAMETERS, logger):
        self.NAME = FIXED_PARAMETERS['dataset']
        self.train_bool = FIXED_PARAMETERS['train_bool']
        self.score_path = './stored_models/MUSE_%s_score.p' % (self.NAME)
        self.model_path = './stored_models/MUSE_%s_model.p' % (self.NAME)
        self.linearmodel_path = './stored_models/MUSE_%s_linearmodel.p' % (self.NAME)
        self.wordmodel_path = './stored_models/MUSE_%s_wordmodel.p' % (self.NAME)

        self.maxF = 6
        self.minF = 2
        self.maxS = 4
        self.histTypes = ["EQUI_DEPTH", "EQUI_FREQUENCY"]
        self.lowerBounding = False

        self.chi = 2
        self.bias = 1
        self.p = 0.1
        self.iter = 5000
        self.c = 1

        self.MIN_WINDOW_SIZE = 2
        self.MAX_WINDOW_SIZE = 450
        self.solverType = SolverType('L2R_LR_DUAL')
        self.word_model = None
        self.linear_model = None
        self.TIMESERIES_NORM = False
        logger.Log(self.__dict__, level = 0)
        self.logger = logger

    def eval(self, train, test):
        if not self.TIMESERIES_NORM:
            for i in range(train["Samples"]):
                for j in range(train["Dimensions"]):
                    train[i][j].NORM_CHECK = self.TIMESERIES_NORM
            for i in range(test["Samples"]):
                for j in range(test["Dimensions"]):
                    test[i][j].NORM_CHECK = self.TIMESERIES_NORM

        if self.train_bool:
            scores = self.fit(train)
            pickle.dump(scores, open(self.score_path, 'wb'))
            pickle.dump(self.model, open(self.model_path, 'wb'))
            pickle.dump(self.linear_model, open(self.linearmodel_path, 'wb'))
            pickle.dump(self.word_model, open(self.wordmodel_path, 'wb'))
        else:
            scores = pickle.load(open(self.score_path, 'rb'))
            self.model = pickle.load(open(self.model_path, 'rb'))
            self.linear_model = pickle.load(open(self.linearmodel_path, 'rb'))
            self.word_model = pickle.load(open(self.wordmodel_path, 'rb'))

        acc, labels = self.predict(scores, test)

        return "WEASEL+MUSE; "+str(round(scores.train_correct/train["Samples"],3))+"; "+str(round(acc,3)), labels


    def fit(self, trainSamples):
        musemodel = self.fitMuse(trainSamples)
        return musemodel


    def predict(self, scores, test):
        self.logger.Log("Test Word Creation")
        words = self.word_model.createWORDS(test)
        bag = self.word_model.createBagOfPatterns(words, test, test["Dimensions"], scores.f)
        bag = self.word_model.dict.Remap(bag)
        self.features = self.initLibLinear(bag, scores.n_features)

        pred_labels = []
        for f in self.features:
            pred_labels.append(self.model.predict(self.linear_model, f))

        acc = sum([pred_labels[i] == test[i][0].label for i in range(test["Samples"])])/test["Samples"]

        return acc, pred_labels


    def fitMuse(self, samples):
        dimensionality = samples["Dimensions"]

        maxCorrect = -1
        bestF = -1
        bestNorm = False
        bestHistType = None

        breaker = False
        for histType in self.histTypes:
            for normMean in [True, False]:
                self.windowLengths = self.getWindowLengths(samples, normMean)
                self.logger.Log("Fitting for Norm=%s and histType=%s" % (normMean, histType))
                model = MUSE(self.maxF, self.maxS, histogramType=histType, windowLengths=self.windowLengths, normMean=normMean, lowerBounding=self.lowerBounding, logger = self.logger)
                words = model.createWORDS(samples)

                f = self.minF
                while f <= self.maxF:
                    bag = model.createBagOfPatterns(words, samples, dimensionality, f)
                    bag = model.filterChiSquared(bag, self.chi)

                    problem = self.initLibLinearProblem(bag, model.dict, self.bias)
                    correct = self.trainLibLinear(problem, 10)

                    if correct > maxCorrect:
                        self.logger.Log("New Best Correct at Norm=%s and histType=%s and F=%s of: %s" % (normMean, histType, f, correct))
                        maxCorrect = correct
                        bestF = f
                        bestNorm = normMean
                        bestHistType = histType

                    if correct == samples["Samples"]:
                        breaker = True
                        break

                    f += 2
                if breaker:
                    break
            if breaker:
                break

        self.logger.Log("Final Fitting")
        self.word_model = MUSE(bestF, self.maxS, bestHistType, self.windowLengths, bestNorm, True, logger = self.logger)
        words = self.word_model.createWORDS(samples)
        bag = self.word_model.createBagOfPatterns(words, samples, dimensionality, bestF)
        bag = self.word_model.filterChiSquared(bag, self.chi)
        problem = self.initLibLinearProblem(bag, self.word_model.dict, self.bias)

        param = Parameter(self.solverType, self.c, self.iter, self.p)
        self.model = Linear()
        self.linear_model = self.model.train(problem, param)

        return MUSEMODEL(bestNorm, bestHistType, bestF, maxCorrect, samples["Samples"], problem.n)



    def GetMax(self, samples, number):
        m = len(samples[0][0].data)
        for i in range(samples["Samples"]):
            for j in range(len(samples[i].keys())):
                m = max(m, len(samples[i][j].data))
        return min(m, number)


    def getWindowLengths(self, samples, norm):
        min = max(3, self.MIN_WINDOW_SIZE) if (norm) and (self.MIN_WINDOW_SIZE <= 2) else self.MIN_WINDOW_SIZE
        ma = self.GetMax(samples, self.MAX_WINDOW_SIZE)
        windowLengths = []
        for w in range(min, ma+1):
            windowLengths.append(w)
        return windowLengths


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
        random.shuffle(perm)

        for i in range(1, n_folds):
            fold_start.append(int(math.floor(i*l/n_folds)))

        fold_start.append(l)
        correct = 0

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
                rows.append(perm[j])
                subprob.y.append(prob.y[perm[j]])

            for j in range(e, l):
                rows.append(perm[j])
                subprob.y.append(prob.y[perm[j]])

            subprob.x = [prob.x[j] for j in rows]
            fold_model = model.train(subprob, param)

            fold_x = []
            fold_y = []
            for u in range(b,e):
                fold_x.append(prob.x[perm[u]])
                fold_y.append(prob.y[perm[u]])

            fold_labels = []
            for h in range(len(fold_y)):
                fold_labels.append(model.predict(fold_model, fold_x[h]))

            for u in range(len(fold_y)):
                correct += 1 if fold_y[u] == fold_labels[u] else 0


        return correct


class MUSEMODEL():
    def __init__(self, norm, hist, f, correct, size, n_features):
        self.norm = norm
        self.histType = hist
        self.f = f
        self.train_correct = correct
        self.train_size = size
        self.n_features = n_features
