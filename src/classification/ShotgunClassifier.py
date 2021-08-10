from src.timeseries.TimeSeries import *
from joblib import Parallel, delayed

'''
The Shotgun Classifier as published in:
 Schäfer, P.: Towards time series classification without human preprocessing.
 In Machine Learning and Data Mining in Pattern Recognition,
 pages 228–242. Springer, 2014.
'''

class ShotgunClassifier():

    def __init__(self, FIXED_PARAMETERS, logger):
        self.NAME = FIXED_PARAMETERS['dataset']
        self.factor = 1.
        self.MAX_WINDOW_LENGTH = 250
        self.NORMALIZATION = [True, False]
        logger.Log(self.__dict__, level = 0)
        self.logger = logger


    def eval(self, train, test):
        correctTraining = self.fit(train)
        train_acc = correctTraining/train["Samples"]

        self.logger.Log("Final Model...")
        self.logger.Log("Norm:%s  WindowLength:%s  TrainScore:%s" % (self.model.norm, self.model.window, self.model.correct))

        correctTesting, labels = self.predict(self.model, test, testing = True)
        print(correctTesting)
        test_acc = correctTesting/test["Samples"]
        return "Shotgun; "+str(round(train_acc,3))+"; "+str(round(test_acc,3)), labels


    def fit(self, train):
        bestCorrectTraining = 0

        for normMean in self.NORMALIZATION:
            model, correct = self.fitEnsemble(normMean, train, self.factor)
            if correct > bestCorrectTraining:
                bestCorrectTraining = correct
                self.model = model[0]
        return bestCorrectTraining


    def fitIndividual(self, NormMean, samples, windows, i):
        # self.logger.Log("Window: %s" % windows[i], level = 1 if i%5==0 else 0)
        model = ShotgunModel(NormMean, windows[i], samples, samples["Labels"])
        correct, pred_labels = self.predict(model, samples)
        model.correct = correct
        self.logger.Log("Correct for Norm=%s & Window=%s: %s" % (NormMean, windows[i], model.correct))
        self.results.append(model)


    def fitEnsemble(self, normMean, samples, factor):
        minWindowLength = 5
        maxWindowLength = getMax(samples, self.MAX_WINDOW_LENGTH)
        windows = self.getWindowsBetween(minWindowLength, maxWindowLength)
        self.logger.Log("Windows: %s" % windows)

        correctTraining = 0
        self.results = []

        self.logger.Log("%s  Fitting for a norm of %s" % (self.NAME, str(normMean)))
        Parallel(n_jobs=1, backend="threading")(delayed(self.fitIndividual)(normMean, samples, windows, i) for i in range(len(windows)))

        # Find best correctTraining
        for i in range(len(self.results)):
            if self.results[i].correct > correctTraining:
                correctTraining = self.results[i].correct

        # Remove Results that are no longer satisfactory
        new_results = []
        for i in range(len(self.results)):
            if self.results[i].correct >= (correctTraining * factor):
                new_results.append(self.results[i])

        return new_results, correctTraining


    def predict(self, model, test_samples, testing = False):
        p = [None for _ in range(test_samples["Samples"])]
        means = [None for _ in range(len(model.labels))]
        stds = [None for _ in range(len(model.labels))]
        means, stds = self.calcMeansStds(model.window, model.samples, means, stds, model.norm)


        for i in range(test_samples["Samples"]):
            query = test_samples[i]
            distanceTo1NN = math.inf

            wQueryLen = min(len(query.data), model.window)
            disjointWindows = getDisjointSequences(query, wQueryLen, model.norm)

            for j in range(len(model.labels)):
                ts = model.samples[j].data
                if (ts != query.data) or testing:
                    totalDistance = 0.

                    for q in disjointWindows:
                        resultDistance = distanceTo1NN
                        for w in range(len(ts) - model.window+1):
                            distance = self.getEuclideanDistance(ts, q.data, means[j][w], stds[j][w], resultDistance, w)
                            resultDistance = min(distance, resultDistance)
                        totalDistance += resultDistance
                        if totalDistance > distanceTo1NN:
                            break

                    if totalDistance < distanceTo1NN:
                        p[i] = model.labels[j]
                        distanceTo1NN = totalDistance

        correct = sum([p[i] == test_samples[i].label for i in range(test_samples["Samples"])])
        return correct, p


    def getEuclideanDistance(self, ts, q, meanTs, stdTs, minValue, w):
        distance = 0.0
        for ww in range(len(q)):
            value1 = (ts[w + ww] - meanTs) * stdTs
            value = q[ww] - value1
            distance += value * value

            if distance >= minValue:
                return math.inf
        return distance


    def calcMeansStds(self, windowLength, trainSamples, means, stds, normMean):
        for i in range(trainSamples["Samples"]):
            w = min(windowLength, trainSamples["Size"])
            means[i] = [None for _ in range(trainSamples["Size"] - w + 1)]
            stds[i] = [None for _ in range(trainSamples["Size"] - w + 1)]
            means[i], stds[i] = self.calcIncreamentalMeanStddev(w, trainSamples[i].data, means[i], stds[i])
            for j in range(len(stds[i])):
                stds[i][j] = 1.0 / stds[i][j] if stds[i][j] > 0 else 1.0
                means[i][j] = means[i][j] if normMean else 0
        return means, stds


    def calcIncreamentalMeanStddev(self, windowLength, series, MEANS, STDS):
        SUM = 0.
        squareSum = 0.

        rWindowLength = 1.0 / windowLength
        for ww in range(windowLength):
            SUM += series[ww]
            squareSum += series[ww] * series[ww]
        MEANS[0] = SUM * rWindowLength
        buf = squareSum * rWindowLength - MEANS[0] * MEANS[0]

        STDS[0] = np.sqrt(buf) if buf > 0 else 0

        for w in range(1, (len(series) - windowLength + 1)):
            SUM += series[w + windowLength - 1] - series[w - 1]
            MEANS[w] = SUM * rWindowLength

            squareSum += series[w + windowLength - 1] * series[w + windowLength - 1] - series[w - 1] * series[w - 1]
            buf = squareSum * rWindowLength - MEANS[w] * MEANS[w]
            STDS[w] = np.sqrt(buf) if buf > 0 else 0

        return MEANS, STDS


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


class ShotgunModel():
    def __init__(self, norm, w, samples, labels):
        self.norm = norm
        self.window = w
        self.samples = samples
        self.labels = labels
        self.correct = None
