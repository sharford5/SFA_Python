from  src.transformation.BOSSVS import *
import random
from statistics import mode
import progressbar
from joblib import Parallel, delayed

'''
The Bag-of-SFA-Symbols in Vector Space classifier as published in
 Schäfer, P.: Scalable time series classification. DMKD (2016)
'''

class BOSSVSClassifier():

    def __init__(self, d):
        self.NAME = d
        self.factor = 0.95
        self.maxF = 16
        self.minF = 4
        self.maxS = 4
        self.MAX_WINDOW_LENGTH = 250
        self.folds = 10


    def createFoldIndex(self, l, n_folds):
        random.seed(0)
        fold_index = [0]
        perm = [i for i in range(l)]
        random.shuffle(perm)

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
        self.createFoldIndex(train["Samples"], self.folds)

        correctTraining = self.fit(train)
        train_acc = correctTraining/train["Samples"]

        correctTesting, labels = self.prediction(self.model, test)
        test_acc = correctTesting/test["Samples"]

        return ("BOSSVS; "+str(round(train_acc,3))+"; "+str(round(test_acc,3))), labels


    def fit(self, train, ):
        maxCorrect = -1

        self.minWindowLength = 10
        maxWindowLength = min(self.MAX_WINDOW_LENGTH, train["Size"])

        count = math.sqrt(maxWindowLength)
        distance = (maxWindowLength - self.minWindowLength) / count

        windows = []
        c = self.minWindowLength
        while c <= maxWindowLength:
            windows.append(int(c))
            c += math.floor(distance)

        for normMean in [True, False]:
            model = self.fitEnsemble(windows, normMean, train)
            correct, labels = self.prediction(model, train)

            if maxCorrect <= correct:
                maxCorrect = correct
                self.model = model
        return maxCorrect


    def fitIndividual(self, NormMean, samples, windows, i, bar):
        uniqueLabels = np.unique(samples["Labels"])
        model = {"window": windows[i], "normMean": NormMean, "correctTraining": 0}
        bossvs = BOSSVS(self.maxF, self.maxS, windows[i], NormMean)
        words = bossvs.createWords(samples)

        f = self.minF
        keep_going = True
        while (keep_going) & (f <= min(windows[i], self.maxF)):
            bag = bossvs.createBagOfPattern(words, samples, f)

            correct = 0
            for s in range(self.folds):
                idf = bossvs.createTfIdf(bag, self.train_indices[s], uniqueLabels, samples["Labels"])
                correct += self.predict(self.test_indices[s], bag, idf, samples["Labels"])[0]

            if correct > model["correctTraining"]:
                model["correctTraining"] = correct
                model["f"] = f
            if correct == samples["Samples"]:
                keep_going = False

            f += 2

        bag = bossvs.createBagOfPattern(words, samples, model["f"])
        model["idf"] = bossvs.createTfIdf(bag, [i for i in range(samples["Samples"])], uniqueLabels, samples["Labels"])
        model["bossvs"] = bossvs
        bar.update(i)
        self.results.append(model)


    def fitEnsemble(self, windows, normMean, samples):
        correctTraining = 0
        self.results = []

        print(self.NAME + "  Fitting for a norm of " + str(normMean))
        with progressbar.ProgressBar(max_value=len(windows)) as bar:
            Parallel(n_jobs=4, backend="threading")(delayed(self.fitIndividual, check_pickle=False)(normMean, samples, windows, i, bar) for i in range(len(windows)))
        print()

        # Find best correctTraining
        for i in range(len(self.results)):
            if self.results[i]["correctTraining"] > correctTraining:
                correctTraining = self.results[i]["correctTraining"]

        # Remove Results that are no longer satisfactory
        new_results = []
        for i in range(len(self.results)):
            if self.results[i]["correctTraining"] >= (correctTraining * self.factor):
                new_results.append(self.results[i])

        return new_results


    def predict(self, indices, bagOfPatternsTestSamples, matrixTrain, labels):
        pred_labels = [None for _ in range(len(indices))]
        correct = 0

        for x, i in enumerate(indices):
            bestDistance = 0.
            for key, value in matrixTrain.items():
                label = key
                stat = matrixTrain[key]
                distance = 0.0
                for key2, value2 in bagOfPatternsTestSamples[i].items():
                    Value = stat[key2] if key2 in stat.keys() else 0.
                    distance += value2 * (Value + 1.0)

                #No mag normal option

                if distance > bestDistance:
                    bestDistance = distance
                    pred_labels[x] = label

            if pred_labels[x] == labels[i]:
                correct += 1

        return correct, pred_labels


    def prediction(self, model, samples):
        uniqueLabels = np.unique(samples["Labels"])
        pred_labels = pd.DataFrame(np.zeros((samples["Samples"], len(model))))
        pred_vector = [None for _ in range(samples["Samples"])]
        indicesTest = [i for i in range(samples["Samples"])]

        for i, score in enumerate(model):
            bossvs = score["bossvs"]
            wordsTest = bossvs.createWords(samples)
            bagTest = bossvs.createBagOfPattern(wordsTest, samples, score["f"])

            p = self.predict(indicesTest, bagTest, score["idf"], samples["Labels"])

            for j in range(len(p[1])):
                pred_labels.iloc[j,i] = p[1][j]

        for i in range(samples["Samples"]):
            try:
                pred_vector[i] = mode(pred_labels.iloc[i,:].tolist())
            except: #Guess if there is no favorite
                pred_vector[i] = random.choice(uniqueLabels)

        correct = sum([pred_vector[i] == samples[i].label for i in range(samples["Samples"])])
        return correct, pred_labels




