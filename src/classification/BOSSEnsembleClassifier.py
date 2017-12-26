from  src.transformation.BOSS import *
import progressbar
from joblib import Parallel, delayed

'''
The Bag-of-SFA-Symbols Ensemble Classifier as published in
 Schäfer, P.: The boss is concerned with time series classification
 in the presence of noise. DMKD (2015)
'''

class BOSSEnsembleClassifier():

    def __init__(self, d):
        self.NAME = d
        self.factor = 0.96
        self.maxF = 16
        self.minF = 6
        self.maxS = 4
        self.MAX_WINDOW_LENGTH = 250


    def eval(self, train, test):
        scores = self.fit(train)

        labels, correctTesting = self.predict(self.model, test)
        test_acc = correctTesting/test["Samples"]

        return "BOSS Ensemble; "+str(round(scores,3))+"; "+str(round(test_acc,3)), labels


    def fit(self, train):
        self.minWindowLength = 10
        maxWindowLength = self.MAX_WINDOW_LENGTH
        for i in range(train["Samples"]):
            maxWindowLength = min([len(train[i].data), maxWindowLength])

        self.windows = range(maxWindowLength, self.minWindowLength, -1)

        NORMALIZATION = [True, False]
        bestCorrectTraining = 0.
        bestScore = None

        for norm in NORMALIZATION:
            models, correctTraining = self.fitEnsemble(norm, train)
            labels, correctTesting = self.predict(models, train)

            if bestCorrectTraining < correctTesting:
                bestCorrectTraining = correctTesting
                bestScore = correctTesting/train["Samples"]
                self.model = models

        return bestScore

    def fitIndividual(self, NormMean, samples, i, bar):
        model = self.BOSSModel(NormMean, self.windows[i])

        boss = BOSS(self.maxF, self.maxS, self.windows[i], NormMean)
        train_words = boss.createWords(samples)

        f = self.minF
        keep_going = True
        while (f <= self.maxF) & (keep_going == True):
            bag = boss.createBagOfPattern(train_words, samples, f)
            s = self.prediction(bag, bag, samples["Labels"], samples["Labels"], False)
            if s[0] > model[1]:
                model[1] = s[0]
                model[2] = f
                model[3] = boss
                model[5] = bag
                model[6] = samples["Labels"]
            if s[0] == samples["Samples"]:
                keep_going = False
            f += 2

        self.results.append(model)
        bar.update(i)

    def fitEnsemble(self, NormMean, samples):
        correctTraining = 0
        self.results = []

        print(self.NAME + "  Fitting for a norm of " + str(NormMean))
        with progressbar.ProgressBar(max_value=len(self.windows)) as bar:
            Parallel(n_jobs=3, backend="threading")(delayed(self.fitIndividual, check_pickle=False)(NormMean, samples, i, bar) for i in range(len(self.windows)))
        print()

        #Find best correctTraining
        for i in range(len(self.results)):
            if self.results[i][1] > correctTraining:
                correctTraining = self.results[i][1]

        # Remove Results that are no longer satisfactory
        new_results = []
        for i in range(len(self.results)):
            if self.results[i][1] >= (correctTraining * self.factor):
                new_results.append(self.results[i])

        return new_results, correctTraining


    def BossScore(self, normed, windowLength):
        return ["BOSS Ensemble", 0, 0, normed, windowLength, pd.DataFrame(), 0]


    def BOSSModel(self, normed, windowLength):
        return self.BossScore(normed, windowLength)


    def prediction(self, bag_test, bag_train, label_test, label_train, training_check):
        p_labels = [0 for i in range(len(label_test))]
        p_correct = 0

        for i in range(len(bag_test)):
            minDistance = 2147483647
            p_labels[i] = 'Nan'

            noMatchDistance = 0
            for key in bag_test[i].keys():
                noMatchDistance += bag_test[i][key] ** 2

            for j in range(len(bag_train)):
                if (bag_train[j] != bag_test[i]) | (training_check):
                    distance = 0
                    for key in bag_test[i].keys():
                        buf = bag_test[i][key] - bag_train[j][key] if key in bag_train[j].keys() else bag_test[i][key]
                        distance += buf ** 2

                        if distance >= minDistance:
                            continue

                    if (distance != noMatchDistance) & (distance < minDistance):
                        minDistance = distance
                        p_labels[i] = label_train[j]


            if label_test[i] == p_labels[i]:
                p_correct += 1

        return p_correct, p_labels


    def predict(self, models, samples):
        Label_Matrix = pd.DataFrame(np.zeros((samples["Samples"], len(models))))
        Label_Vector = pd.DataFrame(np.zeros((samples["Samples"])))

        for i, model in enumerate(models):
            wordsTest = model[3].createWords(samples)

            test_bag = model[3].createBagOfPattern(wordsTest, samples, model[2])
            p_correct, p_labels = self.prediction(test_bag, model[5], samples["Labels"], model[6], True)

            for j in range(len(p_labels)):
                Label_Matrix.loc[j, i] = p_labels[j]

        unique_labels = np.unique(samples["Labels"])

        for i in range(len(Label_Vector)):
            maximum = 0
            best = 0
            d = Label_Matrix.iloc[i, :].tolist()
            for key in unique_labels:
                if d.count(key) > maximum:
                    maximum = d.count(key)
                    best = key
            Label_Vector.iloc[i] = best

        correctTesting = 0
        for i in range(len(Label_Vector)):
            if int(Label_Vector.iloc[i]) == int(samples["Labels"][i]):
                correctTesting += 1

        return Label_Vector, correctTesting



