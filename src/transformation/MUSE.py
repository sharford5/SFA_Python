from  src.transformation.SFA import *
import progressbar
from joblib import Parallel, delayed

global MAX_WINDOW_LENGTH
MAX_WINDOW_LENGTH = 450

class MUSE():

    def __init__(self, maxF, maxS, histogramType, windowLengths, normMean, lowerBounding):
        self.maxF = maxF + maxF % 2
        self.alphabetSize = maxS
        self.windowLengths = windowLengths
        self.normMean = normMean
        self.lowerBounding = lowerBounding
        self.dict = Dictionary()
        self.signature = [None for _ in range(len(self.windowLengths))]
        self.histogramType = histogramType


    def createWORDS(self, samples):
        self.words = [None for _ in range(len(self.windowLengths))]

        print("Fitting ")
        with progressbar.ProgressBar(max_value=len(self.windowLengths)) as bar:
            Parallel(n_jobs=4, backend="threading")(delayed(self.createWords, check_pickle=False)(samples, w, bar) for w in range(len(self.windowLengths)))
        print()

        return self.words

    def createWords(self, samples, index, bar = None):
        if self.signature[index] == None:
            self.signature[index] = SFA(self.histogramType, False, self.lowerBounding, False)
            self.signature[index].mv_fitWindowing(samples, self.windowLengths[index], self.maxF, self.alphabetSize, self.normMean, False)
            # self.signature[index].printBins()

        words = []
        for m in range(samples["Samples"]):
            for n in range(samples["Dimensions"]):
                if len(samples[m][n].data) >= self.windowLengths[index]:
                    words.append(self.signature[index].transformWindowingInt(samples[m][n], self.maxF))
                else:
                    words.append([])

        self.words[index] = words

        if bar != None:
            bar.update(index)


    def createBagOfPatterns(self, words, samples, dimensionality, f):
        bagOfPatterns = []

        usedBits = int2byte(self.alphabetSize)
        mask = (1 << (usedBits * f)) - 1

        j = 0
        for dim in range(samples["Samples"]):
            bop = BagOfBigrams(samples[dim][0].label)
            for w in range(len(self.windowLengths)):
                if self.windowLengths[0] >= f:
                    for d in range(dimensionality):
                        dLabel = str(d)
                        for offset in range(len(words[w][j+d])):
                            word = str(w)+"_"+str(int(dLabel))+"_"+str(words[w][j + d][offset] & mask)
                            dict = self.dict.getWord(word)
                            bop.bob[dict] = bop.bob[dict] + 1 if dict in bop.bob.keys() else 1

                            if offset - self.windowLengths[w] >= 0:
                                prevWord = str(w)+"_"+str(int(dLabel))+"_"+ str(((words[w][j + d][offset - self.windowLengths[w]] & mask)))
                                newWord = self.dict.getWord(prevWord + "_" + word)

                                bop.bob[newWord] = bop.bob[newWord] + 1 if newWord in bop.bob.keys() else 1
            bagOfPatterns.append(bop)
            j += dimensionality
        return bagOfPatterns




    def filterChiSquared(self, bob, chi_limit):
        classFrequencies = {}
        for list in bob:
            label = list.label
            classFrequencies[label] = classFrequencies[label] + 1 if label in classFrequencies.keys() else 1

        featureCount = {}
        classProb = {}
        observed = {}
        chiSquare = {}

        # count number of samples with this word
        for bagOfPattern in bob:
            label = bagOfPattern.label
            bag_dict = bagOfPattern.bob
            for key in bag_dict.keys():
                if bag_dict[key] > 0:
                    featureCount[key] = featureCount[key] + 1 if key in featureCount.keys() else 1
                    key2 = label << 32 | key
                    observed[key2] = observed[key2] + 1 if key2 in observed.keys() else 1

        # samples per class
        for list in bob:
            label = list.label
            classProb[label] = classProb[label] + 1 if label in classProb.keys() else 1

        # chi square: observed minus expected occurence
        for prob_key, prob_value in classProb.items():
            prob_value /= len(bob)
            for feature_key, feature_value in featureCount.items():
                key = prob_key << 32 | feature_key
                expected = prob_value * feature_value
                chi = get(observed, key) - expected
                newChi = chi * chi / expected

                if (newChi >= chi_limit) & (newChi > get(chiSquare, feature_key)):
                    chiSquare[feature_key] = newChi

        #best elements above limit
        for j in range(len(bob)):
            for key, _ in bob[j].bob.items():
                if get(chiSquare, key) < chi_limit:
                    bob[j].bob[key] = 0

        bob = self.dict.Remap(bob)
        return bob




class BagOfBigrams():
    def __init__(self, label):
        self.bob = {}
        self.label = int(label)


class Dictionary():

    def __init__(self):
        self.dict = {}
        self.dictChi = {}


    def reset(self):
        self.dict = {}
        self.dictChi = {}


    def getWord(self, word):
        word2 = 0
        if word in self.dict.keys():
            word2 = self.dict[word]
        else:
            word2 = len(self.dict.keys()) + 1
            self.dict[word] = word2
        return word2


    def getWordChi(self, word):
        word2 = 0
        if word in self.dictChi.keys():
            word2 = self.dictChi[word]
        else:
            word2 = len(self.dictChi.keys()) + 1
            self.dictChi[word] = word2
        return word2


    def size(self):
        if len(self.dictChi) != 0:
            return len(self.dictChi)+1
        else:
            return len(self.dict)


    def Remap(self, bagOfPatterns):
        for j in range(len(bagOfPatterns)):
            oldMap = bagOfPatterns[j].bob
            bagOfPatterns[j].bob = {}
            for word_key, word_value in oldMap.items():
                if word_value > 0:
                    bagOfPatterns[j].bob[self.getWordChi(word_key)] = word_value

        return bagOfPatterns


def get(dictionary, key):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return 0