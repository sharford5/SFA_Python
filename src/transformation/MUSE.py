from  src.transformation.SFA import *
from joblib import Parallel, delayed

class MUSE():

    def __init__(self, maxF, maxS, histogramType, windowLengths, normMean, lowerBounding, logger = None):
        self.maxF = maxF + maxF % 2
        self.alphabetSize = maxS
        self.windowLengths = windowLengths
        self.normMean = normMean
        self.lowerBounding = lowerBounding
        self.dict = Dictionary()
        self.signature = [None for _ in range(len(self.windowLengths))]
        self.histogramType = histogramType
        self.BIGRAM=True
        logger.Log(self.__dict__, level = 0)
        self.logger = logger

    def createWORDS(self, samples, data = 'Train'):
        self.words = [None for _ in range(len(self.windowLengths))]
        Parallel(n_jobs=1, backend="threading")(delayed(self.createWords)(samples, w, data) for w in range(len(self.windowLengths)))
        return self.words


    def createWords(self, samples, index, data):
        if self.signature[index] == None:
            self.signature[index] = [None for _ in range(samples['Dimensions'])]
            for i in range(samples['Dimensions']):
                self.signature[index][i] = SFA(self.histogramType, self.lowerBounding, logger = self.logger, mftUseMaxOrMin=False)
                self.signature[index][i].mv_fitWindowing(samples, self.windowLengths[index], self.maxF, self.alphabetSize, self.normMean, self.lowerBounding, dim = i)
                self.signature[index][i].printBins(self.logger)

        words = []
        for m in range(samples["Samples"]):
            for n in range(samples["Dimensions"]):
                if len(samples[m][n].data) >= self.windowLengths[index]:
                    words.append(self.signature[index][n].transformWindowingInt(samples[m][n], self.maxF))
                else:
                    words.append([])

        self.logger.Log("Generating %s Words for Norm=%s and Window=%s" % (data, self.normMean, self.windowLengths[index]))
        self.words[index] = words


    def createBagOfPatterns(self, words, samples, dimensionality, f):
        bagOfPatterns = []

        usedBits = int2byte(self.alphabetSize)
        mask = (1 << (usedBits * f)) - 1

        j = 0
        for i in range(samples["Samples"]):
            bop = BagOfBigrams(samples[i][0].label)

            for w in range(len(self.windowLengths)):
                if self.windowLengths[w] >= f:
                    for dim in range(dimensionality):
                        for offset in range(len(words[w][j+dim])):
                            word = MuseWord(w, dim, words[w][j + dim][offset] & mask, 0)
                            dict = self.dict.getWord(word)
                            bop.bob[dict] = bop.bob[dict] + 1 if dict in bop.bob.keys() else 1
                            print(words[w][j + dim][offset], "-- ", word.w, "; ", word.dim, "; ", word.word)

                            if (self.windowLengths[len(self.windowLengths)-1] < 200) and (self.BIGRAM) and (offset - self.windowLengths[w] >= 0):
                                bigram = MuseWord(w, dim, (words[w][j + dim][offset - self.windowLengths[w]] & mask), words[w][j + dim][offset] & mask)
                                newWord = self.dict.getWord(bigram)
                                bop.bob[newWord] = bop.bob[newWord] + 1 if newWord in bop.bob.keys() else 1
                                print(bigram.w, "; ", bigram.dim, "; ", bigram.word, "; ", bigram.word2)

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

class MuseWord():
    def __init__(self, w, dim, word, word2):
        self.w = w
        self.dim = dim
        self.word = word
        self.word2 = word2

class Dictionary():

    def __init__(self):
        self.dict = {}
        self.dictChi = {}
        self.hasEmptyKey = False
        self.mask = 7
        self.keys = []


    def reset(self):
        self.dict = {}
        self.dictChi = {}


    # def getWord(self, word):
    #     word2 = 0
    #     if word in self.dict.keys():
    #         word2 = self.dict[word]
    #     else:
    #         word2 = len(self.dict.keys()) + 1
    #         self.dict[word] = word2
    #     return word2

    def getWord(self, word):
        index = self.indexOf(word)
        newWord = -1
        if index > -1:
            newWord = self.indexGet(index)
        else:
            newWord = self.dict.size() + 1
            self.dict.put(word, newWord)

        return newWord


    def indexOf(self, key):
        mask = self.mask
        if key == None:
            return mask + 1 if self.hasEmptyKey else ~(mask + 1)
        else:
            keys = self.keys
            slot = hash(key) & mask

            #KType existing;
            existing = keys[slot] if len(keys) >= slot else None
            while not (existing == None):
                if self.equals(existing,  key):
                    return slot

                slot = (slot + 1) & mask

            return ~slot


    def equals(self, v1, v2):
        return (v1 == v2) or ((v1 != None) and



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




