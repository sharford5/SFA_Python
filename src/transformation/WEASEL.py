from  src.transformation.SFA import *
import progressbar
from joblib import Parallel, delayed

global MAX_WINDOW_LENGTH
MAX_WINDOW_LENGTH = 250

class WEASEL():

    def __init__(self, maxF, maxS, windowLength, normMean):
        self.maxF = maxF
        self.symbols = maxS
        self.windowLengths = windowLength

        self.normMean = normMean
        self.signature = [None for w in range(len(self.windowLengths))]
        self.dict = Dictionary()


    def createWORDS(self, samples):
        self.words = [None for _ in range(len(self.windowLengths))]

        print("Fitting ")
        with progressbar.ProgressBar(max_value=len(self.windowLengths)) as bar:
            Parallel(n_jobs=4, backend="threading")(delayed(self.createWords, check_pickle=False)(samples, w, bar) for w in range(len(self.windowLengths)))
        print()

        return self.words


    def createWords(self, samples, index, bar = None):
        if self.signature[index] == None:
            self.signature[index] = SFA("INFORMATION_GAIN", True, False)
            self.signature[index].fitWindowing(samples, self.windowLengths[index], self.maxF, self.symbols, self.normMean, False)
            # self.signature[index].printBins()

        words = []
        for i in range(samples["Samples"]):
            words.append(self.signature[index].transformWindowingInt(samples[i], self.maxF))

        self.words[index] = words
        if bar != None:
            bar.update(index)


    def createWord(self, numbers, maxF, bits):
        shortsPerLong = int(round(60 / bits))
        to = min([len(numbers), maxF])

        b = 0
        s = 0
        shiftOffset = 1
        for i in range(s, (min(to, shortsPerLong + s))):
            shift = 1
            for j in range(bits):
                if (numbers[i] & shift) != 0:
                    b |= shiftOffset
                shiftOffset <<= 1
                shift <<= 1

        limit = 2147483647
        total = 2147483647 + 2147483648
        while b > limit:
            b = b - total - 1
        return b


    def int2byte(self, number):
        log = 0
        if (number & 0xffff0000) != 0:
            number >>= 16
            log = 16
        if number >= 256:
            number >>= 8
            log += 8
        if number >= 16:
            number >>= 4
            log += 4
        if number >= 4:
            number >>= 2
            log += 2
        return log + (number >> 1)


    def createBagOfPatterns(self, words, samples, f):
        bagOfPatterns = [BagOfBigrams(samples[j].label) for j in range(samples["Samples"])]
        usedBits = int2byte(self.symbols)
        mask = (1 << (usedBits * f)) - 1
        highestBit = int2byte(MAX_WINDOW_LENGTH)+1

        for j in range(samples["Samples"]):
            for w in range(len(self.windowLengths)):
                for offset in range(len(words[w][j])):
                    word = self.dict.getWord((words[w][j][offset] & mask) << highestBit | w)
                    bagOfPatterns[j].bob[word] = bagOfPatterns[j].bob[word] + 1 if word in bagOfPatterns[j].bob.keys() else 1

                    if offset - self.windowLengths[w] >= 0:
                        prevWord = self.dict.getWord((words[w][j][offset - self.windowLengths[w]] & mask) << highestBit | w)
                        newWord = self.dict.getWord((prevWord << 32 | word ) << highestBit)
                        bagOfPatterns[j].bob[newWord] = bagOfPatterns[j].bob[newWord] + 1 if newWord in bagOfPatterns[j].bob.keys() else 1

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