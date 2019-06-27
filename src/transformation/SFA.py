from  src.transformation.MFT import *
from src.timeseries.TimeSeries import *
from src.timeseries.TimeSeries import TimeSeries
import math
import pandas as pd
import numpy as np

'''
 Symbolic Fourier Approximation as published in
 Schäfer, P., Högqvist, M.: SFA: a symbolic fourier approximation and
 index for similarity search in high dimensional datasets.
 In: EDBT, ACM (2012)
'''

class SFA():

    def __init__(self, histogram_type, LB = True, logger = None, mftUseMaxOrMin = False):
        self.initialized = False
        self.HistogramType = histogram_type
        self.lowerBounding = LB
        self.mftUseMaxOrMin = mftUseMaxOrMin

        logger.Log(self.__dict__, level = 0)
        self.logger = logger

    def initialize(self, wordLength, symbols, normMean):
        self.initialized = True
        self.wordLength = wordLength
        self.maxWordLength = wordLength
        self.symbols = symbols
        self.normMean = normMean
        self.alphabetSize = symbols
        self.transformation = None

        self.orderLine = []
        self.bins = pd.DataFrame(np.zeros((wordLength, self.alphabetSize))) + math.inf
        self.bins.iloc[:, 0] = -math.inf


    def printBins(self, logger = None):
        if logger != None:
            logger.Log(str(self.bins), level = 0)
        else:
            print(self.bins)


    def mv_fitWindowing(self, samples, windowSize, wordLength, symbols, normMean, lowerBounding, dim):
        sa = {}
        index = 0
        for i in range(samples["Samples"]):
            new_list = getDisjointSequences(samples[i][dim], windowSize, normMean)
            for j in range(len(new_list)):
                sa[index] = new_list[j]
                index += 1

        sa["Samples"] = index
        self.fitWindowing(sa, windowSize, wordLength, symbols, normMean, lowerBounding)


    def fitWindowing(self, samples, windowSize, wordLength, symbols, normMean, lowerBounding):
        self.transformation = MFT(windowSize, normMean, lowerBounding)
        sa = {}
        index = 0

        for i in range(samples["Samples"]):
            new_list = getDisjointSequences(samples[i], windowSize, normMean)
            for j in range(len(new_list)):
                sa[index] = new_list[j]
                index += 1

        sa["Samples"] = index
        self.fitTransform(sa, wordLength, symbols, normMean)


    def fitTransform(self, samples, wordLength, symbols, normMean):
        return self.transform(samples, self.fitTransformDouble(samples, wordLength, symbols, normMean))


    def transform(self, samples, approximate):
        transformed = []
        for i in range(samples["Samples"]):
            transformed.append(self.transform2(samples[i].data, approximate[i]))
        return transformed


    def transform2(self, series, one_approx, str_return = False):
        if one_approx == "null":
            one_approx = self.transformation.transform(series, self.maxWordLength)

        if str_return:
            return sfaToWord(self.quantization(one_approx))
        else:
            return self.quantization(one_approx)


    def transformWindowing(self, series, str_return = False):
        mft = self.transformation.transformWindowing(series, self.maxWordLength)

        words = []
        for i in range(len(mft)):
            words.append(self.quantization(mft[i]))

        if str_return:
            return sfaToWordList(words)
        else:
            return words


    def transformWindowingInt(self, series, wordLength):
        words = self.transformWindowing(series)
        intWords = []
        for i in range(len(words)):
            intWords.append(self.createWord(words[i], wordLength, self.int2byte(self.alphabetSize)))
        return intWords


    def fitTransformDouble(self, samples, wordLength, symbols, normMean):
        if self.initialized == False:
            self.initialize(wordLength, symbols, normMean)
            if self.transformation == None:
                self.transformation = MFT(len(samples[0].data), normMean, self.lowerBounding)

        transformedSamples = self.fillOrderline(samples, wordLength)

        if self.HistogramType == "EQUI_DEPTH":
            self.divideEquiDepthHistogram()
        elif self.HistogramType == "EQUI_FREQUENCY":
            self.divideEquiWidthHistogram()
        elif self.HistogramType == "INFORMATION_GAIN":
            self.divideHistogramInformationGain()

        self.orderLine = []

        return transformedSamples


    def fillOrderline(self, samples, wordLength):
        self.orderLine = [[None for _ in range(samples["Samples"])] for _ in range(wordLength)]

        transformedSamples = []
        for i in range(samples["Samples"]):
            transformedSamples_small = self.transformation.transform(samples[i].data, wordLength)
            transformedSamples.append(transformedSamples_small)
            for j in range(len(transformedSamples_small)):
                value = float(format(round(transformedSamples_small[j],2), '.2f')) + 0 #is a bad way of removing values of -0.0
                obj = (value, samples[i].label)
                self.orderLine[j][i] = obj

        for l, list in enumerate(self.orderLine):
            del_list = list
            new_list = []
            while len(del_list) != 0:
                current_min_value = math.inf
                current_min_location = -1
                label = -math.inf
                for j in range(len(del_list)):
                    if (del_list[j][0] < current_min_value) | ((del_list[j][0] == current_min_value) & (del_list[j][1] < label)):
                        current_min_value = del_list[j][0]
                        label = del_list[j][1]
                        current_min_location = j
                new_list.append(del_list[current_min_location])
                del del_list[current_min_location]
            self.orderLine[l] = new_list

        return transformedSamples


    def quantization(self, one_approx):
        i = 0
        word = [0 for _ in range(len(one_approx))]
        for v in one_approx:
            c = 0
            for C in range(self.bins.shape[1]):
                if v < self.bins.iloc[i,c]:
                    break
                else:
                    c += 1
            word[i] = c-1
            i += 1
        return word


    def divideEquiDepthHistogram(self):
        for i in range(self.bins.shape[0]):
            depth = len(self.orderLine[i]) / self.alphabetSize
            try:
                depth = len(self.orderLine[i]) / self.alphabetSize
            except:
                depth = 0

            pos = 0
            count = 0
            try:
                for j in range(len(self.orderLine[i])):
                    count += 1
                    condition1 = count > math.ceil(depth * (pos))
                    condition2 = pos == 1
                    condition3 = self.bins.iloc[i, pos - 1] != self.orderLine[i][j][0]
                    if (condition1) & (condition2 | condition3):
                        self.bins.iloc[i, pos] = round(self.orderLine[i][j][0],2)
                        pos += 1
            except:
                pass
        self.bins.iloc[:, 0] = -math.inf


    def divideEquiWidthHistogram(self):
        i = 0
        for element in self.orderLine:
            if len(element) != 0:
                first = element[0][0]
                last = element[-1][0]
                intervalWidth = (last - first) / self.alphabetSize

                for c in range(self.alphabetSize-1):
                    self.bins.iloc[i,c+1] = intervalWidth * (c + 1) + first
            i += 1

        self.bins.iloc[:, 0] = -math.inf


    def divideHistogramInformationGain(self):
        for i, element in enumerate(self.orderLine):
            self.splitPoints = []
            self.findBestSplit(element, 0, len(element), self.alphabetSize)
            self.splitPoints.sort()
            for j in range(len(self.splitPoints)):
                self.bins.iloc[i, j + 1] = element[self.splitPoints[j]+1][0]


    def findBestSplit(self, element, start, end, remainingSymbols):
        bestGain = -1
        bestPos = -1
        total = end - start

        self.cOut = {}
        self.cIn = {}

        for pos in range(start, end):
            label = element[pos][1]
            self.cOut[label] = self.cOut[label] + 1. if label in self.cOut.keys() else 1.

        class_entropy = self.entropy(self.cOut, total)
        i = start
        lastLabel = element[i][1]
        i += self.moveElement(lastLabel)

        for split in range(start+1, end-1):
            label = element[i][1]
            i += self.moveElement(label)
            if label != lastLabel:
                gain = self.calculateInformationGain(class_entropy, i, total)
                if gain >= bestGain:
                    bestGain = gain
                    bestPos = split
            lastLabel = label

        if bestPos > -1:
            self.splitPoints.append(bestPos)
            remainingSymbols = remainingSymbols / 2
            if remainingSymbols > 1:
                if (bestPos - start > 2) & (end - bestPos > 2):
                    self.findBestSplit(element, start, bestPos, remainingSymbols)
                    self.findBestSplit(element, bestPos, end, remainingSymbols)
                elif end - bestPos > 4:
                    self.findBestSplit(element, bestPos, int((end - bestPos) / 2), remainingSymbols)
                    self.findBestSplit(element, int((end - bestPos) / 2), end, remainingSymbols)
                elif bestPos - start > 4:
                    self.findBestSplit(element, start, int((bestPos-start)/2), remainingSymbols)
                    self.findBestSplit(element, int((bestPos-start)/2), end, remainingSymbols)


    def moveElement(self, label):
        self.cIn[label] = self.cIn[label] + 1. if label in self.cIn.keys() else 1.
        self.cOut[label] = self.cOut[label] - 1. if label in self.cOut.keys() else -1.
        return 1


    def entropy(self, freq, total):
        e = 0
        if total != 0:
            log2 = 1.0 / math.log(2)
            for k in freq.keys():
                p = freq[k] / total
                if p > 0:
                    e += -1 * p * math.log(p) * log2
        else:
            e = math.inf
        return e


    def calculateInformationGain(self, class_entropy, total_c_in, total):
        total_c_out = total - total_c_in
        return class_entropy - total_c_in / total * self.entropy(self.cIn, total_c_in) - total_c_out / total * self.entropy(self.cOut, total_c_out)


    def createWord(self, numbers, maxF, bits):
        shortsPerLong = int(round(60 / bits))
        to = min(len(numbers), maxF)

        b = int(0)
        s = 0
        shiftOffset = 1
        for i in range(s, (min(to, shortsPerLong + s))):
            shift = 1
            for j in range(bits):
                if (numbers[i] & shift) != 0:
                    b |= shiftOffset
                shiftOffset <<= 1
                shift <<= 1

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






def sfaToWord(word):
    word_string = ""
    alphabet = "abcdefghijklmnopqrstuv"
    for w in word:
        word_string += alphabet[w]
    return word_string

def sfaToWordList(wordList):
    list_string = ""
    for word in wordList:
        list_string += sfaToWord(word)
        list_string += "; "
    return list_string
