from  src.transformation.SFA import *
import random
import math


class BOSSVS():

    def __init__(self, maxF, maxS, windowLength, normMean):
        self.maxF = maxF
        self.symbols = maxS
        self.windowLength = windowLength
        self.normMean = normMean
        self.signature = None


    def createWords(self, samples):
        if self.signature == None:
            self.signature = SFA("EQUI_DEPTH")
            self.signature.fitWindowing(samples, self.windowLength, self.maxF, self.symbols, self.normMean,True)

        words = []
        for i in range(samples["Samples"]):
            sfaWords = self.signature.transformWindowing(samples[i])
            words_small = []
            for word in sfaWords:
                words_small.append(self.createWord(word, self.maxF, int2byte(self.symbols)))
            words.append(words_small)

        return words


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


    def createBagOfPattern(self, words, samples, f):
        bagOfPatterns = []
        usedBits = int2byte(self.symbols)
        mask = (1 << (usedBits * f)) - 1

        for j in range(len(words)):
            BOP = {}
            lastWord = -9223372036854775808
            for offset in range(len(words[j])):
                word = words[j][offset] & mask
                if word != lastWord:
                    if word in BOP.keys():
                        BOP[word] += 1
                    else:
                        BOP[word] = 1
                lastWord = word
            bagOfPatterns.append(BOP)
        return bagOfPatterns


    def createTfIdf(self, bagOfPatterns, sampleIndices, uniqueLabels, labels):
        matrix = {}
        for label in uniqueLabels:
            matrix[label] = {}

        for j in sampleIndices:
            label = labels[j]
            for key, value in bagOfPatterns[j].items():
                matrix[label][key] = matrix[label][key] + value if key in matrix[label].keys() else value

        wordInClassFreq = {}
        for key, value in matrix.items():
            for key2, value2 in matrix[key].items():
                wordInClassFreq[key2] = wordInClassFreq[key2] + 1 if key2 in wordInClassFreq.keys() else 1

        for key, value in matrix.items():
            tfIDFs = matrix[key]
            for key2, value2 in tfIDFs.items():
                wordCount = wordInClassFreq.get(key2)
                if (value2 > 0) & (len(uniqueLabels) != wordCount):
                    tfValue = 1. + math.log10(value2)
                    idfValue = math.log10(1. + len(uniqueLabels) / wordCount)
                    tfIdf = tfValue / idfValue
                    tfIDFs[key2] = tfIdf
                else:
                    tfIDFs[key2] = 0.
            matrix[key] = tfIDFs

        matrix = self.normalizeTfIdf(matrix)
        return matrix


    def normalizeTfIdf(self, classStatistics):
        for key, values in classStatistics.items():
            squareSum = 0.
            for key2, value2 in classStatistics[key].items():
                squareSum += value2 ** 2
            squareRoot = math.sqrt(squareSum)
            if squareRoot > 0:
                for key2, value2 in classStatistics[key].items():
                    classStatistics[key][key2] /= squareRoot
        return classStatistics








