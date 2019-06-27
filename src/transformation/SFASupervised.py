from src.transformation.MFT import *
from src.transformation.SFA import *

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

class SFASupervised():

    def __init__(self, histogram_type, lowerBounding = False, logger = None):
        self.initialized = False
        self.HistogramType = histogram_type
        self.lowerBounding = lowerBounding
        self.MUSE_Bool = False

        logger.Log(self.__dict__, level = 0)
        self.logger = logger

        self.sfa = SFA(histogram_type, LB = self.lowerBounding, logger = self.logger)


    def fitWindowing(self, samples, windowSize, wordLength, symbols, normMean, lowerBounding):
        self.sfa.quantization = self.quantizationSupervised
        self.transformation = MFT(windowSize, normMean, lowerBounding, self.MUSE_Bool)
        sa = {}
        index = 0

        for i in range(samples["Samples"]):
            new_list = getDisjointSequences(samples[i], windowSize, normMean)
            for j in range(len(new_list)):
                sa[index] = new_list[j]
                index += 1

        sa["Samples"] = index
        self.fitTransformed(sa, wordLength, symbols, normMean)


    def fitTransformed(self, samples, wordLength, symbols, normMean):
        length = len(samples[0].data)
        transformedSignal = self.sfa.fitTransformDouble(samples, length, symbols, normMean)

        best = self.calcBestCoefficients(samples, transformedSignal)
        self.bestValues = [0 for i in range(min(len(best), wordLength))]
        self.maxWordLength = 0

        for i in range(len(self.bestValues)):
            if best[i][1] != -math.inf:
                self.bestValues[i] = best[i][0]
                self.maxWordLength = max(best[i][0] + 1, self.maxWordLength)

        self.maxWordLength += self.maxWordLength % 2
        self.sfa.maxWordLength = self.maxWordLength
        return self.sfa.transform(samples, transformedSignal)


    def calcBestCoefficients(self, samples, transformedSignal):
        classes = {}
        for i in range(samples["Samples"]):
            if samples[i].label in classes.keys():
                classes[samples[i].label].append(transformedSignal[i])
            else:
                classes[samples[i].label] = [transformedSignal[i]]

        nSamples = len(transformedSignal)
        nClasses = len(classes.keys())
        length = len(transformedSignal[1])

        f = self.getFoneway(length, classes, nSamples, nClasses)
        f_sorted = sorted(f, reverse = True)
        best = []
        inf_index = 0

        for value in f_sorted:
            if value == -math.inf:
                index = f.index(value) + inf_index
                inf_index += 1
            else:
                index = f.index(value)
                best.append([index, value])  #NOTE Changed to indent

        return best


    def getFoneway(self, length, classes, nSamples, nClasses):
        ss_alldata = [0. for i in range(length)]
        sums_args = {}
        keys_class = list(classes.keys())

        for key in keys_class:
            allTs = classes[key]
            sums = [0. for i in range(len(ss_alldata))]
            sums_args[key] = sums
            for ts in allTs:
                for i in range(len(ts)):
                    ss_alldata[i] += ts[i] * ts[i]
                    sums[i] += ts[i]

        square_of_sums_alldata = [0. for i in range(len(ss_alldata))]
        square_of_sums_args = {}
        for key in keys_class:
            # square_of_sums_alldata2 = [0. for i in range(len(ss_alldata))]
            sums = sums_args[key]
            for i in range(len(sums)):
                square_of_sums_alldata[i] += sums[i]
            # square_of_sums_alldata += square_of_sums_alldata2

            squares = [0. for i in range(len(sums))]
            square_of_sums_args[key] = squares
            for i in range(len(sums)):
                squares[i] += sums[i]*sums[i]

        for i in range(len(square_of_sums_alldata)):
            square_of_sums_alldata[i] *= square_of_sums_alldata[i]

        sstot = [0. for i in range(len(ss_alldata))]
        for i in range(len(sstot)):
            sstot[i] = ss_alldata[i] - square_of_sums_alldata[i]/nSamples

        ssbn = [0. for i in range(len(ss_alldata))]    ## sum of squares between
        sswn = [0. for i in range(len(ss_alldata))]    ## sum of squares within

        for key in keys_class:
            sums = square_of_sums_args[key]
            n_samples_per_class = len(classes[key])
            for i in range(len(sums)):
                ssbn[i] += sums[i]/n_samples_per_class

        for i in range(len(square_of_sums_alldata)):
            ssbn[i] += -square_of_sums_alldata[i]/nSamples

        dfbn = nClasses-1                          ## degrees of freedom between
        dfwn = nSamples - nClasses                 ## degrees of freedom within
        msb = [0. for i in range(len(ss_alldata))]   ## variance (mean square) between classes
        msw = [0. for i in range(len(ss_alldata))]   ## variance (mean square) within samples
        f = [0. for i in range(len(ss_alldata))]     ## f-ratio


        for i in range(len(sswn)):
            sswn[i] = sstot[i]-ssbn[i]
            msb[i] = ssbn[i]/dfbn
            msw[i] = sswn[i]/dfwn
            f[i] = msb[i]/msw[i] if msw[i] != 0. else -math.inf

        return f


    def quantizationSupervised(self, one_approx):
        signal = [0 for _ in range(min(len(one_approx), len(self.bestValues)))]

        for a in range(len(signal)):
            i = self.bestValues[a]
            b = 0
            for beta in range(self.sfa.bins.shape[1]):
                if one_approx[i] < self.sfa.bins.iloc[i,beta]:
                    break
                else:
                    b += 1
            signal[a] = b-1

        return signal





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
