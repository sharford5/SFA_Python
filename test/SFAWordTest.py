import sys
import os
sys.path.append(os.getcwd()[:-5])

from src.timeseries.TimeSeriesLoader import uv_load
from  src.transformation.SFA import *

symbols = 8
wordLength = 16
normMean = False

def sfaToWord(word):
    word_string = ""
    alphabet = "abcdefghijklmnopqrstuv"
    for w in word:
        word_string += alphabet[w]
    return word_string


train, test = uv_load("Gun_Point")

sfa = SFA("EQUI_DEPTH")

sfa.fitTransform(train, wordLength, symbols, normMean)

sfa.printBins()

for i in range(test["Samples"]):
    wordList = sfa.transform2(test[i].data, "null")
    print(str(i) + "-th transformed time series SFA word " + "\t" + sfaToWord(wordList))




