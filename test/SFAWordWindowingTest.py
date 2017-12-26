import sys
import os
sys.path.append(os.getcwd()[:-5])

from src.timeseries.TimeSeriesLoader import uv_load
from  src.transformation.SFA import *

symbols = 4
wordLength = 4
windowLength = 64
normMean = True


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


train, test = uv_load("Gun_Point")

sfa = SFA("EQUI_DEPTH")


sfa.fitWindowing(train, windowLength, wordLength, symbols, normMean, True)

sfa.printBins()


for i in range(test["Samples"]):
    wordList = sfa.transformWindowing(test[i])
    print(str(i) + "-th transformed time series SFA word " + "\t" + sfaToWordList(wordList))



