import sys
import os
sys.path.append(os.getcwd()[:-5])

from src.timeseries.TimeSeriesLoader import load
from  src.transformation.SFA import *

symbols = 8
wordLength = 16
normMean = False

def sfaToWord(word):
    word_string = ""
    for w in word:
        word_string += chr(w+97)
    return word_string


train, test, train_labels, test_labels = load("CBF", "\t")

sfa = SFA("EQUI_DEPTH")

sfa.fitTransform(train, train_labels, wordLength, symbols, normMean)

sfa.printBins()

for i in range(test.shape[0]):
    wordList = sfa.transform2(test.iloc[i,:], "null")
    print(str(i) + "-th transformed time series SFA word " + "\t" + sfaToWord(wordList))




