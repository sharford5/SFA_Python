import sys
import os
sys.path.append(os.getcwd()[:-5])

from src.timeseries.TimeSeriesLoader import load
from  src.transformation.SFA import *

symbols = 4
wordLength = 4
windowLength = 64
normMean = True


def sfaToWord(word):
    word_string = ""
    for w in word:
        word_string += chr(w+97)
    return word_string

def sfaToWordList(wordList):
    list_string = ""
    for word in wordList:
        list_string += sfaToWord(word)
        list_string += "; "
    return list_string


train, test, train_labels, test_labels = load("CBF", "\t")

sfa = SFA("EQUI_DEPTH")


sfa.fitWindowing(train, train_labels, windowLength, wordLength, symbols, normMean, True);

sfa.printBins()


for i in range(test.shape[0]):
    wordList = sfa.transformWindowing(test.iloc[i,:])
    print(str(i) + "-th transformed time series SFA word " + "\t" + sfaToWordList(wordList))



