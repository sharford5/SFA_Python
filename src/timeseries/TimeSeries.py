import numpy as np
import math
import pandas as pd
from src.timeseries.TimeSeries import *

def NORM(series, normMean = True):
    global MEAN
    global STD
    MEAN = np.mean(series)
    STD = calculate_std(series)

    series_norm = NORM_WORK(series, normMean, MEAN, STD)
    return series_norm


def NORM_WORK(series, normMean, MEAN, STD):
    ISTD = 1. if STD == 0 else 1./STD

    series_norm = series
    if normMean:
        series_norm = [(series[i] - MEAN) * ISTD for i in range(len(series))]
        MEAN = 0.0
    elif np.any(ISTD != 1.0):
        series_norm = [series[i] * ISTD for i in range(len(series))]

    return series_norm


def getDisjointSequences(series, windowSize, normMean):
    amount = int(math.floor((len(series) / windowSize)))
    subseqences = []

    for i in range(amount):
        subseqences_data = series[(i*windowSize):((i+1)*windowSize)]
        subseqences_data_norm = NORM(subseqences_data, normMean)
        subseqences.append(subseqences_data_norm)
    return subseqences


def calculate_std(series):
    var = 0.
    for i in range(len(series)):
        var += series[i] * series[i]

    norm = 1.0 / len(series)
    buf = (norm*var) - (MEAN*MEAN)

    stddev = np.sqrt(buf) if buf != 0 else 0.
    return stddev


def calcIncreamentalMeanStddev(windowLength, series, MEANS, STDS):
    SUM = 0.
    squareSum = 0.

    rWindowLength = 1.0 / windowLength
    for ww in range(windowLength):
        SUM += series[ww]
        squareSum += series[ww]*series[ww]
    MEANS.append(SUM * rWindowLength)
    buf = squareSum*rWindowLength - MEANS[0]*MEANS[0]

    if buf > 0:
        STDS.append(np.sqrt(buf))
    else:
        STDS.append(0)

    for w in range(1,(len(series)-windowLength+1)):
        SUM += series[w+windowLength-1] - series[w-1]
        MEANS.append(SUM * rWindowLength)

        squareSum += series[w+windowLength-1]*series[w+windowLength-1] - series[w-1]*series[w-1]
        buf = squareSum * rWindowLength - MEANS[w]*MEANS[w]
        if buf > 0:
            STDS.append(np.sqrt(buf))
        else:
            STDS.append(0)


def createWord(numbers, maxF, bits):
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


def int2byte(number):
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


def compareTo(score, bestScore):
    if score[1] > bestScore[1]:
        return -1
    elif (score[1] == bestScore[1]) & (score[4] > bestScore[4]):
        return -1
    return 1