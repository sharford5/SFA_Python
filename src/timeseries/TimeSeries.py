import numpy as np
import math


def getDisjointSequences(series, windowSize, normMean):
    amount = int(math.floor((len(series.data) / windowSize)))
    subseqences = []

    for i in range(amount):
        subseqences_data = TimeSeries(series.data[(i*windowSize):((i+1)*windowSize)], series.label, series.NORM_CHECK)
        subseqences_data.NORM(normMean)
        subseqences.append(subseqences_data)

    return subseqences


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

    return MEANS, STDS


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




class TimeSeries():
    def __init__(self, data, label, NORM_CHECK = True):
        self.data = data
        self.label = label
        self.normed = False
        self.mean = 0.
        self.std = 1.

        self.NORM_CHECK = NORM_CHECK

    def NORM(self, normMean):
        self.mean = np.mean(self.data)
        self.calculate_std()

        thisNorm = not self.normed

        if (self.NORM_CHECK) & (thisNorm):
            self.NORM_WORK(normMean)


    def calculate_std(self):
        var = 0.
        for i in range(len(self.data)):
            var += self.data[i] * self.data[i]

        norm = 1.0 / len(self.data)
        buf = (norm * var) - (self.mean * self.mean)

        self.std = np.sqrt(buf) if buf != 0 else 0.


    def NORM_WORK(self, normMean):
        ISTD = 1. if self.std == 0 else 1. / self.std

        if normMean:
            self.data = [(self.data[i] - self.mean) * ISTD for i in range(len(self.data))]
            self.mean = 0.0
        elif np.any(ISTD != 1.):
            self.data = [self.data[i] * ISTD for i in range(len(self.data))]

        self.normed = True




