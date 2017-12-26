import numpy as np
import math

from src.timeseries.TimeSeries import calcIncreamentalMeanStddev
from src.timeseries.TimeSeries import TimeSeries


class MFT:

    def __init__(self, windowSize, normMean, lowerBounding, MUSE_Bool = False):
        self.windowSize = windowSize
        self.MUSE = MUSE_Bool

        self.startOffset = 2 if normMean else 0
        self.norm = 1.0 / np.sqrt(windowSize) if lowerBounding else 1.0


    def transform(self, series, wordlength):
        FFT_series = np.fft.fft(series)
        data_new = []
        windowSize = len(series)

        for i in range(int(math.ceil(len(series) / 2))):
            data_new.append(FFT_series[i].real)
            data_new.append(FFT_series[i].imag)
        data_new[1] = 0.0
        data_new = data_new[:self.windowSize]

        length = min([windowSize - self.startOffset, wordlength])
        copy = data_new[(self.startOffset):(length + self.startOffset)]
        while len(copy) != wordlength:
            copy.append(0)

        sign = 1
        for i in range(len(copy)):
            copy[i] *= self.norm * sign
            sign *= -1

        return copy


    def transformWindowing(self, series_full, wordLength):
        series = series_full.data
        WORDLENGTH = max(self.windowSize, wordLength + self.startOffset) if self.MUSE else min(self.windowSize, wordLength + self.startOffset)
        WORDLENGTH = WORDLENGTH + WORDLENGTH % 2
        phis = [0. for i in range(WORDLENGTH)]
        for u in range(0, WORDLENGTH, 2):
            uHalve = -u / 2
            phis[u] = math.cos(2 * math.pi * uHalve / self.windowSize)
            phis[u+1] = -math.sin(2 * math.pi * uHalve / self.windowSize)

        final = max(1, len(series) - self.windowSize + 1)
        self.MEANS = []
        self.STDS = []

        self.MEANS, self.STDS = calcIncreamentalMeanStddev(self.windowSize, series, self.MEANS, self.STDS)
        transformed = []

        data = series
        mftData_FFT = []

        for t in range(final):
            if t > 0:
                k = 0
                while k < WORDLENGTH:
                    real1 = mftData_FFT[k] + data[t + self.windowSize-1] - data[t-1]
                    imag1 = mftData_FFT[k + 1]

                    real = (real1 * phis[k]) - (imag1 * phis[k + 1])
                    imag = (real1 * phis[k + 1]) + (imag1 * phis[k])
                    mftData_FFT[k] = real
                    mftData_FFT[k + 1] = imag
                    k += 2
            else:
                mftData_fft = np.fft.fft(data[:self.windowSize])
                mftData_FFT = [0. for _ in range(WORDLENGTH)]

                i = 0
                for j in range(min(self.windowSize, WORDLENGTH)):
                    if j % 2 == 0:
                        mftData_FFT[j] = mftData_fft[i].real
                    else:
                        mftData_FFT[j] = mftData_fft[i].imag
                        i += 1
                mftData_FFT[1] = 0.


            copy = [0. for i in range(wordLength)]
            copy_value = mftData_FFT[(self.startOffset):(self.startOffset + wordLength)]
            copy[:len(copy_value)] = copy_value
            copy = TimeSeries(copy, series_full.label, series_full.NORM_CHECK)
            copy = self.normalizeFT(copy, self.STDS[t])
            transformed.append(copy)

        return transformed

    def normalizeFT(self, copy, std):
        normalisingFactor = 1. / std if (copy.NORM_CHECK) & (std > 0) else 1.
        normalisingFactor *= self.norm

        sign = 1
        for i in range(len(copy.data)):
            copy.data[i] *= sign * normalisingFactor
            sign *= -1
        return copy.data

