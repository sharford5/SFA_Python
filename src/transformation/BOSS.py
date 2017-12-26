from  src.transformation.SFA import *


class BOSS():

    def __init__(self, maxF, maxS, windowLength, normMean):
        self.maxF = maxF
        self.symbols = maxS
        self.windowLength = windowLength
        self.normMean = normMean
        self.signature = None


    def createWords(self, samples):
        if self.signature == None:
            self.signature = SFA("EQUI_DEPTH")
            self.signature.fitWindowing(samples, self.windowLength, self.maxF, self.symbols, self.normMean, True)
            # self.signature.printBins()

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


    def bag2dict(self, bag):
        bag_dict = []
        for list in bag:
            new_dict = {}
            for element in list:
                if element in new_dict.keys():
                    new_dict[element] += 1
                else:
                    new_dict[element] = 1
            bag_dict.append(new_dict)
        return bag_dict