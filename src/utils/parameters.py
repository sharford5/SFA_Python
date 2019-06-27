import argparse
import io
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument("--test", default='SFAWordTest', type=str, help='Current Options include \
                                                                Early Classification:[TEASER]; \
                                                                Multivariate:[MUSE]; \
                                                                Univariate Classification: [WEASEL, BOSSEnsemble, BOSSVS, ShotgunEnsemble, Shotgun]; \
                                                                SFA Tests: [SFAWordTest, SFAWordWindowingTest]')
parser.add_argument("--dataset", default='ItalyPowerDemand', type=str)  #ItalyPowerDemand   Gun_Point  JapaneseVowels  DigitShapeRandom   PenDigits  CBF
parser.add_argument("--train_bool", default=1, type=int)

parser.add_argument("--histogram_type", default="INFORMATION_GAIN", type=str, help='Controls Options include')  #INFORMATION_GAIN  EQUI_DEPTH
parser.add_argument("--symbols", default=8, type=int)
parser.add_argument("--wordLength", default=8, type=int)
parser.add_argument("--windowLength", default=8, type=int)
parser.add_argument("--normMean", default=1, type=int)

parser.add_argument("--MAX_WINDOW_LENGTH", default=350, type=int, help='limitation on the number of windows checked')

args = parser.parse_args()


def load_parameters():
    FIXED_PARAMETERS = {
        "data_path": "./datasets/",
        "log_path": "./logs/",

        "test": args.test,
        "dataset": args.dataset,
        "train_bool": args.train_bool,

        "histogram_type": args.histogram_type,
        "symbols": args.symbols,
        "wordLength": args.wordLength,
        "windowLength": args.windowLength,
        "normMean": args.normMean,

        "MAX_WINDOW_LENGTH": args.MAX_WINDOW_LENGTH,

    }
    return FIXED_PARAMETERS
