
from FourPeaksOpt import runFourPeaksOpt
from kColorsOpt import runkColorsOpt
from QueensOpt import runQueensOpt
from FlipFlipOpt import runFlipFlopOpt
from KnapsackOpt import runKnapsackOpt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from NNAgent import runNNOpt



def runNNexp():
    runNNOpt()


def runRandomOptExp():

    global tracked_times, start_time_track

    runFourPeaksOpt()
    runkColorsOpt()
    # runQueensOpt()
    runFlipFlopOpt()
    runKnapsackOpt()

    runNNexp()



def main():
    np.random.seed(42)

    runRandomOptExp()




if __name__ == "__main__":
    main()