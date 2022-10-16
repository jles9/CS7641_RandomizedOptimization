
from FourPeaksOpt import runFourPeaksOpt
from kColorsOpt import runkColorsOpt
from QueensOpt import runQueensOpt
from FlipFlipOpt import runFlipFlopOpt
from KnapsackOpt import runKnapsackOpt
import numpy as np

'''
def time_track_callback(itr, attempt=None, done=None, state=None, fitness=None, curve=None, user_data=None):
    # We can get this callback called at each iteration of an algorithim
    # Use a global variable to get around this function not having a return type

    if itr==0:
        tracked_times = []
        start_time_track = time.time()
        
    else:
        end = time.time()
        time_this_itr = end-start_time_track
        tracked_times.append(time_this_itr)

    return True

'''

def runRandomOptExp():

    global tracked_times, start_time_track

    runFourPeaksOpt()
    #runkColorsOpt()
    # runQueensOpt()
    runFlipFlopOpt()
    runKnapsackOpt()




def main():
    np.random.seed(42)

    runRandomOptExp()




if __name__ == "__main__":
    main()