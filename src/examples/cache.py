import numpy as np
import time

import explot as ep



def my_experiment(x, seed=None):
    """
    A simple example for an experiment function with seed.
    
    repetition_index as an parameter is set by ExPlot.
    """

    # simulate heavy computation
    time.sleep(1)
    
    # invent some noisy results
    np.random.seed(seed)
    return np.sin(x) + .5 * np.random.randn()



def main():
    """
    Calls a computationally expensive experiment and caches the results. If you
    call it a second time, you should see the plot instantly.
    """
    seed = 0
    repetitions = 3
    cachedir = '/tmp'
    
    # plot varying 'x' and cache the results
    ep.plot(my_experiment, 
            x=range(10), 
            seed=seed, 
            repetitions=repetitions,
            cachedir=cachedir)
    return



if __name__ == '__main__':
    main()
