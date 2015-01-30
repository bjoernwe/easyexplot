import numpy as np
import time

import easyexplot as eep



def my_experiment(x, seed=None, repetition_index=None):
    """
    A simple example for an experiment function with seed.
    
    repetition_index as an parameter is set by EasyExPlot.
    """
    
    # Initialize RNG
    # The seed should depend deterministically on all the input parameters.
    # Without dependence on repetition_index, each repetition would look the 
    # same. Without dependence on x, we wouldn't generate a noisy function but
    # one that has the same shift everywhere.
    unique_seed = abs(hash((x, seed, repetition_index)))
    np.random.seed(unique_seed)

    # simulate heavy computation
    time.sleep(1)
    
    # invent some noisy results
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
    eep.plot(my_experiment, 
             x=range(10), 
             seed=seed, 
             repetitions=repetitions, 
             cachedir=cachedir)
    return



if __name__ == '__main__':
    main()
