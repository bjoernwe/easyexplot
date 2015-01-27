import numpy as np
import time

import easyexplot as eep



def my_experiment(x, f='sin', shift=False, seed=None):
    """
    A simple example for an experiment function with seed.
    """
    
    # initialize RNG with given seed
    print 'got seed:', seed
    np.random.seed(seed)

    # simulate heavy computation
    time.sleep(1)
    
    # invent some results
    if f == 'sin':
        result_value = np.sin(x)
    elif f == 'cos':
        result_value = np.cos(x)
    else:
        result_value = 0
    if shift:
        result_value += .3
    result_value += .5 * np.random.randn()
    return result_value



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
