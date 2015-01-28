import time

import easyexplot as eep



def my_experiment(x, fast=False):
    """
    A simple example which execution time depends on the input value.
    """
    # simulate heavy computation
    if fast:
        x *= .8
    time.sleep(.1 + x**.5 / 1000)
    return 0



def main():
    """
    Plots how long the experiment took for execution.
    """
    repetitions = 10
    
    # plot varying 'x' and cache the results
    eep.plot(my_experiment, 
             x=range(10),
             fast=[False, True], 
             repetitions=repetitions,
             plot_elapsed_time=True)
    return



if __name__ == '__main__':
    main()
