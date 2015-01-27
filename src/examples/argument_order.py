import numpy as np

from matplotlib import pyplot as plt

import easyexplot as eep


def my_experiment(x, f='sin', shift=False):
    """
    A simple example for an experiment function.
    """
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
    Calls the plot function on my_experiment() with arguments ordered 
    differently.
    """
    repetitions = 10
    processes = None
    
    # regular call of the experiment    
    print my_experiment(x=0, f='sin', shift=False)

    # plot varying 'x' as well as 'shift' and 'f' in this order
    plt.subplot(1, 2, 1)
    eep.plot(my_experiment, 
             x=range(5), 
             f=['sin', 'cos'], 
             shift=[False, True], 
             argument_order=['shift', 'f'], 
             repetitions=repetitions, 
             show_plot=False, 
             processes=processes)

    # plot varying 'x' as well as 'f' and 'shift' in this order
    plt.subplot(1, 2, 2)
    eep.plot(my_experiment, 
             x=range(5), 
             f=['sin', 'cos'], 
             shift=[False, True], 
             argument_order=['f', 'shift'], 
             repetitions=repetitions, 
             show_plot=True, 
             processes=processes)
    return



if __name__ == '__main__':
    main()
