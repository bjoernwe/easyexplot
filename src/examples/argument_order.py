import numpy as np

from matplotlib import pyplot as plt

import explot as ep


def my_experiment(x, f='sin', shift=False, dummy=None):
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
    print my_experiment(x=0, f='sin', shift=False, dummy=(0,0))

    # plot varying 'x' as well as 'shift' and 'f' in this order
    plt.subplot(1, 2, 1)
    ep.plot(my_experiment, 
            x=range(5), 
            f=['sin', 'cos'], 
            shift=[False, True], 
            dummy=(0,0),
            argument_order=['x', 'shift', 'f'],
            ignore_arguments=['dummy'], 
            repetitions=repetitions, 
            show_plot=False, 
            processes=processes)

    # plot varying 'x' as well as 'f' and 'shift' in this order
    plt.subplot(1, 2, 2)
    ep.plot(my_experiment, 
            x=range(5), 
            f=['sin', 'cos'], 
            shift=[False, True], 
            dummy=(0,0),
            argument_order=['x', 'f', 'shift'], 
            ignore_arguments=['dummy'], 
            repetitions=repetitions, 
            show_plot=True, 
            processes=processes)
    return



if __name__ == '__main__':
    main()
