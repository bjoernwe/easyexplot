import collections
import datetime
import functools
import inspect
import itertools
import multiprocessing
import numpy as np
import os
import pickle
import random
import sys
import textwrap
import time
import traceback


__version__ = 0.2

RESULT_PATH = 'easyexplot_results'

Result = collections.namedtuple('Result', ['values', 
                                           'time_start', 
                                           'time_stop', 
                                           'iter_args', 
                                           'kwargs', 
                                           'script', 
                                           'repetitions', 
                                           'result_prefix'])
"""
A `namedtuple` to store results of `evaluate`.

Attributes
----------
values : array_like
    An ndim array that contains all function evaluations. Each axis corresponds 
    to one iterable argument. The last axis stores different repetitions of
    experiments.
time_start : time.struct_time
    Timestamp before (parallel) evaluation started.
time_stop : time.struct_time
    Timestamp after (parallel) evaluation stopped.
iter_args : OrderedDict
    Am ordered dictionary of all iterable argument names and there values. 
kwargs : dict
    Dictionary of all the non-iterable arguments used for evaluation.
script : str
    Name of the calling script.
repetitions : int
    Number of repetitions used for evaluation.
result_prefix : str
    A unique prefix (consisting of date and time), used for instance for saving 
    the result in a file.
"""



def evaluate(experiment_function, repetitions=1, processes=None, argument_order=None, save_result=False, **kwargs):
    """
    Evaluates the real-valued function f using the given keyword arguments. 
    Usually, one or more of the arguments are iterables (for instance a list of 
    integers) which are used for parallel evaluation. The result is a 
    `namedtuple` which can be passed to the plot function.
    
    Parameters
    ----------
    experiment_function : function
        A function that takes the given `kwargs` and returns double or int.
    repetitions : int, optional
        Defines how often `experiment_function` is evaluated. Useful for 
        calculating mean and standard deviation in noisy experiments.
    processes : int or None, optional
        Number of CPU cores used. If None (default), all but one cores are used.
    argument_order : list of strings
        Some of the iterable argument names may be given in a list to force a
        certain order. Without this, Python's kwargs have an undefined order 
        which may result in plots other than intended.
    save_result : bool, optional
        If True, the pickled result is stored in `RESULT_PATH` (default: False).
    kwargs : dict, optional
        Keyword arguments passed to function `experiment_function`.
        
    Returns
    -------
    Result
        A structure summarizing the result of the evaluation. 
    """
    
    # get default arguments of function f and update them with given ones
    #
    # this is not strictly necessary but otherwise the argument lists lacks
    # the default ones which should be included in the plot
    fargspecs = inspect.getargspec(experiment_function)
    fkwargs = {}
    if fargspecs.defaults is not None:
        fkwargs.update(dict(zip(fargspecs.args[-len(fargspecs.defaults):], fargspecs.defaults)))
    fkwargs.update(kwargs)

    # OrderedDict of argument names and values
    iter_args = collections.OrderedDict()
    if argument_order is not None:
        iter_args.update([(name, None) for name in argument_order if name in fkwargs and _is_iterable(fkwargs[name])])
    iter_args.update({name: fkwargs.pop(name) for (name, values) in fkwargs.items() if _is_iterable(values)})

    # create a argument list like [(arg0[0], arg1[0]), (arg0[1], arg1[1]), ...]
    # and then each tuple repeated for the specified number of repetitions
    iter_args_values_tupled = itertools.product(*iter_args.values())
    iter_args_values_tupled_repeated = [arg for arg in iter_args_values_tupled for _ in range(repetitions)]

    # make sure, all arguments are defined for function f
    if len(fargspecs.args) > len(iter_args) + len(fkwargs):
        undefined_args = set(fargspecs.args) - set(fkwargs.keys()) - set(iter_args.keys())
        sys.stderr.write('Error: Undefined arguments: %s' % str.join(', ', undefined_args))
        return
    
    # wrap function f
    f_partial = functools.partial(_f_wrapper, 
                                  iter_arg_names=iter_args.keys(), 
                                  experiment_function=experiment_function, 
                                  **fkwargs)
    
    # number of parallel processes
    if processes is None:
        processes = max(0, multiprocessing.cpu_count() - 1)
            
    # start a pool of processes
    time_start = time.localtime()
    if processes <= 1:
        result_values = map(f_partial, iter_args_values_tupled_repeated)
    else:
        pool = multiprocessing.Pool(processes=processes)
        result_values = pool.map(f_partial, iter_args_values_tupled_repeated, chunksize=1)
        pool.close()
        pool.join()
    time_stop = time.localtime()
    
    # re-arrange repetitions in ndim array
    iter_args_values_lengths = [len(values) for values in iter_args.values()]
    values_shape = tuple(iter_args_values_lengths + [repetitions])
    result_values = np.reshape(result_values, values_shape)
        
    # calculate a prefix for result files
    timestamp = time.strftime('%Y%m%d_%H%M%S', time_start)
    number_of_results = 0
    if os.path.exists(RESULT_PATH):
        number_of_results = len(set([os.path.splitext(f)[0] for f in os.listdir(RESULT_PATH) if f.startswith(timestamp)]))
    result_prefix = '%s_%02d' % (timestamp, number_of_results)

    # prepare result
    result = Result(values=result_values,
                    time_start=time_start,
                    time_stop=time_stop,
                    iter_args=iter_args,
                    kwargs=fkwargs,
                    script=([s[1] for s in inspect.stack() if os.path.basename(s[1]) != 'plotter.py'] + [None])[0],
                    repetitions=repetitions,
                    result_prefix=result_prefix)
    
    if save_result:
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        with open('%s/%s.pkl' % RESULT_PATH, result_prefix, 'wb') as f:
            pickle.dump(result, f)
    
    return result



def _f_wrapper(args, iter_arg_names, experiment_function, **kwargs):
    """
    [Intended for internal use only] A simple wrapper for the experiment 
    function that allows having specific arguments ('iter_args') as the first 
    argument. This is the method that is actually managed and called by the 
    multiprocessing pool. Therefore the argument 'niceness' is removed from 
    **kwargs and used to increment the niceness of the current process 
    (default: 10). Also the Python's and NumPy's random number generators are 
    initialized with a new seed.
    
    Parameters
    ----------
    args : list
        The values for `iter_args` when calling `experiment_function`.
    iter_arg_names : tuple of strings
        Names of the arguments
    experiment_function : function
        The function to call as experiment_function(iter_arg_names[0]=args[0], iter_arg_names[1]=args[1], ..., **kwargs).
    kwargs : dict, optional
        All other arguments for `experiment_function`.
    """
    os.nice(kwargs.pop('niceness', 20))
    random.seed()
    np.random.seed()
    if iter_arg_names is not None:
        for i, iter_arg_name in enumerate(iter_arg_names):
            kwargs[iter_arg_name] = args[i]
    try:
        result = experiment_function(**kwargs)
    except Exception as e:
        sys.stderr.write(traceback.format_exc())
        raise e
    return result



def plot(experiment_function, repetitions=1, processes=None, argument_order=None, save_result=False, show_plot=True, save_plot=False, **kwargs):
    """
    Plots the real-valued function f using the given keyword arguments. At least
    one of the arguments must be an iterable (for instance a list of integers), 
    which is used to evaluate and plot the experiment function for different 
    input values.
    
    Parameters
    ----------
    experiment_function : function
        A function that takes the given `kwargs` and returns double or int.
    repetitions : int, optional
        Defines how often `experiment_function` is evaluated. Useful for 
        calculating mean and standard deviation in noisy experiments.
    processes : int or None, optional
        Number of CPU cores used. If None (default), all but one cores are used.
    argument_order : list of strings
        Some of the iterable argument names may be given in a list to force a
        certain order. Without this, Python's kwargs have an undefined order 
        which may result in plots other than intended.
    save_result : bool, optional
        If True, the pickled result is stored in `RESULT_PATH` (default: False).
    show_plot : bool, optional
        Indicates whether pyplot.show() is called or not (default: True).
    save_plot : bool, optional
        Indicates whether the plot is saved as a PNG file in 
        './experimentr_results' (default: False).
    kwargs : dict, optional
        Keyword arguments passed to function `experiment_function`.
        
    Returns
    -------
    Result
        A structure summarizing the result of the evaluation. 
    """

    # run the experiment
    result = evaluate(experiment_function, repetitions=repetitions, processes=processes, argument_order=argument_order, save_result=save_result, **kwargs)
    if result is None:
        return

    plot_result(result, save_plot=save_plot, show_plot=show_plot)
    return result



def plot_result(result, save_plot=True, show_plot=True):
    """
    Plots the result of an experiment. The result can be given as the return 
    value of evaluate() directly or as the filename of a previously pickled 
    results, e.g., '20141205_150015_00.pkl'.
    
    Parameters
    ----------
    result : Result or str
        The result to plot.
    show_plot : bool, optional
        Indicates whether pyplot.show() is called or not (default: True).
    save_plot : bool, optional
        Indicates whether the plot is saved as a PNG file in 
        './experimentr_results' (default: False).
    
    Returns
    -------
    Result
        Either the result given or the unpickled one. 
    """

    # import here makes evaluate() independent from matplotlib
    from matplotlib import pyplot as plt
    
    # read result from file
    if isinstance(result, str):
        result = pickle.load(open(RESULT_PATH + result))
        
    # sort the iterable arguments according to whether they are numeric
    numeric_iter_args = collections.OrderedDict([(name, values) for (name, values) in result.iter_args.items() if _all_numeric(values)])
    non_numeric_iter_args = collections.OrderedDict([(name, values) for (name, values) in result.iter_args.items() if not _all_numeric(values)])

    # cmap for plots
    cmap = plt.get_cmap('jet')

    if len(numeric_iter_args) == 0:
        assert len(non_numeric_iter_args) >= 1
        x_values = np.arange(result.values.shape[0])
        indices_per_axis = [[slice(None)]] + [range(len(values)) for values in result.iter_args.values()[1:]]
        index_tuples = list(itertools.product(*indices_per_axis))
        width = .8 / len(index_tuples)
        for i, index_tuple in enumerate(index_tuples):
            y_values = np.mean(result.values[index_tuple], axis=-1)
            y_errors = np.std(result.values[index_tuple], axis=-1)
            plt.bar(x_values + i*width, y_values, yerr=y_errors, width=width, color=cmap(1.*i/len(index_tuples)))
        plt.gca().set_xticks(np.arange(len(x_values)) + .4)
        plt.gca().set_xticklabels([str(value) for value in non_numeric_iter_args.values()[0]])
        legend = []
        non_numeric_name_value_lists = [[(name, value) for value in values] for (name, values) in non_numeric_iter_args.items()[1:]]
        if len(non_numeric_name_value_lists) >= 1:
            for name_value_combination in itertools.product(*non_numeric_name_value_lists):
                legend.append(str.join(', ', ["%s = %s" % (name, value) for (name, value) in name_value_combination]))
            plt.legend(legend, loc='best')
        plt.xlabel(non_numeric_iter_args.keys()[0])
    elif len(numeric_iter_args) == 1:
        indices_per_axis = [[slice(None)] if _all_numeric(values) else range(len(values)) for values in result.iter_args.values()]
        index_tuples = list(itertools.product(*indices_per_axis))
        for i, index_tuple in enumerate(index_tuples):
            x_values = numeric_iter_args.values()[0]
            y_values = np.mean(result.values[index_tuple], axis=-1)
            y_errors = np.std(result.values[index_tuple], axis=-1)
            plt.errorbar(x_values, y_values, yerr=y_errors, linewidth=1., color=cmap(1.*i/len(index_tuples)))
        legend = []
        non_numeric_name_value_lists = [[(name, value) for value in values] for (name, values) in non_numeric_iter_args.items()]
        if len(non_numeric_name_value_lists) >= 1:
            for name_value_combination in itertools.product(*non_numeric_name_value_lists):
                legend.append(str.join(', ', ["%s = %s" % (name, value) for (name, value) in name_value_combination]))
            plt.legend(legend, loc='best')
        plt.xlabel(numeric_iter_args.keys()[0])
    elif len(numeric_iter_args) == 2:
        assert len(non_numeric_iter_args) == 0
        result_values = np.mean(result.values, axis=-1)
        plt.imshow(result_values.T, origin='lower', cmap=cmap)
        plt.xlabel(numeric_iter_args.keys()[0])
        plt.ylabel(numeric_iter_args.keys()[1])
    else:
        sys.stderr.write('Error: Do not know (yet) how to plot a result with more than two iterables.')
        return
        
    # calculate running time
    time_diff = time.mktime(result.time_stop) - time.mktime(result.time_start)
    time_delta = datetime.timedelta(seconds=time_diff)
    time_start_str = time.strftime('%Y-%m-%d %H:%M:%S', result.time_start)
    if result.time_start.tm_yday == result.time_stop.tm_yday:
        time_stop_str = time.strftime('%H:%M:%S', result.time_start)
    else:
        time_stop_str = time.strftime('%Y-%m-%d %H:%M:%S', result.time_start)
        
    # describe plot
    plt.suptitle(result.script)
    plotted_args = result.kwargs.copy()
    if result.repetitions > 1:
        plotted_args['repetitions'] = result.repetitions
    parameter_text = 'Parameters: %s' % str.join(', ', ['%s=%s' % (k,v) for k,v in plotted_args.items()])
    parameter_text = str.join('\n', textwrap.wrap(parameter_text, 100))
    plt.title('Time: %s - %s (%s)\n' % (time_start_str, time_stop_str, time_delta) + 
              parameter_text,
              fontsize=12)
    plt.subplots_adjust(top=0.8)

    # save plot in file
    if save_plot:
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        plt.savefig('%s/%s.png' % (RESULT_PATH, result.result_prefix))

    # show plot
    if show_plot:
        plt.show()

    return result



def _is_numeric(x):
    if x is None:
        return False
    if isinstance(x, bool):
        return False
    try:
        float(x)
        return True
    except ValueError:
        return False
    
    
    
def _all_numeric(iterable):
    return all([_is_numeric(x) for x in iterable])



def _is_iterable(x):
    if isinstance(x, str):
        return False
    return isinstance(x, collections.Iterable) 



def list_results():
    """
    Prints a summary of all previous results saved in sub-directory 
    `RESULT_PATH`.
    """
    if os.path.exists('plotter_results'):
        files = [f for f in os.listdir(RESULT_PATH) if os.path.splitext(f)[1] == '.pkl']
        files = sorted(files)
        for f in files:
            result = pickle.load(open(RESULT_PATH + f))
            print '%s  <%s>  \t%s, %s' % (result.result_prefix, 
                                       os.path.basename(str(result.script)),
                                       result.iter_arg_name,
                                       ', '.join(['%s=%s' % (k,v) for (k,v) in result.kwargs.items()]))
    return



def my_experiment(x, y=0, f='sin', shift=False):
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
    result_value += y + np.sin(y) + .5 * np.random.randn()
    return result_value



def main():
    """
    Calls the plot function on my_experiment().
    """
    repetitions = 50
    show_plot = True
    save_plot = False

    # regular call of the experiment    
    print my_experiment(x=0, f='sin', shift=False)

    # plot with varying x
    plot(my_experiment, x=range(10), f='sin', shift=False, repetitions=repetitions, show_plot=show_plot, save_plot=save_plot)
    
    # plot with varying x and f
    plot(my_experiment, x=range(10), f=['sin', 'cos'], shift=False, repetitions=repetitions, show_plot=show_plot, save_plot=save_plot)
    
    # plot with varying x as well as f and shift
    plot(my_experiment, x=range(10), f=['sin', 'cos'], shift=[False, True], repetitions=repetitions, show_plot=show_plot, save_plot=save_plot)

    # plot with varying x as well as f and shift, but force a certain order
    plot(my_experiment, x=range(10), f=['sin', 'cos'], shift=[False, True], argument_order=['f', 'shift'], repetitions=repetitions, show_plot=show_plot, save_plot=save_plot)
    
    # bar plot for x=0 with varying f and shift (as well as forced order of parameters)
    plot(my_experiment, x=0, f=['sin', 'cos'], shift=[False, True], repetitions=repetitions, show_plot=show_plot, save_plot=save_plot)
    plot(my_experiment, x=0, f=['sin', 'cos'], shift=[False, True], argument_order=['f'], repetitions=repetitions, show_plot=show_plot, save_plot=save_plot)
    
    # 2d plot
    plot(my_experiment, x=range(10), y=range(10), repetitions=repetitions, show_plot=show_plot, save_plot=save_plot)



if __name__ == '__main__':
    main()
