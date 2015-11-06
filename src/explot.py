import collections
import datetime
import functools
import hashlib
import inspect
import itertools
import multiprocessing
import numpy as np
import os
import sys
import textwrap
import time
import traceback


__version__ = '0.5.0'

Result = collections.namedtuple('Result', ['values', 
                                           'time_start', 
                                           'time_stop', 
                                           'iter_args', 
                                           'kwargs', 
                                           'elapsed_times',
                                           'seeds',
                                           'script', 
                                           'function_name',
                                           'repetitions',
                                           'cachedir'])
"""
A `namedtuple` to store results of `evaluate`.

Attributes
----------
values : array
    An ndim array that contains all function evaluations. Each axis corresponds 
    to one iterable argument. The last axis stores different repetitions of
    experiments.
time_start : time.struct_time
    Timestamp before (parallel) evaluation started.
time_stop : time.struct_time
    Timestamp after (parallel) evaluation stopped.
iter_args : OrderedDict
    Am ordered dictionary of all iterable argument names and their values. 
kwargs : dict
    Dictionary of all the non-iterable arguments used for evaluation.
elapsed_times : array
    Array containing execution times (in milliseconds) for each function 
    evaluation. The shape is the same as for `values`.
seeds : array
    Array containing the seeds used for each experiment. The shape is the same 
    as for `values`.
script : str
    Name of the calling script.
function_name : str
    Name of the experiment.
repetitions : int
    Number of repetitions used for evaluation.
cachedir : str or None
    Directory that was used for caching.
"""



def evaluate(experiment_function, repetitions=1, processes=None, argument_order=None, cachedir=None, **kwargs):
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
        calculating mean and standard deviation in noisy experiments. Note 
        though, that noisy experiments need proper initialization of the random
        number generator (RNG) to produce meaningful results. Your experiment 
        should take a seed value as well as an additional argument 
        `repetition_index` which will be set by ExPlot. The RNG then must 
        be initialized with a value that depends on all your input arguments, 
        including `seed` and `repetition_index`.
    processes : int or None, optional
        Number of CPU cores used. If None (default), all but one cores are used.
    argument_order : list of strings
        Some of the iterable argument names may be given in a list to force a
        certain order. Without this, Python's kwargs have an undefined order 
        which may result in plots other than intended.
    cachedir : str, optional
        If a cache directory is given, joblib.Memory is used to cache the
        results of experiments in that directory.
    kwargs : dict, optional
        Keyword arguments passed to function `experiment_function`. If a `seed`
        argument is given and the experiment is repeated more than once, the
        seed is replaced by a new value, determined by the set of input 
        arguments and the repetition index. Otherwise, repetition wouldn't have 
        any effect.
        
    Returns
    -------
    Result
        A structure summarizing the result of the evaluation. 
    """
    
    # create a dictionary with all the given arguments as well as the implicit
    # ones with default values
    fargspecs = inspect.getargspec(experiment_function)
    fkwargs = {}
    if fargspecs.defaults is not None:
        fkwargs.update(dict(zip(fargspecs.args[-len(fargspecs.defaults):], fargspecs.defaults)))
    fkwargs.update(kwargs)
    
    # warn if random experiment is cached without a (proper) seed
    if cachedir is not None and fkwargs.get('seed', None) is None:
        print "Warning: You are caching results for a problem that seems to be stochastic.\n" + \
            "In that case your function should take a seed and use it to initialize your\n" + \
            "random number generator. ExPlot will replace the the seed by a unique value\n" + \
            "in every repetition."
                
    # OrderedDict of argument names and values
    iter_args = collections.OrderedDict()
    # if parameters are ordered, then they are processed first
    if argument_order is not None:
        iter_args.update([(name, None) for name in argument_order if name in fkwargs and _is_iterable(fkwargs[name])])
    new_iter_args_dict = {name: fkwargs.pop(name) for (name, values) in fkwargs.items() if _is_iterable(values)}
    new_iter_args_dict = sorted(new_iter_args_dict.items(), key=lambda x: len(x[1]), reverse=True)
    iter_args.update(new_iter_args_dict)
    del(new_iter_args_dict)
    
    # here we create a list of tuples, containing all combinations of input 
    # arguments and repetition index: 
    # [(arg0[0], arg1[0]), (arg0[0], arg1[1]), (arg0[1], arg1[0]) ...]
    iter_args_values_tupled = itertools.product(*(iter_args.values() + [range(repetitions)]))

    # wrap function f
    f_partial = functools.partial(_f_wrapper, 
                                  iter_arg_names=iter_args.keys(), 
                                  experiment_function=experiment_function,
                                  cachedir=cachedir,
                                  **fkwargs)
    
    # number of parallel processes
    if processes is None:
        processes = max(0, multiprocessing.cpu_count() - 1)
            
    # start a pool of processes
    time_start = time.localtime()
    if processes <= 1:
        result_list = map(f_partial, iter_args_values_tupled)
    else:
        pool = multiprocessing.Pool(processes=processes)
        result_list = pool.map(f_partial, iter_args_values_tupled, chunksize=1)
        pool.close()
        pool.join()
    time_stop = time.localtime()
    
    # separate the result value from runtime measurement
    result_array = np.array(result_list)
    result_values = np.array(result_array[:,0], dtype='float')
    result_times = np.array(result_array[:,1], dtype='float')
    result_seeds = np.array(result_array[:,2], dtype='float')
    
    # re-arrange ndim array
    iter_args_values_lengths = [len(values) for values in iter_args.values()]
    values_shape = tuple(iter_args_values_lengths + [repetitions])
    result_values = np.reshape(result_values, values_shape)
    result_times = np.reshape(result_times, values_shape)
    result_seeds = np.reshape(result_seeds, values_shape)
        
    # prepare result
    result = Result(values=result_values,
                    time_start=time_start,
                    time_stop=time_stop,
                    iter_args=iter_args,
                    kwargs=fkwargs,
                    elapsed_times=result_times,
                    seeds=result_seeds,
                    script=([s[1] for s in inspect.stack() if os.path.basename(s[1]) != os.path.basename(__file__)] + [None])[0],
                    function_name=experiment_function.__name__,
                    repetitions=repetitions,
                    cachedir=cachedir)
    return result



def _f_wrapper(args, iter_arg_names, experiment_function, cachedir=None, **kwargs):
    """
    [Intended for internal use only] A simple wrapper for the experiment 
    function that allows having specific arguments ('iter_args') as the first 
    argument. This is the method that is actually managed and called by the 
    multiprocessing pool. Therefore the argument 'niceness' is removed from 
    **kwargs and used to increment the niceness of the current process 
    (default: 10).
    
    Parameters
    ----------
    args : list
        The values for `iter_args` when calling `experiment_function`. The last
        item in addition gives the index of the current repetition of the 
        experiment.
    iter_arg_names : tuple of strings
        Names of the arguments
    experiment_function : function
        The function to call as experiment_function(iter_arg_names[0]=args[0], iter_arg_names[1]=args[1], ..., **kwargs).
    cachedir : str, optional
        If a cache directory is given, joblib.Memory is used to cache the
        results of experiments in that directory.
    kwargs : dict, optional
        All other arguments for `experiment_function`.
        
    Returns
    -------
    tuple
        (result_value, elapsed_time, used_seed)
    """
    # reduce niceness of process
    os.nice(kwargs.pop('niceness', 20))
    
    # set current value for iterable arguments
    if iter_arg_names is not None:
        for i, iter_arg_name in enumerate(iter_arg_names):
            kwargs[iter_arg_name] = args[i]
       
    # replace seed for repeating experiments
    repetition_index = args[-1]     
    if kwargs.get('seed', None) is not None:
        unique_seed = hash(frozenset(kwargs.items() + [('repetition_index', repetition_index)])) % np.iinfo(np.uint32).max
        kwargs['seed'] = unique_seed
    used_seed = kwargs.get('seed', None)

    # cache function with joblib. the hash is calculated manually, because due
    # to the wrapper, joblib would not detect a modified experiment_function
    exp_func_hash = None
    f_wrapper_cached = _f_wrapper_timed
    if cachedir is not None:
        import joblib
        exp_func_code, _, _ = joblib.func_inspect.get_func_code(experiment_function)
        exp_func_hash = hashlib.sha1(exp_func_code).hexdigest()
        memory = joblib.Memory(cachedir=cachedir, verbose=0)
        f_wrapper_cached = memory.cache(_f_wrapper_timed)
    
    # execute experiment
    try:
        result, elapsed_time = f_wrapper_cached(wrapped_func=experiment_function, wrapped_hash=exp_func_hash, **kwargs)
    except Exception as e:
        sys.stderr.write(traceback.format_exc())
        raise e
    return (result, elapsed_time, used_seed)



def _f_wrapper_timed(wrapped_func, wrapped_hash, **kwargs):
    """
    [Intended for internal use only] A simple wrapper that executes 
    `wrapped_func` with the given `kwargs`. The result is returned together with
    the elapsed time in milliseconds.
    
    Parameters
    ----------
    wrapped_func : function
        The function to execute.
    wrapped_hash : str
        Hash of the source code of `wrapped_function`. The argument is not used,
        but it avoids outdated caches in combination with joblib.Memory.
    kwargs : dict
        Keyword arguments passed to `wrapped_func`
        
    Returns
    -------
    tuple
        (wrapped_func(**kwargs), elapsed_time)
    """
    dt1 = datetime.datetime.now()
    result = wrapped_func(**kwargs)
    dt2 = datetime.datetime.now()
    elapsed_time = (dt2 - dt1).total_seconds() * 1000
    return (result, elapsed_time) 



def plot(experiment_function, repetitions=1, processes=None, argument_order=None, cachedir=None, plot_elapsed_time=False, show_plot=True, save_plot_path=None, **kwargs):
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
        calculating mean and standard deviation in noisy experiments. Note 
        though, that noisy experiments need proper initialization of the random
        number generator (RNG) to produce meaningful results. Your experiment 
        should take a seed value as well as an additional argument 
        `repetition_index` which will be set by ExPlot. The RNG then must 
        be initialized with a value that depends on all your input arguments, 
        including `seed` and `repetition_index`.
    processes : int or None, optional
        Number of CPU cores used. If None (default), all but one cores are used.
    argument_order : list of strings
        Some of the iterable argument names may be given in a list to force a
        certain order. Without this, Python's kwargs have an undefined order 
        which may result in plots other than intended. Also, the first argument
        determines if a line plot or a bar plot is used.
    cachedir : str, optional
        If a cache directory is given, joblib.Memory is used to cache the
        results of experiments in that directory.
    plot_elapsed_time : bool, optional
        Indicated whether the elapsed time is plotted instead of the actual
        result.
    show_plot : bool, optional
        Indicates whether pyplot.show() is called or not (default: True).
    save_plot_path : string, optional
        Optional path where resulting plot is saved as a PNG file.
    kwargs : dict, optional
        Keyword arguments passed to function `experiment_function`. If a `seed`
        argument is given and the experiment is repeated more than once, the
        seed is replaced by a new value, determined by the set of input 
        arguments and the repetition index. Otherwise, repetition wouldn't have 
        any effect.
        
    Returns
    -------
    Result
        A structure summarizing the result of the evaluation. 
    """

    # run the experiment
    result = evaluate(experiment_function, 
                      repetitions=repetitions, 
                      processes=processes, 
                      argument_order=argument_order,
                      cachedir=cachedir, 
                      **kwargs)
    if result is None:
        return

    plot_result(result, plot_elapsed_time=plot_elapsed_time, show_plot=show_plot, save_plot_path=save_plot_path)
    return result



def plot_result(result, plot_elapsed_time=False, show_plot=True, save_plot_path=None):
    """
    Plots the result of an experiment.
    
    Parameters
    ----------
    result : Result
        The result to plot.
    plot_elapsed_time : bool, optional
        Indicated whether the elapsed time is plotted instead of the actual
        result.
    show_plot : bool, optional
        Indicates whether pyplot.show() is called or not (default: True).
    save_plot_path : string, optional
        Optional path where resulting plot is saved as a PNG file.
    
    Returns
    -------
    Result
        A result produced by `plot`. 
    """
    
    if len(result.iter_args) <= 0:
        print 'Nothing to plot! At least one argument must be an iterable, e.g., a list of integers.'
        return

    # import here makes evaluate() independent from matplotlib
    from matplotlib import pyplot as plt
    
    # wither work with function values or with elapsed times
    if plot_elapsed_time:
        data = result.elapsed_times
    else:
        data = result.values

    # prepare indices for result array
    indices_per_axis = [[slice(None)]] + [range(len(values)) for values in result.iter_args.values()[1:]]
    index_tuples = list(itertools.product(*indices_per_axis))
    
    # cmap for plots
    cmap = plt.get_cmap('jet')
    
    # line plot or bar plot?
    if _all_numeric(result.iter_args.values()[0]):
        #
        # regular line plot
        #
        x_values = result.iter_args.values()[0]
        for i, index_tuple in enumerate(index_tuples):
            y_values = np.mean(data[index_tuple], axis=-1)
            if result.repetitions > 1:
                y_errors = np.std(data[index_tuple], axis=-1)
                plt.errorbar(x_values, y_values, yerr=y_errors, linewidth=1., color=cmap(1.*i/len(index_tuples)))
            else:
                plt.plot(x_values, y_values, linewidth=1., color=cmap(1.*i/len(index_tuples)))
    else:
        #
        # bar plot
        #
        x_values = np.arange(data.shape[0])
        width = .8 / len(index_tuples)
        for i, index_tuple in enumerate(index_tuples):
            y_values = np.mean(data[index_tuple], axis=-1)
            y_errors = np.std(data[index_tuple], axis=-1)
            plt.bar(x_values + i*width, y_values, yerr=y_errors, width=width, color=cmap(1.*i/len(index_tuples)))
        plt.gca().set_xticks(np.arange(len(x_values)) + .4)
        plt.gca().set_xticklabels([str(value) for value in result.iter_args.values()[0]])

    # legend and labels for axis        
    legend = []
    name_value_lists = [[(name, value) for value in values] for (name, values) in result.iter_args.items()[1:]]
    if len(name_value_lists) >= 1:
        for name_value_combination in itertools.product(*name_value_lists):
            legend.append(str.join(', ', ["%s = %s" % (name, value) for (name, value) in name_value_combination]))
        plt.legend(legend, loc='best')
    iter_arg_name = result.iter_args.keys()[0]
    plt.xlabel(iter_arg_name)
    if plot_elapsed_time:
        plt.ylabel('elapsed time [ms]')
    else:
        plt.ylabel('%s(%s)' % (result.function_name, iter_arg_name))
            
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
    if result.cachedir is not None:
        plotted_args['cachedir'] = result.cachedir
    if plot_elapsed_time:
        plotted_args['plot_elapsed_time'] = plot_elapsed_time
    parameter_text = 'Parameters: %s' % str.join(', ', ['%s=%s' % (k,v) for k,v in plotted_args.items()])
    parameter_text = str.join('\n', textwrap.wrap(parameter_text, 100))
    plt.title('Time: %s - %s (%s)\n' % (time_start_str, time_stop_str, time_delta) + 
              parameter_text,
              fontsize=12)
    plt.subplots_adjust(top=0.8)

    # save plot in file
    if save_plot_path:
        if not os.path.exists(save_plot_path):
            os.makedirs(save_plot_path)
        timestamp = time.strftime('%Y%m%d_%H%M%S', result.time_start)
        number_of_results = len(set([os.path.splitext(f)[0] for f in os.listdir(save_plot_path) if f.startswith(timestamp)]))
        result_prefix = '%s_%02d' % (timestamp, number_of_results)
        plt.savefig('%s/%s.png' % (save_plot_path, result_prefix))

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



def my_experiment(x, y=0, f='sin', shift=False, seed=None, repetition_index=None):
    """
    A simple example for an experiment function.
    """
    unique_seed = abs(hash((x, y, f, shift, seed, repetition_index)))
    np.random.seed(unique_seed)

    time.sleep(1)
    
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
    seed = 0
    repetitions = 10
    show_plot = True
    save_plot = False
    processes = None
    cachedir = '/tmp'
    plot_elapsed_time = False
    
    # regular call of the experiment    
    print my_experiment(x=0, f='sin', seed=seed, shift=False)

    # plot with varying x
    plot(my_experiment, x=range(10), f='sin', seed=seed, shift=False, repetitions=repetitions, show_plot=show_plot, save_plot=save_plot, processes=processes, cachedir=cachedir, plot_elapsed_time=plot_elapsed_time)
    
    # plot with varying x and f
    plot(my_experiment, x=range(10), f=['sin', 'cos'], seed=seed, shift=False, repetitions=repetitions, show_plot=show_plot, save_plot=save_plot, processes=processes, cachedir=cachedir, plot_elapsed_time=plot_elapsed_time)
    
    # plot with varying x as well as f and shift
    plot(my_experiment, x=range(10), f=['sin', 'cos'], seed=seed, shift=[False, True], repetitions=repetitions, show_plot=show_plot, save_plot=save_plot, processes=processes, cachedir=cachedir, plot_elapsed_time=plot_elapsed_time)

    # plot with varying x as well as f and shift, but force a certain order
    plot(my_experiment, x=range(10), f=['sin', 'cos'], seed=seed, shift=[False, True], argument_order=['x', 'f', 'shift'], repetitions=repetitions, show_plot=show_plot, save_plot=save_plot, processes=processes, cachedir=cachedir, plot_elapsed_time=plot_elapsed_time)
    
    # bar plot for x=0 with varying f and shift (as well as forced order of parameters)
    plot(my_experiment, x=0, f=['sin', 'cos'], shift=[False, True], seed=seed, repetitions=repetitions, show_plot=show_plot, save_plot=save_plot, processes=processes, cachedir=cachedir, plot_elapsed_time=plot_elapsed_time)
    plot(my_experiment, x=0, f=['sin', 'cos'], shift=[False, True], seed=seed, argument_order=['f'], repetitions=repetitions, show_plot=show_plot, save_plot=save_plot, processes=processes, cachedir=cachedir, plot_elapsed_time=plot_elapsed_time)
    
    # 2d plot
    plot(my_experiment, x=range(10), y=range(10), seed=seed, repetitions=repetitions, show_plot=show_plot, save_plot=save_plot, processes=processes, cachedir=cachedir, plot_elapsed_time=plot_elapsed_time)



if __name__ == '__main__':
    main()
