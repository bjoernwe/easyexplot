# EasyExPlot #

When running scientific experiments, you can often find yourself in the situation of calling the same functions again and again - only with different arguments. The code for these repeated experiments and their visualization easily becomes cumbersome and highly redundant. That's where EasyExPlot can help!

EasyExPlot is a small Python tool that helps you plot your experiments with different combinations of input arguments. The idea is to automatically recognize which arguments are varied between experiments, run all the experiments parallelized over all the cores of your machine, and visualize the results with properly labeled axes and everything.

### Usage ###

Let's assume we have some experiment, which is nothing more than a real-valued function with different parameters.


```
#!python

def my_experiment(x=0, f='sin', shift=False):
    # calculate some stuf
    # ...
    return result_value
```

Now, evaluating and plotting this function for different arguments works as simple as this:

```
#!python

import easyexplot as eep

# plot function for all values of x in range(10)
eep.plot(my_experiment, x=range(10), f='sin')
```

EasyExPlot automatically recognizes that parameter *x* is the one to iterate over and produces the following plot:

<img src="https://raw.githubusercontent.com/bjoernwe/easyexplot/master/README/20150122_163422_00.png" width="640px">

Suspecting that the function is noisy, we let EasyExPlot evaluate the function multiple times with the argument *repetitions*:

```
#!python
eep.plot(my_experiment, x=range(10), f='sin', repetitions=50)
```

![20150122_142548_00.png](https://bitbucket.org/repo/nX7gry/images/3251664006-20150122_142548_00.png)

Nice, we have mean and standard deviation now! But how do the other parameters (*f* and *shift*) influence the result of the experiment? Again, we replace the arguments in question with lists of possible values and EasyExPlot takes care of running the experiment for different combinations of arguments.

```
#!python
eep.plot(my_experiment, x=range(10), f=['sin', 'cos'], shift=[False, True], repetitions=50)
```

![20150122_142811_00.png](https://bitbucket.org/repo/nX7gry/images/1825501275-20150122_142811_00.png)

That was easy. But what if the experiment doesn't have a numeric parameter like *x*? Then the result is plotted as bars:

```
#!python
eep.plot(my_experiment, f=['sin', 'cos'], shift=[False, True], repetitions=50)
```

![20150122_143556_00.png](https://bitbucket.org/repo/nX7gry/images/3575699363-20150122_143556_00.png)
