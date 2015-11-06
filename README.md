# EasyExPlot #

When running scientific experiments, I often find myself in the situation of calling the same functions again and again - only with different arguments. The code for these repeated experiments and their visualization easily becomes cumbersome and highly redundant. That's where EasyExPlot is supposed to help.

EasyExPlot is a small Python tool that helps you plot your experiments with different combinations of input arguments. The idea is to automatically recognize which arguments are varied between experiments, run all the experiments parallelized over all the cores of your machine, and visualize the results with properly labeled axes and everything.

### Usage ###

Let's assume we have some experiment, which is nothing more than a real-valued function with different parameters.


```python
def my_experiment(x=0, f='sin', shift=False):
    # calculate some stuf
    # ...
    return result_value
```

Now, evaluating and plotting this function for different arguments works as simple as this:

```python
import easyexplot as eep

# plot function for all values of x in range(10)
eep.plot(my_experiment, x=range(10), f='sin')
```

EasyExPlot automatically recognizes that parameter *x* is the one to iterate over and produces the following plot:

<img src="https://raw.githubusercontent.com/bjoernwe/easyexplot/master/README/20150122_163422_00.png" width="640px">

Note how all arguments are automatically included in the plot to make the comparison of different plots easier.

Suspecting that the function is noisy, we let EasyExPlot evaluate the function multiple times with the argument *repetitions*:

```python
eep.plot(my_experiment, x=range(10), f='sin', repetitions=100)
```

<img src="https://raw.githubusercontent.com/bjoernwe/easyexplot/master/README/20150122_163542_00.png" width="640px">

Nice, we have mean and standard deviation now! But how do the other parameters (*f* and *shift*) influence the result of the experiment? Again, we replace the arguments in question with lists of possible values and EasyExPlot takes care of running the experiment for different combinations of arguments.

```python
eep.plot(my_experiment, x=range(10), f=['sin', 'cos'], 
         shift=[False, True], repetitions=100)
```

<img src="https://raw.githubusercontent.com/bjoernwe/easyexplot/master/README/20150122_163547_00.png" width="640px">

That was easy. But what if the result doesn't have varying numeric parameters like *x*? Then the result is plotted as bars:

```python
eep.plot(my_experiment, f=['sin', 'cos'], shift=[False, True], 
         repetitions=100)
```

<img src="https://raw.githubusercontent.com/bjoernwe/easyexplot/master/README/20150122_163600_00.png" width="640px">

Enjoy!
