import numpy as np
import unittest

import explot as ep



def experiment(x, y=0, f='sin', shift=False, seed=None):
    np.random.seed(seed)
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
    


class TestExPlot(unittest.TestCase):

    def testDefaultValues(self):
        result = ep.evaluate(experiment, x=0)
        self.assertEqual(len(result.kwargs), 5)
        self.assertEqual(result.kwargs['x'], 0)
        self.assertEqual(result.kwargs['y'], 0)
        self.assertEqual(result.kwargs['f'], 'sin')
        self.assertEqual(result.kwargs['shift'], False)
        self.assertEqual(result.kwargs['seed'], None)
        
    def testSeed(self):
        # different when no seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2)
        r2 = ep.evaluate(experiment, x=range(100), repetitions=2)
        self.assertNotEqual(list(r1.values.flatten()), list(r2.values.flatten()))
        # same when same seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0)
        r2 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0)
        self.assertEqual(list(r1.values.flatten()), list(r2.values.flatten()))
        # different when different seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0)
        r2 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=1)
        self.assertNotEqual(list(r1.values.flatten()), list(r2.values.flatten()))
        # different when repeated
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0)
        self.assertNotEqual(list(r1.values[:,0].flatten()), list(r1.values[:,1].flatten()))

    def testSeedValues(self):
        # nan when no seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2)
        self.assertTrue(np.any(np.isnan(r1.seeds)))
        # same when same seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0)
        r2 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0)
        self.assertEqual(list(r1.seeds.flatten()), list(r2.seeds.flatten()))
        # different when different seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0)
        r2 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=1)
        self.assertNotEqual(list(r1.seeds.flatten()), list(r2.seeds.flatten()))
        # different when repeated
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0)
        self.assertNotEqual(list(r1.seeds[:,0].flatten()), list(r1.seeds[:,1].flatten()))


if __name__ == "__main__":
    unittest.main()
    