import numpy as np
import unittest

import explot as ep



def experiment(x, y=0, f='sin', shift=False, dummy=None, seed=None):
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


def experiment_repidx(x, seed, repetition_index):
    np.random.seed(seed)
    return x + np.random.randn() + 1e6*repetition_index


class TestExPlot(unittest.TestCase):

    def testDefaultValues(self):
        result = ep.evaluate(experiment, x=0)
        self.assertEqual(len(result.kwargs), 6)
        self.assertEqual(result.kwargs['x'], 0)
        self.assertEqual(result.kwargs['y'], 0)
        self.assertEqual(result.kwargs['f'], 'sin')
        self.assertEqual(result.kwargs['shift'], False)
        self.assertEqual(result.kwargs['dummy'], None)
        self.assertEqual(result.kwargs['seed'], None)
        
    def testSeed(self):
        """
        Tests whether results are the same/different as expected depending on
        seed.
        """
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

    def testSeedManageExternal(self):
        """
        Tests whether results are the same/different as expected depending on
        externally managed seed.
        """
        # different when no seed
        r1 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, manage_seed='external')
        r2 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, manage_seed='external')
        self.assertNotEqual(list(r1.values.flatten()), list(r2.values.flatten()))
        # same when same seed
        r1 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, seed=0, manage_seed='external')
        r2 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, seed=0, manage_seed='external')
        self.assertEqual(list(r1.values.flatten()), list(r2.values.flatten()))
        # different when different seed
        r1 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, seed=0, manage_seed='external')
        r2 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, seed=1, manage_seed='external')
        self.assertNotEqual(list(r1.values.flatten()), list(r2.values.flatten()))
        # different when repeated
        r1 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, seed=0, manage_seed='external')
        self.assertNotEqual(list(r1.values[:,0].flatten()), list(r1.values[:,1].flatten()))

    def testSeedValuesManageNo(self):
        """
        Tests whether the seed values make sense when manage_seed set to 'no'.
        """
        # nan when no seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, manage_seed='no')
        self.assertTrue(np.any(np.isnan(r1.seeds)))
        # same as given seed, also during repetitions
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0, manage_seed='no')
        self.assertTrue(np.allclose(r1.seeds.flatten(), 0))
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=1, manage_seed='no')
        self.assertTrue(np.allclose(r1.seeds.flatten(), 1))
        
    def testSeedValuesManageAuto(self):
        """
        Tests whether the auto-generated seed values make sense.
        """
        # nan when no seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, manage_seed='auto')
        self.assertTrue(np.any(np.isnan(r1.seeds)))
        # same when same seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0, manage_seed='auto')
        r2 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0, manage_seed='auto')
        self.assertEqual(list(r1.seeds.flatten()), list(r2.seeds.flatten()))
        # different when different seed
        r1 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=0, manage_seed='auto')
        r2 = ep.evaluate(experiment, x=range(100), repetitions=2, seed=1, manage_seed='auto')
        self.assertNotEqual(list(r1.seeds.flatten()), list(r2.seeds.flatten()))
        # different when repeated
        r1 = ep.evaluate(experiment, x=range(100), repetitions=3, seed=0, manage_seed='auto')
        self.assertNotEqual(list(r1.seeds[:,1].flatten()), list(r1.seeds[:,2].flatten()))
        # seed not changed in first repetition
        self.assertTrue(np.allclose(r1.seeds[:,0], 0))
        
    def testSeedValuesManageExternal(self):
        """
        Tests whether the seed values make sense when manage_seed set to 'external'.
        """
        # nan when no seed
        r1 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, manage_seed='external')
        self.assertTrue(np.any(np.isnan(r1.seeds)))
        # same as given seed, also during repetitions
        r1 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, seed=0, manage_seed='external')
        self.assertTrue(np.allclose(r1.seeds.flatten(), 0))
        r1 = ep.evaluate(experiment_repidx, x=range(100), repetitions=2, seed=1, manage_seed='external')
        self.assertTrue(np.allclose(r1.seeds.flatten(), 1))
        
    def testScriptName(self):
        r = ep.evaluate(experiment, x=0)
        self.assertTrue(r.script.endswith('explot_test.py'))
        
    def testArgumentSeed(self):
        """
        Tests whether calc_argument_seed takes into account all the right 
        values.
        """
        def f(a, b=0, c=0, **kwargs):
            return ep.calc_argument_hash()
        reference_seed = hash(frozenset({'a': 42, 'b': 0, 'c': 1, 'x': False})) % np.iinfo(np.uint32).max
        self.assertEqual(f(a=42, c=1, x=False)[0], reference_seed)
        
    def testIgnoreArguments(self):
        r1 = ep.evaluate(experiment, x=range(10), seed=0)
        r2 = ep.evaluate(experiment, x=range(10), dummy=(0,1), argument_order=['x'], seed=0)
        r3 = ep.evaluate(experiment, x=range(10), dummy=(0,1), argument_order=['x'], ignore_arguments=['dummy'], seed=0)
        self.assertFalse(np.all(r1.values==r2.values))
        self.assertNotEqual(r1.iter_args, r2.iter_args)
        self.assertTrue(np.all(r1.values==r3.values))
        self.assertEqual(r1.iter_args, r3.iter_args)
        


if __name__ == "__main__":
    unittest.main()
    