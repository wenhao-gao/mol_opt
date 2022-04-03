"""
Unit tests for the genetic algorithm for molecular optimization.
"""
import unittest
import numpy as np
from syn_net.utils.ga_utils import crossover, mutation, fitness_sum


class TestOptimization(unittest.TestCase):
    """
    Tests for molecular optimization using a genetic algorithm: (1) crossover,
    (2) mutation, (3) one generation.
    """
    def test_crossover(self):
        """
        Tests crossover in the genetic algorithm.
        """
        np.random.seed(seed=137)
        num_parents    = 10
        fp_size        = 128
        offspring_size = 30
        population     = np.ceil(np.random.random(size=(num_parents, fp_size)) * 2 - 1)

        offspring      = crossover(parents=population, offspring_size=offspring_size)
        new_scores     = np.array([fitness_sum(_) for _ in offspring])

        new_scores_ref = np.array([69, 72, 65, 62, 79, 70, 70, 60, 62, 71,
                                   61, 79, 65, 63, 73, 66, 66, 64, 69, 71,
                                   74, 67, 64, 64, 67, 66, 56, 58, 69, 69])

        self.assertEqual(new_scores.all(), new_scores_ref.all())

    def test_mutation(self):
        """
        Tests mutation in the genetic algorithm.
        """
        np.random.seed(seed=137)
        num_parents    = 10
        fp_size        = 128
        population     = np.ceil(np.random.random(size=(num_parents, fp_size)) * 2 - 1)

        offspring      = mutation(offspring_crossover=population, num_mut_per_ele=4, mut_probability=0.5)
        new_scores     = np.array([fitness_sum(_) for _ in offspring])

        new_scores_ref = np.array([70, 64, 62, 60, 68, 66, 65, 59, 68, 77,])

        self.assertEqual(new_scores.all(), new_scores_ref.all())

    def test_generation(self):
        """
        Tests a single generation in the genetic algorithm.
        """
        np.random.seed(seed=137)
        num_parents    = 10
        fp_size        = 128
        offspring_size = 30
        ngen           = 3
        population     = np.ceil(np.random.random(size=(num_parents, fp_size)) * 2 - 1)

        scores = [fitness_sum(_) for _ in population]

        for _ in range(ngen):
            offspring      = crossover(parents=population, offspring_size=offspring_size)
            offspring      = mutation(offspring_crossover=offspring, num_mut_per_ele=4, mut_probability=0.5)
            new_population = np.concatenate([population, offspring], axis=0)
            new_scores     = np.array(scores + [fitness_sum(_) for _ in offspring])
            scores         = []

            for parent_idx in range(num_parents):
                max_score_idx = np.where(new_scores == np.max(new_scores))[0][0]
                scores.append(new_scores[max_score_idx])
                population[parent_idx, :] = new_population[max_score_idx, :]
                new_scores[max_score_idx] = -999999

        scores_ref     = np.array([87.0, 86.0, 84.0, 84.0, 84.0,
                                   82.0, 82.0, 82.0, 82.0, 82.0])

        new_scores_ref = np.array([-9.99999e+05, 8.10000e+01, 8.10000e+01, 7.90000e+01, 7.90000e+01,
                                    7.80000e+01, 7.70000e+01, 7.60000e+01, 7.60000e+01, 7.50000e+01,
                                    7.00000e+01, 7.80000e+01, 7.30000e+01, 7.00000e+01, 8.10000e+01,
                                    8.00000e+01,-9.99999e+05, 7.80000e+01, 7.30000e+01,-9.99999e+05,
                                    7.40000e+01,-9.99999e+05, 7.90000e+01, 7.60000e+01, 7.80000e+01,
                                    7.90000e+01, 7.50000e+01, 7.90000e+01,-9.99999e+05,-9.99999e+05,
                                   -9.99999e+05,-9.99999e+05,-9.99999e+05, 7.90000e+01, 7.30000e+01,
                                    7.20000e+01, 7.60000e+01,-9.99999e+05, 7.70000e+01, 8.00000e+01])

        self.assertEqual(np.array(scores).all(), scores_ref.all())
        self.assertEqual(new_scores.all(), new_scores_ref.all())
