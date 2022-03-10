import csv
from functools import partial
import os
import random
import string
import unittest

from molpal.objectives.lookup import LookupObjective

class TestLookupObjective(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alphabet = {c: i for i, c in enumerate(string.ascii_lowercase)}

        cls.random_xs = random.sample(cls.alphabet.keys(), 10)
        cls.random_items = {c: cls.alphabet[c] for c in cls.random_xs}

        cls.empty_csv = 'test_lookup_empty.csv'
        cls.singleton_csv = 'test_lookup_singleton.csv'
        cls.normal_csv = 'test_lookup_normal.csv'
        cls.weird_csv = 'test_lookup_weird.csv'

        with open(cls.empty_csv, 'w') as fid:
            pass
        with open(cls.singleton_csv, 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(['a', cls.alphabet['a']])
        with open(cls.normal_csv, 'w') as fid:
            writer = csv.writer(fid)
            writer.writerows(cls.alphabet.items())
        with open(cls.weird_csv, 'w') as fid:
            writer = csv.writer(fid)
            for c in 'abc':
                writer.writerow([None, cls.alphabet[c], None, None, c])

        cls.empty = LookupObjective(cls.empty_csv, lookup_title_line=False)
        cls.singleton = LookupObjective(cls.singleton_csv, 
                                         lookup_title_line=False)
        cls.normal = LookupObjective(cls.normal_csv, lookup_title_line=False)
        cls.weird = LookupObjective(cls.weird_csv, lookup_title_line=False,
                                     lookup_smiles_col=4, lookup_data_col=1)

    def test_empty(self):
        scores = self.empty.calc(self.random_xs)
        self.assertEqual(scores, {x: None for x in self.random_xs})

    def test_singleton(self):
        scores_a = self.singleton.calc('a')
        self.assertEqual(scores_a, {'a': self.alphabet['a']})

    def test_singleton_not_contained(self):
        scores_b = self.singleton.calc('b')
        self.assertNotEqual(scores_b, {'b': self.alphabet['b']})

    def test_normal_all_contained(self):
        scores = self.normal.calc(self.random_xs)
        self.assertEqual(scores, {x: self.alphabet[x] for x in self.random_xs})
    
    def test_normal_none_contained(self):
        not_contained_strings = ['foo', 'bar', 'baz', 'qux']
        scores = self.normal.calc(not_contained_strings)
        self.assertEqual(scores, {s: None for s in not_contained_strings})
    
    def test_normal_some_contained(self):
        xs = ['foo', 'bar', 'baz', 'qux']
        xs.extend(self.random_xs)
        scores = self.normal.calc(xs)
        for k, v in scores.items():
            if k in self.alphabet:
                self.assertEqual(v, self.alphabet[k])
            else:
                self.assertIsNone(v)

    def test_weird(self):
        scores = self.weird.calc('abc')
        self.assertEqual(scores, {x: self.alphabet[x] for x in 'abc'})

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.empty_csv)
        os.unlink(cls.singleton_csv)
        os.unlink(cls.normal_csv)
        os.unlink(cls.weird_csv)

if __name__ == "__main__":
    unittest.main()