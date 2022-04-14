import itertools
from typing import Iterable, Union, Tuple, List
import numpy as np
from emukit.core.parameter import Parameter


class StringParameter(Parameter):
    """
    A class for string inputs of fized length and fixed alphabet
    """
    def __init__(self, name: str, alphabet: list,length: int):
        """
        :param name: Name of parameter
        :length: length of strings
        :alphabet: possible characters to make up string
        """
        self.name = name
        self.alphabet = alphabet
        self.length = length

    def sample_uniform(self, point_count: int=1) -> np.ndarray:
        """
        Generates multiple random strings from the grammar
        :param point_count: number of data points to generate.
        :returns: Generated points with shape (point_count, 1)
        """
        samples = np.random.randint(0,len(self.alphabet),(point_count,self.length))
        samples = np.array(self.alphabet)[samples]
        samples=np.array([" ".join(x) for x in samples])
        return np.array(samples).reshape(-1,1)


