import itertools
from typing import Iterable, Union, Tuple, List
import numpy as np
from emukit.core.parameter import Parameter


class CandidateStringParameter(Parameter):
    """
    A class for string inputs consisting of a set of valid strings
    """
    def __init__(self, name: str, candidates: np.ndarray):
        """
        :param name: Name of parameter
        :candidates: set of possible strings (np.array of shape (n,1) ) needs spaces between characters
        """
        self.name = name
        self.candidates = candidates
        # get length of longest string (to init SSK)
        self.length = np.max([len("".join(candidate[0].split(" "))) for candidate in candidates])
        # get alphabet of characters used in candidate set (to init SSK)
        self.alphabet = list({l for word in candidates for l in word[0]})

    def sample_uniform(self, point_count: int=1) -> np.ndarray:
        """
        Generates multiple random strings from candidate set
        :param point_count: number of data points to generate.
        :returns: Generated points with shape (point_count, 1)
        """
        samples = np.random.randint(0,self.candidates.shape[0],(point_count,1))
        samples = self.candidates[samples]
        return np.array(samples).reshape(-1,1)


