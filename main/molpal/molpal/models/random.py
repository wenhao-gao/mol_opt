from typing import Optional, Sequence, Tuple

import numpy as np

from main.molpal.molpal.models import Model

class RandomModel(Model):
    """A baseline model that returns values at random

    Attributes (instance)
    ----------
    rg : np.random.Generator
        the model's random number generator
    
    Parameters
    ----------
    seed : Optional[int], default=None
        the seed for the random number generator
    """
    def __init__(self, seed: Optional[int] = None, **kwargs):
        self.rg = np.random.default_rng(seed)

        super().__init__(**kwargs)

    @property
    def provides(self):
        return {'means', 'vars'}

    @property
    def type_(self):
        return 'random'

    def train(self, *args, **kwargs):
        return True

    def get_means(self, xs: Sequence) -> np.ndarray:
        return self.rg.random(len(xs))

    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        return self.rg.random(len(xs)), self.rg.random(len(xs))
