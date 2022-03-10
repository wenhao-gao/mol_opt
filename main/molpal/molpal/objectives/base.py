from abc import ABC, abstractmethod
from typing import Collection, Dict, Optional, TypeVar

T = TypeVar("T")


class Objective(ABC):
    """An Objective is a class for calculating the objective function.

    An Objective indicates values failed to be scored for any reason with a
    value of None. Classes that implement the objective interface should not
    utilize None in any other way.

    Attributes
    ----------
    c : int
        Externally, the objective is always maximized, so all values returned
        inside the Objective are first multiplied by c before being exposed to
        a client of the Objective. If an objective is to be minimized, then c
        is set to -1, otherwise it is set to 1.
    """

    def __init__(self, minimize: bool = False, **kwargs):
        self.c = -1 if minimize else 1

    def __call__(self, *args, **kwargs) -> Dict[T, Optional[float]]:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, xs: Collection[T], *args, **kwargs) -> Dict[T, Optional[float]]:
        """Calculate the objective function for a collection of inputs"""
