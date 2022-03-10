from itertools import chain, product
from pathlib import Path
import tempfile
from typing import List, Tuple, TypeVar
T = TypeVar('T')
U = TypeVar('U')

def get_temp_file():
    p_tmp = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    return str(p_tmp)

def distribute_and_flatten(
        xs_yss: List[Tuple[T, List[U]]]) -> List[Tuple[T, U]]:
    """Distribute x over a list ys for each item in a list of 2-tuples and
    flatten the resulting lists.

    For each item in a list of Tuple[T, List[U]], distribute T over the List[U]
    to generate a List[Tuple[T, U]] and flatten the resulting list of lists.

    Example
    -------
    >>> xs_yys = [(1, ['a','b','c']), (2, ['d']), (3, ['e', 'f'])]
    >>> distribute_and_flatten(xs_yss)
    [(1, 'a'), (1, 'b'), (1, 'c'), (2, 'd'), (3, 'e'), (3, 'f')]

    Returns
    -------
    List[Tuple[T, U]]
        the distributed and flattened list
    """
    return list(chain(*[product([x], ys) for x, ys in xs_yss]))