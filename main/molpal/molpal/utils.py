from itertools import islice
from typing import Iterable, Iterator, List


def batches(it: Iterable, size: int) -> Iterator[List]:
    """Batch an iterable into batches of the given size, with the final batch
    potentially being smaller"""
    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])