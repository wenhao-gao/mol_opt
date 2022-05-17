from functools import total_ordering
import numpy as np
import random


@total_ordering
class StorageElement:
    def __init__(self, smi, score):
        self.smi = smi
        self.score = score

    def __eq__(self, other):
        return np.isclose(self.score, other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __hash__(self):
        return hash(self.smi)


def unravel_elems(elems):
    return tuple(map(list, zip(*[(elem.smi, elem.score) for elem in elems])))


class MaxRewardPriorityQueue:
    def __init__(self):
        self.elems = []

    def __len__(self):
        return len(self.elems)

    def add_list(self, smis, scores):
        new_elems = [StorageElement(smi=smi, score=score) for smi, score in zip(smis, scores)]
        self.elems.extend(new_elems)
        self.elems = list(set(self.elems))

    def get_elems(self):
        return unravel_elems(self.elems)

    def squeeze_by_kth(self, k):
        k = min(k, len(self.elems))
        self.elems = sorted(self.elems, reverse=True)[:k]
        return self.elems[-1].score

    def squeeze_by_thr(self, thr):
        self.elems = sorted(self.elems, reverse=True)
        k = next((i for i, elem in enumerate(self.elems) if elem.score < thr), len(self.elems))

        self.elems = self.elems[:k]
        return unravel_elems(self.elems)

    def sample_batch(self, batch_size):
        sampled_elems = random.choices(population=self.elems, k=batch_size)
        return unravel_elems(sampled_elems)
