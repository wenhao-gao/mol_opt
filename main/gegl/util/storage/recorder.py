import os
from functools import total_ordering
import numpy as np
import random
import json
# import neptune

from util.chemistry.rd_filter import RDFilter


@total_ordering
class RecorderElement:
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


class Recorder:
    def __init__(self, scoring_num_list, record_filtered=True, prefix=""):
        self.elems = []
        self.filtered_elems = []
        self.seen_smis = set()
        self.record_filtered = record_filtered
        if self.record_filtered:
            self.rd_filter = RDFilter()

        self.scoring_num_list = scoring_num_list
        self.prefix = prefix

        self.max_size = max(scoring_num_list)
        self.t = 0

    def __len__(self):
        return len(self.elems)

    def add_list(self, smis, scores):
        new_elems = [RecorderElement(smi=smi, score=score) for smi, score in zip(smis, scores)]
        new_elems = list(set(new_elems))
        new_elems = list(filter(lambda elem: elem.smi not in self.seen_smis, new_elems))
        self.seen_smis = self.seen_smis.union(smis)

        self.elems.extend(new_elems)
        self.elems = list(set(self.elems))

        if len(self.elems) > self.max_size:
            self.elems = sorted(self.elems, reverse=True)[: self.max_size]

        if self.record_filtered:
            filtered_new_elems = new_elems
            if len(self.filtered_elems) > 0:
                min_filtered_elem_score = min(self.filtered_elems).score
                filtered_new_elems = list(
                    filter(lambda elem: (elem.score > min_filtered_elem_score), filtered_new_elems)
                )

            if len(filtered_new_elems) > self.max_size:
                filtered_new_elems = sorted(filtered_new_elems, reverse=True)[: self.max_size]

            filtered_new_elems = list(
                filter(lambda elem: self.rd_filter(elem.smi) > 0.5, filtered_new_elems)
            )

            self.filtered_elems.extend(filtered_new_elems)

            if len(self.filtered_elems) > self.max_size:
                self.filtered_elems = sorted(self.filtered_elems, reverse=True)[: self.max_size]

    def full(self):
        return len(self.elems) == self.max_size and (
            (not self.record_filtered)
            or (self.record_filtered and len(self.filtered_elems) == self.max_size)
        )

    def evaluate(self, rd_filtered):
        if not rd_filtered:
            scores = [elem.score for elem in sorted(self.elems, reverse=True)]
        else:
            scores = [elem.score for elem in sorted(self.filtered_elems, reverse=True)]

        evaluation_elemwise_scores = np.array(scores)

        evaluation_score = 0.0
        for scoring_num in self.scoring_num_list:
            evaluation_score += evaluation_elemwise_scores[:scoring_num].mean() / len(
                self.scoring_num_list
            )

        return evaluation_score

    def log(self):
        score = self.evaluate(rd_filtered=False)
        # neptune.log_metric(f"{self.prefix}eval_optimized_score", score)

        if self.record_filtered:
            filtered_score = self.evaluate(rd_filtered=True)
            # neptune.log_metric(f"{self.prefix}eval_filtered_score", filtered_score)

        self.t += 1

    def log_final(self):
        for elem in self.elems:
            pass 
            # neptune.log_text("optimized_smi", elem.smi)
            # neptune.log_metric("optimized_score", elem.score)

        if self.record_filtered:
            for elem in self.filtered_elems:
                pass 
                # neptune.log_text("filtered_smi", elem.smi)
                # neptune.log_metric("filtered_score", elem.score)

    def get_topk(self, k):
        self.elems = sorted(self.elems, reverse=True)[:k]
        return [elem.smi for elem in self.elems], [elem.score for elem in self.elems]
