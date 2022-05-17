import random
from rdkit import Chem

from joblib import delayed

from model.genetic_operator.mutate import mutate
from model.genetic_operator.crossover import crossover

import gc


def reproduce(parent_a, parent_b, mutation_rate):
    parent_a, parent_b = Chem.MolFromSmiles(parent_a), Chem.MolFromSmiles(parent_b)
    new_child = crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mutate(new_child, mutation_rate)

    smis = Chem.MolToSmiles(new_child, isomericSmiles=True) if new_child is not None else None

    return smis


class GeneticOperatorHandler:
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def query(self, query_size, mating_pool, pool):
        smis = random.choices(mating_pool, k=query_size * 2)
        smi0s, smi1s = smis[:query_size], smis[query_size:]
        smis = pool(
            delayed(reproduce)(smi0, smi1, self.mutation_rate) for smi0, smi1 in zip(smi0s, smi1s)
        )
        smis = list(filter(lambda smi: smi is not None, smis))
        gc.collect()

        return smis
