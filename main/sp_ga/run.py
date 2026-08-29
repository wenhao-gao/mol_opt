"""Surrogate-prescreened Graph GA for the official PMO ``mol_opt`` API.

The surrogate only ranks unevaluated children. Population fitness, logs, and
benchmark metrics always use values returned by the real PMO oracle.
"""

from __future__ import print_function

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

from main.graph_ga.run import make_mating_pool, reproduce
from main.optimizer import BaseOptimizer

rdBase.DisableLog("rdApp.error")

FP_SIZE = 1024


def _fingerprint(molecule):
    return AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius=2, nBits=FP_SIZE
    )


def _canonical(smiles):
    # type: (str) -> Optional[str]
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(
        molecule, canonical=True
    )


def _fp_array(smiles_list):
    # type: (List[str]) -> np.ndarray
    matrix = np.zeros((len(smiles_list), FP_SIZE), dtype=np.uint8)
    for row, smiles in enumerate(smiles_list):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(
                "invalid candidate reached surrogate: {!r}".format(smiles)
            )
        DataStructs.ConvertToNumpyArray(
            _fingerprint(molecule), matrix[row]
        )
    return matrix


def _rank(smiles, scores, population_size):
    # type: (List[str], List[float], int) -> Tuple[List[str], List[float]]
    unique = {}  # type: Dict[str, float]
    for candidate, score in zip(smiles, scores):
        canonical = _canonical(candidate)
        if canonical is None:
            continue
        previous = unique.get(canonical)
        if previous is None or float(score) > previous:
            unique[canonical] = float(score)
    ranked = sorted(unique.items(), key=lambda item: item[1], reverse=True)
    ranked = ranked[:population_size]
    return [item[0] for item in ranked], [item[1] for item in ranked]


def _children(population, scores, count, mutation_rate):
    # type: (List[str], List[float], int, float) -> List[str]
    molecules = [Chem.MolFromSmiles(smiles) for smiles in population]
    if any(molecule is None for molecule in molecules):
        raise ValueError("population contains invalid SMILES")
    minimum = min(scores)
    weights = [float(score) - minimum + 1e-10 for score in scores]
    mating_pool = make_mating_pool(
        molecules, weights, max(count, len(molecules))
    )
    generated = []  # type: List[str]
    for _ in range(count):
        child = reproduce(mating_pool, mutation_rate)
        if child is None:
            continue
        try:
            generated.append(Chem.MolToSmiles(child, canonical=True))
        except (RuntimeError, ValueError):
            continue
    return generated


def _novel_candidates(raw, seen, limit):
    # type: (List[str], Set[str], int) -> List[str]
    selected = []  # type: List[str]
    local_seen = set()  # type: Set[str]
    for smiles in raw:
        canonical = _canonical(smiles)
        if canonical is None or canonical in seen or canonical in local_seen:
            continue
        local_seen.add(canonical)
        selected.append(canonical)
        if len(selected) >= limit:
            break
    return selected


class SP_GA_Optimizer(BaseOptimizer):
    def __init__(self, args=None):
        super(SP_GA_Optimizer, self).__init__(args)
        self.model_name = "sp_ga"

    def _prescreen(self, candidates, population, needed, config):
        # type: (List[str], List[str], int, dict) -> List[str]
        train_smiles = list(self.oracle.mol_buffer)
        train_scores = np.asarray(
            [self.oracle.mol_buffer[smiles][0] for smiles in train_smiles]
        )
        model = RandomForestRegressor(
            n_estimators=int(config["n_trees"]),
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=self.n_jobs,
            random_state=self.seed + len(self.oracle),
        )
        model.fit(_fp_array(train_smiles), train_scores)
        candidate_features = _fp_array(candidates)
        tree_predictions = np.vstack(
            [tree.predict(candidate_features) for tree in model.estimators_]
        )
        mean = tree_predictions.mean(axis=0)
        uncertainty = tree_predictions.std(axis=0)
        reference = [
            _fingerprint(Chem.MolFromSmiles(smiles))
            for smiles in population[:32]
        ]
        novelty = np.asarray(
            [
                1.0
                - max(
                    DataStructs.BulkTanimotoSimilarity(
                        _fingerprint(Chem.MolFromSmiles(smiles)),
                        reference,
                    )
                )
                for smiles in candidates
            ]
        )
        acquisition = (
            mean
            + float(config["uncertainty_weight"]) * uncertainty
            + float(config["novelty_weight"]) * novelty
        )
        order = np.argsort(-acquisition)[:needed]
        return [candidates[int(index)] for index in order]

    def _fallback(self, needed):
        # type: (int) -> List[str]
        seen = set(self.oracle.mol_buffer)
        order = np.random.permutation(len(self.all_smiles))
        candidates = []  # type: List[str]
        for index in order:
            canonical = _canonical(self.all_smiles[int(index)])
            if canonical is None or canonical in seen:
                continue
            candidates.append(canonical)
            seen.add(canonical)
            if len(candidates) >= needed:
                break
        return candidates

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        population_size = int(config["population_size"])
        starting = np.random.choice(
            self.all_smiles, population_size, replace=False
        ).tolist()
        population_scores = self.oracle(starting)
        population, population_scores = _rank(
            starting, population_scores, population_size
        )

        while not self.finish:
            remaining = self.oracle.max_oracle_calls - len(self.oracle)
            needed = min(int(config["batch_size"]), remaining)
            pool_size = needed * int(config["candidate_multiplier"])
            candidates = []  # type: List[str]
            attempts = 0
            while len(candidates) < pool_size and attempts < 10:
                raw = _children(
                    population,
                    population_scores,
                    pool_size - len(candidates),
                    float(config["mutation_rate"]),
                )
                candidates = _novel_candidates(
                    candidates + raw,
                    set(self.oracle.mol_buffer),
                    pool_size,
                )
                attempts += 1

            if not candidates:
                candidates = self._fallback(needed)
            if (
                len(self.oracle) >= int(config["surrogate_start"])
                and len(candidates) > needed
            ):
                selected = self._prescreen(
                    candidates, population, needed, config
                )
            else:
                selected = candidates[:needed]
            if not selected:
                raise RuntimeError(
                    "SP-GA could not generate an unevaluated molecule"
                )
            selected_scores = self.oracle(selected)
            population, population_scores = _rank(
                population + selected,
                population_scores + selected_scores,
                population_size,
            )

            if self.finish:
                break
