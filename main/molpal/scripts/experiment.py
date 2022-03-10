from collections import Counter
import csv
import heapq
from itertools import islice
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np

sys.path.append('../molpal')
from molpal.acquirer import metrics

Point = Tuple[str, float]

class Experiment:
    """An Experiment represents the output of a MolPAL run

    It can be queried for the progress at a given iteration, the order in which
    points were acquired, and other conveniences
    """
    def __init__(self, experiment: Union[Path, str],
                 d_smi_idx: Optional[Dict] = None):
        self.experiment = Path(experiment)
        self.d_smi_idx = d_smi_idx

        try:
            chkpts_dir = self.experiment / 'chkpts'
            self.chkpts = sorted(
                chkpts_dir.iterdir(), key=lambda p: int(p.stem.split('_')[-1])
            )
            config = Experiment.read_config(self.experiment / 'config.ini')
            self.k = int(config['top-k'])
            self.metric = config['metric']
            self.beta = float(config.get('beta', 2.))
            self.xi = float(config.get('xi', 0.001))

            self.new_style = True
        except FileNotFoundError:
            self.new_style = False

        data_dir = self.experiment / 'data'
        final_csv = data_dir / 'all_explored_final.csv'
        final_scores, final_failures = Experiment.read_scores(final_csv)
        self.__size = len({**final_scores, **final_failures})

        scores_csvs = [p for p in data_dir.iterdir() if 'final' not in p.stem]
        self.scores_csvs = sorted(
            scores_csvs, key=lambda p: int(p.stem.split('_')[-1])
        )

    def __len__(self) -> int:
        """the total number of inputs sampled in this experiment"""
        return self.__size

    def __getitem__(self, i: int) -> Dict:
        """Get the score data for iteration i, where i=0 is the
        initialization batch"""
        scores, failures = Experiment.read_scores(self.scores_csvs[i])

        return {**scores, **failures}
    
    def __iter__(self) -> Iterable[Dict]:
        """iterate through all the score data at each iteration"""
        for scores_csv in self.scores_csvs:
            scores, failures = Experiment.read_scores(scores_csv)
            yield {**scores, **failures}
    
    @property
    def num_iters(self) -> int:
        """the total number of iterations in this experiment, including the
        initialization batch"""
        return len(self.scores_csvs)

    @property
    def init_size(self) -> int:
        """the size of this experiment's initialization batch"""
        return len(self[0])

    def get(self, i: int, N: Optional[int] = None) -> Dict:
        scores, failures = Experiment.read_scores(self.scores_csvs[i], N)
        return {**scores, **failures}

    def new_points_by_epoch(self) -> List[Dict]:
        """get the set of new points acquired at each iteration in the list of 
        scores_csvs that are already sorted by iteration"""
        new_points_by_epoch = []
        all_points = {}

        for scores in self:
            new_points = {smi: score for smi, score in scores.items()
                          if smi not in all_points}
            new_points_by_epoch.append(new_points)
            all_points.update(new_points)
        
        return new_points_by_epoch

    def predictions(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """get the predictions for exploration iteration i.

        The exploration iterations are 1-indexed to account for the 
        iteration of the initialization batch. So self.predictions[1] corresponds to the first exploration iteteration
        
        Returns
        -------
        means : np.ndarray
        vars : np.ndarray

        Raises
        ------
        ValueError
            if i is less than 1
        """
        if i not in range(1, self.num_iters):
            raise ValueError(
                f'arg: i must be in {{1..{self.num_iters}}}. got {i}'
            )
        preds_npz = np.load(self.chkpts[i] / 'preds.npz')

        return preds_npz['Y_pred'], preds_npz['Y_var']

    def utilities(self) -> List[np.ndarray]:
        if not self.new_style:
            raise NotImplementedError(
                'Utilities cannot be calculated for an old style MolPAL run'
            )
        Us = []

        for i in range(1, self.num_iters):
            Y_pred, Y_var = self.predictions(i)
            ys = list(self[i-1].values())

            Y = np.nan_to_num(np.array(ys, dtype=float), nan=-np.inf)
            current_max = np.partition(Y, -self.k)[-self.k]

            Us.append(metrics.calc(
                self.metric, Y_pred, Y_var,
                current_max, 0., self.beta, self.xi, False
            ))

        return Us

    def points_in_order(self) -> List[Point]:
        """Get all points acquired during this experiment's run in the order
        in which they were acquired"""
        if self.d_smi_idx is None:
            raise NotImplementedError(
                'Cannot get points in order without setting "self.d_smi_idx"'
            )
        if not self.new_style:
            raise NotImplementedError(
                'Cannot get points in order for an old style MolPAL run'
            )
        init_batch, *exp_batches = self.new_points_by_epoch()

        all_points_in_order = []
        all_points_in_order.extend(init_batch.items())

        for new_points, U in zip(exp_batches, self.utilities()):
            us = np.array([U[self.d_smi_idx[smi]] for smi in new_points])

            new_points_in_order = [
                smi_score for _, smi_score in sorted(
                    zip(us, new_points.items()), reverse=True
                )
            ]
            all_points_in_order.extend(new_points_in_order)
        
        return all_points_in_order
    
    def reward_curve(
        self, true_top_k: List[Point], reward: str = 'scores'
    ):
        """Calculate the reward curve of a molpal run

        Parameters
        ----------
        experiment : Experiment
            the data structure corresponding to the MolPAL experiment
        true_top_k : List
            the list of the true top-k molecules as tuples of their SMILES string
            and associated score
        reward : str, default='scores'
            the type of reward to calculate

        Returns
        -------
        np.ndarray
            the reward as a function of the number of molecules sampled
        """
        all_points_in_order = self.points_in_order()
        k = len(true_top_k)

        if reward == 'scores':
            _, true_scores = zip(*true_top_k)
            missed_scores = Counter(true_scores)

            all_hits_in_order = np.zeros(len(all_points_in_order), dtype=bool)
            for i, (_, score) in enumerate(all_points_in_order):
                if score not in missed_scores:
                    continue
                all_hits_in_order[i] = True
                missed_scores[score] -= 1
                if missed_scores[score] == 0:
                    del missed_scores[score]
            reward_curve = 100 * np.cumsum(all_hits_in_order) / k

        elif reward == 'smis':
            true_top_k_smis = {smi for smi, _ in true_top_k}
            all_hits_in_order = np.array([
                smi in true_top_k_smis
                for smi, _ in all_points_in_order
            ], dtype=bool)
            reward_curve = 100 * np.cumsum(all_hits_in_order) / k

        elif reward == 'top-k-ave':
            reward_curve = np.zeros(len(all_points_in_order), dtype='f8')
            heap = []

            for i, (_, score) in enumerate(all_points_in_order[:k]):
                if score is not None:
                    heapq.heappush(heap, score)
                top_k_avg = sum(heap) / k
                reward_curve[i] = top_k_avg
            reward_curve[:k] = top_k_avg

            for i, (_, score) in enumerate(all_points_in_order[k:]):
                if score is not None:
                    heapq.heappushpop(heap, score)

                top_k_avg = sum(heap) / k
                reward_curve[i+k] = top_k_avg

        elif reward == 'total-ave':
            _, all_scores_in_order = zip(*all_points_in_order)
            Y = np.array(all_scores_in_order, dtype=float)
            Y = np.nan_to_num(Y)
            N = np.arange(0, len(Y)) + 1
            reward_curve = np.cumsum(Y) / N
            
        else:
            raise ValueError

        return reward_curve

    def calculate_reward(
        self, i: int, true: List[Point], is_sorted = True,
        avg: bool = True, smis: bool = True, scores: bool = True
    ) -> Tuple[float, float, float]:
        N = len(true)
        if not is_sorted:
            true = sorted(true, key=lambda kv: kv[1], reverse=True)
        
        found = list(self.get(i, N).items())

        found_smis, found_scores = zip(*found)
        true_smis, true_scores = zip(*true)

        if avg:
            found_avg = np.mean(found_scores)
            true_avg = np.mean(true_scores)
            f_avg = found_avg / true_avg
        else:
            f_avg = None

        if smis:
            found_smis = set(found_smis)
            true_smis = set(true_smis)
            correct_smis = len(found_smis & true_smis)
            f_smis = correct_smis / len(true_smis)
        else:
            f_smis = None

        if scores:
            missed_scores = Counter(true_scores)
            missed_scores.subtract(found_scores)
            n_missed_scores = sum(
                count if count > 0 else 0
                for count in missed_scores.values()
            )
            f_scores = (N - n_missed_scores) / N
        else:
            f_scores = None

        return f_avg, f_smis, f_scores

    def calculate_cluster_fraction(
        self, i: int, true_clusters: Tuple[Set, Set, Set]
    ) -> Tuple[float, float, float]:
        # import pdb; pdb.set_trace()

        large, mids, singletons = true_clusters
        N = len(large) + len(mids) + len(singletons)
        
        found = set(list(self.get(i, N).keys()))

        f_large = len(found & large) / len(large)
        f_mids = len(found & mids) / len(mids)
        f_singletons = len(found & singletons) / len(singletons)

        return f_large, f_mids, f_singletons

    @staticmethod
    def read_scores(scores_csv: Union[Path, str],
                    N: Optional[int] = None) -> Tuple[Dict, Dict]:
        """read the scores contained in the file located at scores_csv"""
        scores = {}
        failures = {}

        with open(scores_csv) as fid:
            reader = csv.reader(fid)
            next(reader)

            if N is None:
                for row in reader:
                    try:
                        scores[row[0]] = float(row[1])
                    except:
                        failures[row[0]] = None
            else:
                for row in islice(reader, N):
                    try:
                        scores[row[0]] = float(row[1])
                    except:
                        failures[row[0]] = None

        return scores, failures
    
    @staticmethod
    def read_config(config_file: str) -> Dict:
        """parse an autogenerated MolPAL config file to a dictionary"""
        with open(config_file) as fid:
            return dict(line.split(' = ') for line in fid.read().splitlines())
        
    @staticmethod
    def boltzmann(xs: Iterable[float]) -> float:
        X = np.array(xs)
        E = np.exp(-X)
        Z = E.sum()
        return (X * E / Z).sum()
        