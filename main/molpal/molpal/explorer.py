"""This module contains the Explorer class, which is an abstraction
for batch Bayesian optimization."""
from collections.abc import Iterable
import csv
import heapq
import json
from operator import itemgetter
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

from main.molpal.molpal import acquirer, featurizer, models, pools

T = TypeVar('T')

class Explorer:
    """An Explorer explores a pool of inputs using Bayesian optimization

    Attributes
    ----------
    pool : MoleculePool
        the pool of inputs to explore
    featurizer : Featurizer
        the featurizer this explorer will use convert molecules from SMILES
        strings into feature representations
    acquirer : Acquirer
        an acquirer which selects molecules to explore next using a prior
        distribution over the inputs
    objective : Objective
        an objective calculates the objective function of a set of inputs
    model : Model
        a model that generates a posterior distribution over the inputs using
        observed data
    retrain_from_scratch : bool
        whether the model will be retrained from scratch at each iteration.
        If False, train the model online.
        NOTE: The definition of 'online' is model-specific.
    iter : int
        the current iteration of exploration. I.e., the loop iteration the
        explorer has yet to start. This means that the current predictions will
        be the ones used in the previous iteration (because they have yet to be
        updated for the current iteration)
    scores : Dict[T, float]
        a dictionary mapping an input's identifier to its corresponding
        objective function value
    failed : Dict[T, None]
        a dictionary containing the inputs for which the objective function
        failed to evaluate
    adjustment : int
        the number of results that have been read from a file as
        opposed to being actually calculated
    new_scores : Dict[T, float]
        a dictionary mapping an input's identifier to its corresponding
        objective function value for the most recent batch of labeled inputs
    updated_model : bool
        whether the predictions are currently out-of-date with the model
    top_k_avg : float
        the average of the top-k explored inputs
    Y_pred : np.ndarray
        a list parallel to the pool containing the mean predicted score
        for an input
    Y_var : np.ndarray
        a list parallel to the pool containing the variance in the predicted
        score for an input. Will be empty if model does not provide variance
    recent_avgs : List[float]
        a list containing the recent top-k averages
    delta : float
        the minimum acceptable fractional difference between the current 
        average and the moving average in order to continue exploration
    max_iters : int
        the maximum number of batches to explore
    window_size : int
        the number of recent averages from which to calculate a moving average
    write_final : bool
        whether the list of explored inputs and their scores should be written
        to a file at the end of exploration
    write_intermediate : bool
        whether the list of explored inputs and their scores should be written
        to a file after each round of exploration
    save_preds : bool
        whether the predictions should be written after each exploration batch
    verbose : int
        the level of output the Explorer prints
    config : str
        the filepath of a configuration file containing the options necessary
        to recreate this Explorer. NOTE: this does not necessarily ensure
        reproducibility if a random seed was not set

    Parameters
    ----------
    name : str
    k : Union[int, float], default=0.01
    window_size : int, default=3
        the number of top-k averages from which to calculate a moving average
    delta : float, default=0.01
    max_iters : int, default=10
    budget : Union[int, float], default=1.
    root : str, default='.'
    write_final : bool, default=True
    write_intermediate : bool, default=False
    save_preds : bool, default=False
    retrain_from_scratch : bool, default=False
    previous_scores : Optional[str], default=None
        the filepath of a CSV file containing previous scoring data which will
        be treated as the initialization batch (instead of randomly selecting
        from the bool.)
    verbose : int, default=0
    **kwargs
        keyword arguments to initialize an Encoder, MoleculePool, Acquirer, 
        Model, and Objective classes

    Raises
    ------
    ValueError
        if k is less than 0
        if budget is less than 0
    """
    def __init__(self, oracle, path: Union[str, Path] = "molpal",
                 k: Union[int, float] = 0.01, window_size: int = 3,
                 delta: float = 0.01, max_iters: int = 10, 
                 budget: Union[int, float] = 1.,
                 write_final: bool = True, write_intermediate: bool = False,
                 chkpt_freq: int = 0, checkpoint_file: Optional[str] = None,
                 retrain_from_scratch: bool = False,
                 previous_scores: Optional[str] = None,
                 **kwargs):
        args = locals()
        
        self.path = path
        kwargs['path'] = self.path
        self.verbose = kwargs.get('verbose', 0)

        self.featurizer = featurizer.Featurizer(
            fingerprint=kwargs['fingerprint'],
            radius=kwargs['radius'], length=kwargs['length']
        )
        self.pool = pools.pool(featurizer=self.featurizer, **kwargs)
        self.acquirer = acquirer.Acquirer(size=len(self.pool), **kwargs)

        if self.acquirer.metric == 'thompson':
            kwargs['dropout_size'] = 1
        self.model = models.model(input_size=len(self.featurizer), **kwargs)
        self.acquirer.stochastic_preds = 'stochastic' in self.model.provides
        self.retrain_from_scratch = retrain_from_scratch

        self.objective = oracle

        self._validate_acquirer()

        # stopping attributes
        self.k = k
        self.delta = delta
        self.budget = budget
        self.max_iters = max_iters
        self.window_size = window_size

        # logging attributes
        self.write_final = write_final
        self.write_intermediate = write_intermediate
        self.chkpt_freq = chkpt_freq if chkpt_freq >= 0 else float('inf')
        self.previous_chkpt_iter = -float('inf')

        # stateful attributes (not including model)
        self.iter = 0
        self.scores = {}
        self.failures = {}
        self.new_scores = {}
        self.adjustment = 0
        self.updated_model = None
        self.recent_avgs = []
        self.Y_pred = np.array([])
        self.Y_var = np.array([])

        if previous_scores:
            self.load_scores(previous_scores)

        args.pop('self')
        args.update(**args.pop('kwargs'))
        args['fps'] = self.pool.fps_
        args['invalid_idxs'] = list(self.pool.invalid_idxs)
        args.pop('config', None)
        self.write_config(args)

        if checkpoint_file:
            self.load(checkpoint_file)

    def __len__(self) -> int:
        """The total expended budget expressed in terms of the number of 
        objective evaluations"""
        return self.num_explored
    
    @property
    def num_explored(self) -> int:
        """The total number of inputs this Explorer has explored adjusted
        for score data that was read as opposed evaluated"""
        return len(self.scores) + len(self.failures) - self.adjustment

    @property
    def path(self) -> Path:
        """The directory containing all automated output of this Explorer"""
        return self.__path
    
    @path.setter
    def path(self, path: Union[str, Path]):
        self.__path = Path(path)
        self.__path.mkdir(parents=True, exist_ok=True)

    @property
    def k(self) -> int:
        """the number of top-scoring inputs from which to calculate an
        average"""
        k = self.__k
            
        return min(k, len(self.pool))

    @k.setter
    def k(self, k: Union[int, float]):
        """Set k either as an integer or as a fraction of the pool.
        
        NOTE: Specifying either a fraction greater than 1 or or a number 
        larger than the pool size will default to using the full pool.
        """
        if isinstance(k, float):
            k = int(k * len(self.pool))
        if k <= 0:
            raise ValueError(f'k(={k}) must be greater than 0!')

        self.__k = k

    @property
    def budget(self) -> int:
        """the maximum budget expressed in terms of the number of allowed 
        objective function evaluations"""
        return self.__budget
    
    @budget.setter
    def budget(self, budget: Union[int, float]):
        """Set budget either as an integer or as a fraction of the pool.
        
        NOTE: Specifying either a fraction greater than 1 or or a number 
        larger than the pool size will default to using the full pool.
        """
        if isinstance(budget, float):
            budget = int(budget * len(self.pool))
        if budget <= 0.:
            raise ValueError(
                f'budget(={budget}) must be greater than 0!')

        self.__budget = budget

    @property
    def top_k_avg(self) -> Optional[float]:
        """The most recent top-k average of the explored inputs. None if k
        inputs have not yet been explored"""
        try:
            return self.recent_avgs[-1]
        except IndexError:
            return None

    @property
    def status(self) -> str:
        """The current status of the exploration in string format"""        
        if self.top_k_avg:
            ave = f'{self.top_k_avg:0.3f}' 
        else:
            if len(self.scores) > 0:
                ave = f'N/A (only {len(self.scores)} scores)'
            else:
                ave = 'N/A (no scores)'
        return (
            f'ITER: {self.iter}/{self.max_iters} | '
            f'TOP-{self.k} AVE: {ave} | '
            f'BUDGET: {len(self)}/{self.budget}'
        )

    @property
    def completed(self) -> bool:
        """whether the explorer fulfilled one of its stopping conditions

        Stopping Conditions
        -------------------
        a. explored the entire pool
           (not implemented right now due to complications with warm starting)
        b. explored for at least <max_iters> iters
        c. exceeded the maximum budget
        d. the current top-k average is within a fraction <delta> of the moving
           top-k average. This requires two sub-conditions to be met:
           1. the explorer has successfully explored at least k inputs
           2. the explorer has completed at least <window_size> iters after
              sub-condition (1) has been met

        Returns
        -------
        bool
            whether a stopping condition has been met
        """
        if self.iter > self.max_iters:
            return True
        if len(self) >= self.budget:
            return True
        if len(self.recent_avgs) < self.window_size:
            return False

        sma = sum(self.recent_avgs[-self.window_size:]) / self.window_size
        return (self.top_k_avg - sma) / sma <= self.delta

    def explore(self):
        self.run()

    def run(self):
        """Explore the MoleculePool until the stopping condition is met"""
        if self.iter == 0:
            print('Starting exploration...')
            print(f'{self.status}.', flush=True)
            self.explore_initial()
        else:
            print('Resuming exploration...')
            print(f'{self.status}.', flush=True)
            self.explore_batch()

        while not self.completed:
            # if self.verbose > 0:
            print(f'{self.status}. Continuing...', flush=True)
            self.explore_batch()

        print('Finished exploring!')
        print(f'FINAL TOP-{self.k} AVE: {self.top_k_avg:0.3f} | '
              f'FINAL BUDGET: {len(self)}/{self.budget}.')
        print(f'Final averages')
        print(f'--------------')
        for k in [0.0001, 0.0005, 0.001, 0.005]:
            print(f'TOP-{k:0.2%}: {self.avg(k):0.3f}')
        
        if self.write_final:
            self.write_scores(final=True)

    def explore_initial(self) -> float:
        """Perform an initial round of exploration
        
        Must be called before explore_batch()

        Returns
        -------
        avg : float
            the average score of the batch
        """
        inputs = self.acquirer.acquire_initial(
            xs=self.pool.smis(),
            cluster_ids=self.pool.cluster_ids(),
            cluster_sizes=self.pool.cluster_sizes,
        )

        # import ipdb; ipdb.set_trace()

        new_scores_scores = self.objective(inputs)
        new_scores = {inputs[i]: new_scores_scores[i] for i in range(len(inputs))}
        self._clean_and_update_scores(new_scores)

        # self.top_k_avg = self.avg()
        if len(self.scores) >= self.k:
            self.recent_avgs.append(self.avg())

        # if self.write_intermediate:
        #     self.write_scores(include_failed=True)

        self.iter += 1

        if (self.iter - self.previous_chkpt_iter) > self.chkpt_freq:
            self.checkpoint()
            self.previous_chkpt_iter = self.iter

        valid_scores = [y for y in new_scores.values() if y is not None]
        return sum(valid_scores) / len(valid_scores)

    def explore_batch(self) -> float:
        """Perform a round of exploration

        Returns
        -------
        avg : float
            the average score of the batch

        Raises
        ------
        InvalidExplorationError
            if called before explore_initial or load_scores
        """
        if self.iter == 0:
            raise InvalidExplorationError(
                'Cannot explore a batch before initialization!'
            )

        if self.num_explored >= len(self.pool):
            self.iter += 1
            return self.top_k_avg

        self._update_model()
        self._update_predictions()

        inputs = self.acquirer.acquire_batch(
            xs=self.pool.smis(), y_means=self.Y_pred, y_vars=self.Y_var,
            explored={**self.scores, **self.failures},
            cluster_ids=self.pool.cluster_ids(),
            cluster_sizes=self.pool.cluster_sizes, t=(self.iter-1),
        )

        new_scores_scores = self.objective(inputs)
        new_scores = {inputs[i]: new_scores_scores[i] for i in range(len(inputs))}
        self._clean_and_update_scores(new_scores)

        if len(self.scores) >= self.k:
            self.recent_avgs.append(self.avg())

        # if self.write_intermediate:
        #     self.write_scores(include_failed=True)
        
        self.iter += 1

        if (self.iter - self.previous_chkpt_iter) > self.chkpt_freq:
            self.checkpoint()
            self.previous_chkpt_iter = self.iter

        valid_scores = [y for y in new_scores.values() if y is not None]
        return sum(valid_scores)/len(valid_scores)

    def avg(self, k: Union[int, float, None] = None) -> float:
        """Calculate the average of the top k molecules
        
        Parameter
        ---------
        k : Union[int, float, None], default = None)
            the number of molecules to consider when calculating the
            average, expressed either as a specific number or as a 
            fraction of the pool. If the value specified is greater than the 
            number of successfully evaluated inputs, return the average of all 
            succesfully evaluated inputs. If None, use self.k
        
        Returns
        -------
        float
            the top-k average
        """
        k = k or self.k
        if isinstance(k, float):
            k = int(k * len(self.pool))
        k = min(k, len(self.scores))

        if k == len(self.scores):
            return sum(score for _, score in self.scores.items()) / k
        
        return sum(score for _, score in self.top_explored(k)) / k

    def top_explored(self, k: Union[int, float, None] = None) -> List[Tuple]:
        """Get the top-k explored molecules
        
        Parameter
        ---------
        k : Union[int, float, None], default=None
            the number of top-scoring molecules to get, expressed either as a 
            specific number or as a fraction of the pool. If the value 
            specified is greater than the number of successfully evaluated 
            inputs, return all explored inputs. If None, use self.k
        
        Returns
        -------
        top_explored : List[Tuple[T, float]]
            a list of tuples containing the identifier and score of the 
            top-k inputs, sorted by their score
        """
        k = k or self.k
        if isinstance(k, float):
            k = int(k * len(self.pool))
        k = min(k, len(self.scores))

        if k / len(self.scores) < 0.8:
            return heapq.nlargest(k, self.scores.items(), key=itemgetter(1))
        
        return sorted(self.scores.items(), key=itemgetter(1), reverse=True)[:k]

    def top_preds(self, k: Union[int, float, None] = None) -> List[Tuple]:
        """Get the current top predicted molecules and their scores
        
        Parameter
        ---------
        k : Union[int, float, None], default=None
            see documentation for avg()
        
        Returns
        -------
        top_preds : List[Tuple[T, float]]
            a list of tuples containing the identifier and predicted score of 
            the top-k predicted inputs, sorted by their predicted score
        """
        k = k or self.k
        if isinstance(k, float):
            k = int(k * len(self.pool))
        k = min(k, len(self.scores))

        selected = []
        for x, y in zip(self.pool.smis(), self.Y_pred):
            if len(selected) < k:
                heapq.heappush(selected, (y, x))
            else:
                heapq.heappushpop(selected, (y, x))

        return [(x, y) for y, x in selected]

    def write_scores(self, m: Union[int, float] = 1., 
                     final: bool = False,
                     include_failed: bool = False) -> None:
        """Write the top M scores to a CSV file

        Writes a CSV file of the top-k explored inputs with the input ID and
        the respective objective function value.

        Parameters
        ----------
        m : Union[int, float], default=1.
            The number of top-scoring inputs to write, expressed either as an
            integer or as a float representing the fraction of explored inputs.
            By default, writes all inputs
        final : bool, default=False
            Whether the explorer has finished. If true, write all explored
            inputs (both successful and failed) and name the output CSV file
            "all_explored_final.csv"
        include_failed : bool, default=False
            Whether to include the inputs for which objective function
            evaluation failed
        """
        if isinstance(m, float):
            m = int(m * len(self))
        m = min(m, len(self))

        p_data = self.path / 'data'
        p_data.mkdir(parents=True, exist_ok=True)

        if final:
            m = len(self)
            p_scores = p_data / f'all_explored_final.csv'
            include_failed = True
        else:
            p_scores = p_data / f'top_{m}_explored_iter_{self.iter}.csv'

        top_m = self.top_explored(m)

        with open(p_scores, 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(['smiles', 'score'])
            writer.writerows(top_m)
            if include_failed:
                writer.writerows(self.failures.items())
        
        if self.verbose > 0:
            print(f'Results were written to "{p_scores}"')

    def load_scores(self, previous_scores: str) -> None:
        """Load the scores CSV located at saved_scores.
        
        If this is being called during initialization, treat the data as the
        initialization batch.

        Parameter
        ---------
        previous_scores : str
            the filepath of a CSV file containing previous scoring information.
            The 0th column of this CSV must contain the input identifier and
            the 1st column must contain a float corresponding to its score.
            A failure to parse the 1st column as a float will treat that input
            as a failure.
        """
        if self.verbose > 0:
            print(f'Loading scores from "{previous_scores}" ... ', end='')

        scores, failures = self._read_scores(previous_scores)
        self.adjustment += len(scores) + len(failures)

        self.scores.update(scores)
        self.failures.update(failures)
        
        if self.iter == 0:
            self.iter = 1
        
        if self.verbose > 0:
            print('Done!')

    def checkpoint(self, path: Optional[str] = None) -> str:
        """write a checkpoint file for the explorer's current state and return
        the corresponding filepath
        
        Parameters
        ----------
        path : Optional[str], default=None
            the directory to under which all checkpoint files should be written
        
        Returns
        -------
        str
            the path of the JSON file containing all state information
        """
        path = path or self.path / 'chkpts' / f'iter_{self.iter}'
        chkpt_dir = Path(path)
        chkpt_dir.mkdir(parents=True, exist_ok=True)
        
        scores_pkl = chkpt_dir / 'scores.pkl'
        pickle.dump(self.scores, open(scores_pkl, 'wb'))

        failures_pkl =  chkpt_dir / 'failures.pkl'
        pickle.dump(self.failures, open(failures_pkl, 'wb'))

        new_scores_pkl =  chkpt_dir / 'new_scores.pkl'
        pickle.dump(self.new_scores, open(new_scores_pkl, 'wb'))

        preds_npz = chkpt_dir / 'preds.npz'
        np.savez(preds_npz, Y_pred=self.Y_pred, Y_var=self.Y_var)

        state = {
            'iter': self.iter,
            'scores': str(scores_pkl.absolute()),
            'failures': str(failures_pkl.absolute()),
            'new_scores': str(new_scores_pkl.absolute()),
            'adjustment': self.adjustment,
            'updated_model': self.updated_model,
            'recent_avgs': self.recent_avgs,
            'preds': str(preds_npz.absolute()),
            'model': self.model.save(chkpt_dir / 'model')
        }

        p_chkpt = chkpt_dir / 'state.json'
        json.dump(state, open(p_chkpt, 'w'), indent=4)

        if self.verbose > 1:
            print(f'Checkpoint file saved to "{p_chkpt}".')

        return str(p_chkpt)

    def load(self, chkpt_file: str):
        """Load in the state of a previous Explorer's checkpoint"""

        if self.verbose > 0:
            print(f'Loading in previous state ... ', end='')

        state = json.load(open(chkpt_file))

        self.iter = state['iter']

        self.scores = pickle.load(open(state['scores'], 'rb'))
        self.failures = pickle.load(open(state['failures'], 'rb'))
        self.new_scores = pickle.load(open(state['new_scores'], 'rb'))
        self.adjustment = state['adjustment']
        
        self.updated_model = state['updated_model']
        self.recent_avgs.extend(state['recent_avgs'])

        preds_npz = np.load(state['preds'])
        self.Y_pred = preds_npz['Y_pred']
        self.Y_var = preds_npz['Y_var']

        self.model.load(state['model'])

        if self.verbose > 0:
            print('Done!')
    
    def write_config(self, args) -> str:
        args['top-k'] = args.pop('k')
        args['no_title_line'] = not args.pop('title_line')

        for k, v in list(args.items()):
            if v is None:
                args.pop(k)
        
        config_file = self.path / 'config.ini'
        with open(config_file, 'w') as fid:
            for k, v in args.items():
                if v is None or v == False:
                    continue
                if isinstance(v, Iterable) and not isinstance(v, str):
                    v = map(str, v)
                    v = '[' + ', '.join(v) + ']'
                fid.write(f'{k.replace("_", "-")} = {v}\n')

        return str(config_file)

    def _clean_and_update_scores(self, new_scores: Dict[T, Optional[float]]):
        """Remove the None entries from new_scores and update the attributes 
        new_scores, scores, and failed accordingly

        Parameter
        ---------
        new_scores : Dict[T, Optional[float]]
            a dictionary containing the corresponding values of the objective
            function for a batch of inputs

        Side effects
        ------------
        (mutates) self.scores : Dict[T, float]
            updates self.scores with the non-None entries from new_scores
        (mutates) self.new_scores : Dict[T, float]
            updates self.new_scores with the non-None entries from new_scores
        (mutates) self.failures : Dict[T, None]
            a dictionary storing the inputs for which scoring failed
        """
        for x, y in new_scores.items():
            if y is None:
                self.failures[x] = y
            else:
                self.scores[x] = y
                self.new_scores[x] = y

    def _update_model(self):
        """Update the prior distribution to generate a posterior distribution

        Side effects
        ------------
        (mutates) self.model : Type[Model]
            updates the model with new data, if there are any
        (sets) self.new_scores : Dict[str, Optional[float]]
            reinitializes self.new_scores to an empty dictionary
        (sets) self.updated_model : bool
            sets self.updated_model to True, indicating that the predictions
            must be updated as well
        """
        if len(self.new_scores) == 0:
            self.updated_model = False
            return

        if self.retrain_from_scratch:
            xs, ys = zip(*self.scores.items())
        else:
            xs, ys = zip(*self.new_scores.items())

        self.model.train(
            xs, np.array(ys), retrain=self.retrain_from_scratch,
            featurizer=self.featurizer,
        )
        self.new_scores = {}
        self.updated_model = True

    def _update_predictions(self):
        """Update the predictions over the pool with the new model

        Side effects
        ------------
        (sets) self.Y_pred : np.ndarray
            a list of floats parallel to the pool inputs containing the mean
            predicted score for each input
        (sets) self.Y_var : np.ndarray
            a list of floats parallel to the pool inputs containing the
            predicted variance for each input
        (sets) self.updated_model : bool
            sets self.updated_model to False, indicating that the predictions 
            are now up-to-date with the current model
        """
        if not self.updated_model and self.Y_pred.size > 0:
            if self.verbose > 1:
                print('Model has not been updated since the last time ',
                     'predictions were set. Skipping update!')
            return

        self.Y_pred, self.Y_var = self.model.apply(
            x_ids=self.pool.smis(), x_feats=self.pool.fps(), 
            batched_size=None, size=len(self.pool), 
            mean_only='vars' not in self.acquirer.needs
        )

        self.updated_model = False
        
    def _validate_acquirer(self):
        """Ensure that the model provides values the Acquirer needs"""
        if self.acquirer.needs > self.model.provides:
            raise IncompatibilityError(
                f'{self.acquirer.metric} metric needs: '
                + f'{self.acquirer.needs} '
                + f'but {self.model.type_} only provides: '
                + f'{self.model.provides}')

    def _read_scores(self, scores_csv: str) -> Tuple[Dict, Dict]:
        """read the scores contained in the file located at scores_csv"""
        scores = {}
        failures = {}
        with open(scores_csv) as fid:
            reader = csv.reader(fid)
            next(reader)
            for row in reader:
                try:
                    scores[row[0]] = float(row[1])
                except:
                    failures[row[0]] = None
        
        return scores, failures

class InvalidExplorationError(Exception):
    pass

class IncompatibilityError(Exception):
    pass
