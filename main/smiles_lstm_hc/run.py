import os, pickle, torch, random, argparse
from functools import total_ordering
from pathlib import Path
import yaml
import numpy as np 
from tqdm import tqdm 
from tdc import Oracle
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
from main.utils.chem import canonicalize_list

from rnn_generator import SmilesRnnMoleculeGenerator
from rnn_utils import load_rnn_model, get_tensor_dataset, load_smiles_from_list


@total_ordering
class OptResult:
    def __init__(self, smiles: str, score: float) -> None:
        self.smiles = smiles
        self.score = score

    def __eq__(self, other):
        return (self.score, self.smiles) == (other.score, other.smiles)

    def __lt__(self, other):
        return (self.score, self.smiles) < (other.score, other.smiles)


class SMILES_LSTM_HC_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "smiles_lstm_hc"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        
        if self.smi_file is not None:
            # Exploitation run
            starting_population = self.all_smiles[:config['population_size']]
        else:
            # Exploration run
            starting_population = np.random.choice(self.all_smiles, config['population_size'])

        pretrained_model_path = os.path.join(path_here, 'pretrained_model', 'model_final_0.473.pt')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_def = Path(pretrained_model_path).with_suffix('.json')
        sample_final_model_only = False

        print('build generator')
        model = load_rnn_model(model_def, pretrained_model_path, device, copy_to_cpu=True)
        generator = SmilesRnnMoleculeGenerator(model=model,
                                                max_len=config['max_len'],
                                                device=device)
        
        int_results = generator.pretrain_on_initial_population(self.oracle, starting_population,
                                                          pretrain_epochs=config['pretrain_n_epochs'])

        results: List[OptResult] = []
        seen: Set[str] = set()

        for k in int_results:
            if k.smiles not in seen:
                results.append(k)
                seen.add(k.smiles)

        patience = 0
        
        while True:

            if len(self.oracle) > 50:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:50]]
            else:
                old_scores = 0

            print(f"Sampling new molecules ...")
            samples = generator.sampler.sample(generator.model, config['mols_to_sample'], max_seq_len=generator.max_len)

            canonicalized_samples = set(canonicalize_list(samples, include_stereocenters=True))
            payload = list(canonicalized_samples.difference(seen))
            payload.sort()  # necessary for reproducibility between different runs
            seen.update(canonicalized_samples)
            scores = self.oracle(payload)
            
            int_results = [OptResult(smiles=smiles, score=score) for smiles, score in zip(payload, scores)]

            results.extend(sorted(int_results, reverse=True)[0:config['keep_top']])
            results.sort(reverse=True)
            subset = [i.smiles for i in results][0:config['keep_top']]

            np.random.shuffle(subset)

            sub_train = subset[0:int(3 * len(subset) / 4)]
            sub_test = subset[int(3 * len(subset) / 4):]

            train_seqs, _ = load_smiles_from_list(sub_train, max_len=generator.max_len)
            valid_seqs, _ = load_smiles_from_list(sub_test, max_len=generator.max_len)

            train_set = get_tensor_dataset(train_seqs)
            valid_set = get_tensor_dataset(valid_seqs)

            opt_batch_size = min(len(sub_train), config['optimize_batch_size'])

            print_every = int(len(sub_train) / opt_batch_size)

            print(f"Tuning LSTM ...")
            if config['optimize_n_epochs'] > 0:
                generator.trainer.fit(train_set, valid_set,
                                 n_epochs=config['optimize_n_epochs'],
                                 batch_size=opt_batch_size,
                                 print_every=print_every,
                                 valid_every=print_every)
                
            # early stopping
            if len(self.oracle) > 50:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:50]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        break
                else:
                    patience = 0

            if self.finish:
                break


