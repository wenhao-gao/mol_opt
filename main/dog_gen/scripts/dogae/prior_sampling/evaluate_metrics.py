"""Evaluate Metrics


Usage:
  evaluate_metrics.py  [--config=<config>]

Options:
  --config=<config>  The path to the config file [default: tables_spec.json].
"""

import json
from os import path
import time
import typing
import random
import sys
import warnings
import multiprocessing as mp
from multiprocessing import Pool

import multiset
import tabulate
import numpy as np
import tqdm
from lazy import lazy
import pandas as pd
from docopt import docopt

from guacamol import frechet_benchmark
import fcd
from rd_filters import rd_filters

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from syn_dags.chem_ops import rdkit_general_ops
from syn_dags.script_utils import train_utils

THIS_FILE_DIR = path.dirname(__file__)


def filter_valid_and_map_to_ms(smiles_in):
    ms_out = []
    for sm in tqdm.tqdm(smiles_in, desc="ValidFilter"):
        try:
            ms = _put_line_into_canonical_multiset(sm)
            ms_out.append(ms)
        except CanonicaliseError:
            pass

    return ms_out


class UniquenessCheck:
    """
    Unique if the generated molecule multiset contains at least one molecule which has not been generated so far.
    """
    def __call__(self, valid_molecule_bags: typing.List[multiset.BaseMultiset]):
        seen_products_set = set()
        unique_product_bags = []

        for product_bag in tqdm.tqdm(valid_molecule_bags, desc="uniqueness check"):
            unique_elements = [elem for elem in product_bag if elem not in seen_products_set]
            seen_products_set.update(set(product_bag.distinct_elements()))
            if len(unique_elements):
                unique_product_bags.append(product_bag)
        return len(unique_product_bags) / len(valid_molecule_bags)


class NoveltyCheck:
    """
    Novel if at least one of the molecules in the generated molecule multiset does not appear in the training dataset.
    """
    def __init__(self, training_canonical_smiles: typing.List[str]):
        self.training_smiles = set(training_canonical_smiles)

    def __call__(self, valid_molecule_bags: typing.List[multiset.BaseMultiset]):
        novel_molecules = []
        for gen_ms in tqdm.tqdm(valid_molecule_bags, desc="novelty check"):
            is_novel = any([elem not in self.training_smiles for elem in gen_ms])
            if is_novel:
                novel_molecules.append(gen_ms)
        return len(novel_molecules) / len(valid_molecule_bags)


class FCDCheck:
    """
    Computes the Fréchet ChemNet Distance between molecules in the training set and those generated.

    Note 1. this is estimated from samples. For models that do not produce many valid molecules the variance is likely to
    be quite high.
    Note 2. We do not follow GuacaMol by returning "FCD Score" as exp(-0.2*FCD)  (see eqn 2 of their paper). We return
    the FCD as it is.
    """
    def __init__(self, training_smi: typing.List[str], sample_size=10000):
        self.fcd_scorer = frechet_benchmark.FrechetBenchmark(training_smi, sample_size=sample_size)

    @lazy
    def chemnet(self):
        return self.fcd_scorer._load_chemnet()

    @lazy
    def cached_ref_stats(self):
        mu, cov = self.fcd_scorer._calculate_distribution_statistics(self.chemnet, self.fcd_scorer.reference_molecules)
        return mu, cov

    def __call__(self, valid_molecule_bags: typing.List[multiset.BaseMultiset]):
        if not len(valid_molecule_bags) >= self.fcd_scorer.sample_size:
            print(f"less samples than ideal... @{len(valid_molecule_bags)}")
            sample_size = len(valid_molecule_bags)
        else:
            sample_size = self.fcd_scorer.sample_size

        # Sample the generated molecule multisets
        samples_bags = random.sample(valid_molecule_bags, sample_size)

        # And then for each pick one molecule randomly from the multiset:
        samples = []
        for s in samples_bags:
            samples.append(random.choice(list(s.distinct_elements())))

        chemnet = self.chemnet

        print("FCD: calculating dist stats on training data...")
        mu_ref, cov_ref = self.cached_ref_stats

        print("FCD: calculating dist stats on new generated molecules...")
        mu, cov = self.fcd_scorer._calculate_distribution_statistics(chemnet, samples)

        print("FCD: ... computed stats!")

        FCD = fcd.calculate_frechet_distance(mu1=mu_ref, mu2=mu,
                                             sigma1=cov_ref, sigma2=cov)
        # ^ See note 2 in class docstring.

        return FCD


class QualityFiltersCheck:
    """
    These are the Quality Filters proposed in the GuacaMol paper, which try to rule out " compounds which are
     potentially unstable, reactive, laborious to synthesize, or simply unpleasant to the eye of medicinal chemists."

    The filter rules are from the GuacaMol supplementary material: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839
    The filter code is from: https://github.com/PatWalters/rd_filters
    Parts of the code below have been taken from the script in this module. This code put in this
     class came with this MIT Licence:

        MIT License

        Copyright (c) 2018 Patrick Walters

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

    """
    def __init__(self, training_data_smi: typing.List[str]):

        alert_file_name = path.join(THIS_FILE_DIR, 'quality_filters', 'alert_collection.csv')
        self.rf = rd_filters.RDFilters(alert_file_name)

        rules_file_path = path.join(THIS_FILE_DIR, 'quality_filters', 'rules.json')
        rule_dict = rd_filters.read_rules(rules_file_path)
        rule_list = [x.replace("Rule_", "") for x in rule_dict.keys() if x.startswith("Rule") and rule_dict[x]]
        rule_str = " and ".join(rule_list)
        print(f"Using alerts from {rule_str}", file=sys.stderr)
        self.rf.build_rule_list(rule_list)
        self.rule_dict = rule_dict

        self.training_data_smi = training_data_smi

    @lazy
    def _training_data_prop(self):
        training_data_qulaity_filters = self.call_on_smiles_no_normalization(self.training_data_smi)
        print(f"Training data filters returned {training_data_qulaity_filters}. Rest normalized on this.")
        return training_data_qulaity_filters

    def __call__(self, valid_molecule_bags: typing.List[multiset.BaseMultiset]):
        smiles = ['.'.join(elem) for elem in valid_molecule_bags]
        return self.call_on_smiles_no_normalization(smiles) / self._training_data_prop

    def call_on_smiles_no_normalization(self, smiles: typing.List[str]):
        num_cores = 4
        print(f"using {num_cores} cores", file=sys.stderr)
        start_time = time.time()
        p = Pool(mp.cpu_count())

        num_smiles_in = len(smiles)
        input_data = [(smi, f"MOL_{i}") for i, smi in enumerate(smiles)]

        res = list(p.map(self.rf.evaluate, input_data))
        df = pd.DataFrame(res, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA"])
        df_ok = df[
            (df.FILTER == "OK") &
            df.MW.between(*self.rule_dict["MW"]) &
            df.LogP.between(*self.rule_dict["LogP"]) &
            df.HBD.between(*self.rule_dict["HBD"]) &
            df.HBA.between(*self.rule_dict["HBA"]) &
            df.TPSA.between(*self.rule_dict["TPSA"])
            ]

        num_input_rows = df.shape[0]
        num_output_rows = df_ok.shape[0]
        fraction_passed = "{:.1f}".format(num_output_rows / num_input_rows * 100.0)
        print(f"{num_output_rows} of {num_input_rows} passed filters {fraction_passed}%", file=sys.stderr)
        elapsed_time = "{:.2f}".format(time.time() - start_time)
        print(f"Elapsed time {elapsed_time} seconds", file=sys.stderr)
        p.close()
        return (num_output_rows / num_smiles_in)


class CanonicaliseError(Exception):
    pass


def _put_line_into_canonical_multiset(reaction_smi_str: str) -> multiset.FrozenMultiset:
    """
    Splits up string into multiple molecules. Canonicalises each of these individually.
    
    Throws a CanonicaliseError exception if it cannot canonicalise at least one molecule in the string.
    """
    # If string is empty then cannot canonicalise so throw exception.
    if len(reaction_smi_str) == 0:
        raise CanonicaliseError
    
    # Now go through and canonicalise each individual molecules
    all_smi = reaction_smi_str.split('.')
    canoncial_smi = []
    for mol_smi in all_smi:
        try:
            canoncial_smi.append(rdkit_general_ops.canconicalize(mol_smi))
        except:
            # ignore any exceptionf for now.
            pass
        
    # We either (a) got at least one molecule 
    if len(canoncial_smi):        
        assert len(canoncial_smi[0]), f"Empty string passed through canonical function{canoncial_smi}"
        return multiset.FrozenMultiset(canoncial_smi)
    
    # Or (b) if not we had no valid molecules and so can throw an Exception
    else:
        raise CanonicaliseError


def _read_in_text_of_smiles(filename):
    with open(filename, 'r') as fo:
        smiles = fo.readlines()
        smiles = [elem.strip() for elem in smiles]
    return smiles


class Params:
    def __init__(self):
        # Config to read:
        arguments = docopt(__doc__)
        self.experiments_config = arguments['--config']

        # Reactants file
        self.training_trees = train_utils.load_tuple_trees('../../dataset_creation/data/'
                                           'uspto-train-depth_and_tree_tuples.pick', np.random.RandomState(10))

        # Get training data smiles strings
        self.training_data_smi_list = self._get_training_data()

    def _get_training_data(self):
        all_train_molecules = set()

        def unpack(iterable):
            if isinstance(iterable, str):
                all_train_molecules.add(iterable)
            elif isinstance(iterable, (tuple, list)):
                for item in iterable:
                    unpack(item)
            else:
                raise RuntimeError

        unpack(self.training_trees)
        all_mols = [rdkit_general_ops.canconicalize(smi) for smi in tqdm.tqdm(all_train_molecules,
                                                                              desc='ensuring molecules canonical')]
        all_mols = sorted(list(set(all_mols)))
        return all_mols


def main(params: Params):
    print("setting up metrics")
    rng = np.random.RandomState(484)
    random.seed(rng.choice(54156))

    metrics_conditioned_on_valid = {
      # All the metrics below are conditioned on validity:
        "novelty": NoveltyCheck(params.training_data_smi_list),
        "uniqueness": UniquenessCheck(),
        "FCD": FCDCheck(params.training_data_smi_list),
        "quality_filters": QualityFiltersCheck(params.training_data_smi_list),
    }

    def _create_metrics_part_of_row(smiles: typing.List[str], metrics, num_generated):
        row = []
        if len(smiles) != num_generated:
            warnings.warn(f"Number generated @{num_generated} different to num passed in @{len(smiles)}")

        # First we filter out to form valid multisets of molecules:
        valid_ms = filter_valid_and_map_to_ms(smiles_in=smiles)

        for met in metrics:
            if met == 'validity':
                row.append(len(valid_ms) / num_generated)
            elif met == 'num_generated':
                row.append(len(smiles))
            else:
                row.append(metrics_conditioned_on_valid[met](valid_ms))
        return row

    print("Reading config")
    with open(params.experiments_config, 'r') as fo:
        running_config = json.load(fo)
        print(running_config)
        print('\n\n\n')

    data_dir = running_config['data_dir']
    table_format = running_config['table_format']

    for i, tables in enumerate(running_config['tables_to_create']):
        print(f"\n\n\n ==== Creating table {i} ====")

        rows_all = []
        header = ['Method'] + tables['metrics']

        rows_all.append(["training data"] + _create_metrics_part_of_row(params.training_data_smi_list, tables['metrics'], len(params.training_data_smi_list)))

        for dataset_name, (location, num_generated) in tables['rows'].items():
            row = [dataset_name]

            print(f"Reading in data for {dataset_name}")
            smiles = _read_in_text_of_smiles(path.join(data_dir, location))
            row += _create_metrics_part_of_row(smiles, tables['metrics'], num_generated)
            rows_all.append(row)
        print('\n\n\n')
        print(tabulate.tabulate(rows_all, headers=header, tablefmt=table_format))


if __name__ == '__main__':
    print("Starting!")
    main(Params())
    print("Done!")
