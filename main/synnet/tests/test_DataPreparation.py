"""
Unit tests for the data preparation.
"""
import unittest
import os
from shutil import copyfile
import pandas as pd
from syn_net.utils.predict_utils import get_mol_embedding
from tqdm import tqdm
from scipy import sparse
import numpy as np
from syn_net.utils.prep_utils import organize, synthetic_tree_generator, prep_data
from syn_net.utils.data_utils import SyntheticTreeSet, Reaction, ReactionSet
from dgllife.model import load_pretrained

class TestDataPrep(unittest.TestCase):
    """
    Tests for the data preparation: (1) reaction data processing, (2) synthetic
    tree prep, (3) featurization, (4) training data preparation for each network.
    """
    def test_process_rxn_templates(self):
        """
        Tests the rxn templates processing.
        """
        # the following file contains the three templates at the top of
        # 'SynNet/data/rxn_set_hb.txt'
        path_to_rxn_templates = './data/rxn_set_hb_test.txt'

        # load the reference building blocks (100 here)
        path_to_building_blocks = './data/building_blocks_matched.csv.gz'
        building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()

        # load the reaction templates
        rxn_templates = []
        with open(path_to_rxn_templates, 'rt') as rxn_template_file:
            for line in rxn_template_file:
                rxn = Reaction(line.split('|')[1].strip())
                rxn.set_available_reactants(building_block_list=building_blocks)
                rxn_templates.append(rxn)

        # save the templates as a ReactionSet
        r = ReactionSet(rxn_templates)
        r.save('./data/rxns_hb.json.gz')

        # load the reference reaction templates
        path_to_ref_rxn_templates = './data/ref/rxns_hb.json.gz'
        r_ref = ReactionSet()
        r_ref.load(path_to_ref_rxn_templates)

        # check here that the templates were correctly saved as a ReactionSet by
        # comparing to a provided reference file in 'SynNet/tests/data/ref/'
        for rxn_idx, rxn in enumerate(r.rxns):
            rxn = rxn.__dict__
            ref_rxn = r_ref.rxns[rxn_idx].__dict__
            self.assertTrue(rxn == ref_rxn)

    def test_synthetic_tree_prep(self):
        """
        Tests the synthetic tree preparation.
        """
        np.random.seed(6)

        # load the `Reactions` (built from 3 reaction templates)
        path_to_rxns = './data/ref/rxns_hb.json.gz'
        r_ref = ReactionSet()
        r_ref.load(path_to_rxns)
        rxns = r_ref.rxns

        # load the reference building blocks (100 here)
        path_to_building_blocks = './data/building_blocks_matched.csv.gz'
        building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()

        num_trials   = 25
        num_finish   = 0
        num_error    = 0
        num_unfinish = 0

        trees = []
        for _ in tqdm(range(num_trials)):
            tree, action = synthetic_tree_generator(building_blocks,
                                                    rxns,
                                                    max_step=5)
            if action == 3:
                trees.append(tree)
                num_finish += 1
            elif action == -1:
                num_error += 1
            else:
                num_unfinish += 1

        synthetic_tree_set = SyntheticTreeSet(sts=trees)
        synthetic_tree_set.save('./data/st_data.json.gz')

        # check that the number of finished trees generated is == 3, and that
        # the number of unfinished trees generated is == 0
        self.assertEqual(num_finish, 3)
        self.assertEqual(num_unfinish, 0)

        # check here that the synthetic trees were correctly saved by
        # comparing to a provided reference file in 'SynNet/tests/data/ref/'
        sts_ref = SyntheticTreeSet()
        sts_ref.load('./data/ref/st_data.json.gz')
        for st_idx, st in enumerate(sts_ref.sts):
            st = st.__dict__
            ref_st = sts_ref.sts[st_idx].__dict__
            self.assertTrue(st == ref_st)

    def test_featurization(self):
        """
        Tests the featurization of the synthetic tree data into step-by-step
        data for training.
        """
        embedding    = 'fp'
        radius       = 2
        nbits        = 4096
        dataset_type = 'train'

        path_st            = './data/ref/st_data.json.gz'
        save_dir           = './data/'
        reference_data_dir = './data/ref/'

        st_set = SyntheticTreeSet()
        st_set.load(path_st)
        data = st_set.sts
        del st_set

        states = []
        steps  = []

        save_idx = 0
        for st in tqdm(data):
            try:
                state, step = organize(st,
                                       target_embedding=embedding,
                                       radius=radius,
                                       nBits=nbits)
            except Exception as e:
                print(e)
                continue
            states.append(state)
            steps.append(step)

        del data

        if len(steps) != 0:
            # save the states and steps
            states = sparse.vstack(states)
            steps  = sparse.vstack(steps)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            sparse.save_npz(f'{save_dir}states_{save_idx}_{dataset_type}.npz', states)
            sparse.save_npz(f'{save_dir}steps_{save_idx}_{dataset_type}.npz', steps)

        # load the reference data, which we will compare against
        states_ref = sparse.load_npz(f'{reference_data_dir}states_{save_idx}_{dataset_type}.npz')
        steps_ref = sparse.load_npz(f'{reference_data_dir}steps_{save_idx}_{dataset_type}.npz')

        # check here that states and steps were correctly saved (need to convert the
        # sparse arrays to non-sparse arrays for comparison)
        self.assertEqual(states.toarray().all(), states_ref.toarray().all())
        self.assertEqual(steps.toarray().all(), steps_ref.toarray().all())

    def test_dataprep(self):
        """
        Tests the training "data preparation" using the test subset data. What
        data preparation refers to here is the preparation of training, testing,
        and validation data by reading in the states and steps for the
        previously written synthetic trees, and re-writing them as separate
        one-hot encoded Action, Reactant 1, Reactant 2, and Reaction network
        files. In other words, the preparation of data for each specific network.
        """
        main_dir = './data/'
        ref_dir = './data/ref/'
        # copy data from the reference directory to use for this particular test
        copyfile(f'{ref_dir}states_0_train.npz', f'{main_dir}states_0_train.npz')
        copyfile(f'{ref_dir}steps_0_train.npz', f'{main_dir}steps_0_train.npz')

        # the lines below will save Action-, Reactant 1-, Reaction-, and Reactant 2-
        # specific files directly to the 'SynNet/tests/data/' directory (e.g.
        # 'X_act_{train/test/valid}.npz' and 'y_act_{train/test/valid}.npz'
        # 'X_rt1_{train/test/valid}.npz' and 'y_rt1_{train/test/valid}.npz'
        # 'X_rxn_{train/test/valid}.npz' and 'y_rxn_{train/test/valid}.npz'
        # 'X_rt2_{train/test/valid}.npz' and 'y_rt2_{train/test/valid}.npz'
        prep_data(main_dir=main_dir, num_rxn=3, out_dim=300)

        # check that the saved files match the reference files in
        # 'SynNet/tests/data/ref':

        # Action network data
        X_act = sparse.load_npz(f'{main_dir}X_act_train.npz')
        y_act = sparse.load_npz(f'{main_dir}y_act_train.npz')

        X_act_ref = sparse.load_npz(f'{ref_dir}X_act_train.npz')
        y_act_ref = sparse.load_npz(f'{ref_dir}y_act_train.npz')

        self.assertEqual(X_act.toarray().all(), X_act_ref.toarray().all())
        self.assertEqual(y_act.toarray().all(), y_act_ref.toarray().all())

        # Reactant 1 network data
        X_rt1 = sparse.load_npz(f'{main_dir}X_rt1_train.npz')
        y_rt1 = sparse.load_npz(f'{main_dir}y_rt1_train.npz')

        X_rt1_ref = sparse.load_npz(f'{ref_dir}X_rt1_train.npz')
        y_rt1_ref = sparse.load_npz(f'{ref_dir}y_rt1_train.npz')

        self.assertEqual(X_rt1.toarray().all(), X_rt1_ref.toarray().all())
        self.assertEqual(y_rt1.toarray().all(), y_rt1_ref.toarray().all())

        # Reaction network data
        X_rxn = sparse.load_npz(f'{main_dir}X_rxn_train.npz')
        y_rxn = sparse.load_npz(f'{main_dir}y_rxn_train.npz')

        X_rxn_ref = sparse.load_npz(f'{ref_dir}X_rxn_train.npz')
        y_rxn_ref = sparse.load_npz(f'{ref_dir}y_rxn_train.npz')

        self.assertEqual(X_rxn.toarray().all(), X_rxn_ref.toarray().all())
        self.assertEqual(y_rxn.toarray().all(), y_rxn_ref.toarray().all())

        # Reactant 2 network data
        X_rt2 = sparse.load_npz(f'{main_dir}X_rt2_train.npz')
        y_rt2 = sparse.load_npz(f'{main_dir}y_rt2_train.npz')

        X_rt2_ref = sparse.load_npz(f'{ref_dir}X_rt2_train.npz')
        y_rt2_ref = sparse.load_npz(f'{ref_dir}y_rt2_train.npz')

        self.assertEqual(X_rt2.toarray().all(), X_rt2_ref.toarray().all())
        self.assertEqual(y_rt2.toarray().all(), y_rt2_ref.toarray().all())

    def test_bb_emb(self):
        """
        Tests the building block embedding function.
        """
        # define some constants
        main_dir = './data/'
        ref_dir = './data/ref/'

        # define model to use for molecular embedding
        model_type = 'gin_supervised_contextpred'
        device = 'cpu'
        model = load_pretrained(model_type).to(device) # used to learn embedding
        model.eval()

        # load the building blocks
        path_to_building_blocks = './data/building_blocks_matched.csv.gz'
        building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()

        # compute the building block embeddings
        embeddings = []
        for smi in tqdm(building_blocks):
            embeddings.append(get_mol_embedding(smi, model=model))

        embeddings = np.array(embeddings)
        np.save(f'{main_dir}building_blocks_emb.npy', embeddings)

        # load the reference embeddings
        embeddings_ref = np.load(f'{ref_dir}building_blocks_emb.npy')

        self.assertEqual(embeddings.all(), embeddings_ref.all())
