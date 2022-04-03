"""
Unit tests for the model predictions.
"""
import unittest
import numpy as np
import pandas as pd
from syn_net.utils.predict_utils import synthetic_tree_decoder_multireactant, mol_fp, load_modules_from_checkpoint
from syn_net.utils.data_utils import SyntheticTreeSet, ReactionSet


class TestPredict(unittest.TestCase):
    """
    Tests for model predictions using greedy search.
    """
    def test_predict(self):
        """
        Tests synthetic tree generation given a molecular embedding. No beam search.
        """
        np.random.seed(seed=137)

        # define some constants (here, for the unit test rxn set)
        ref_dir      = f'./data/ref/'
        nbits        = 4096
        out_dim      = 300
        featurize    = 'fp'
        rxn_template = 'unittest'
        ncpu         = 2

        ### load the purchasable building block embeddings
        bb_emb = np.load(f'{ref_dir}building_blocks_emb.npy')

        # define path to the reaction templates and purchasable building blocks
        path_to_reaction_file   = f'{ref_dir}rxns_hb.json.gz'
        path_to_building_blocks = './data/building_blocks_matched.csv.gz'

        # define paths to pretrained modules
        path_to_act = f'{ref_dir}act.ckpt'
        path_to_rt1 = f'{ref_dir}rt1.ckpt'
        path_to_rxn = f'{ref_dir}rxn.ckpt'
        path_to_rt2 = f'{ref_dir}rt2.ckpt'

        # load the purchasable building block SMILES to a dictionary
        building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
        bb_dict         = {building_blocks[i]: i for i in range(len(building_blocks))}

        # load the reaction templates as a ReactionSet object
        rxn_set = ReactionSet()
        rxn_set.load(path_to_reaction_file)
        rxns    = rxn_set.rxns

        # load the pre-trained modules
        act_net, rt1_net, rxn_net, rt2_net = load_modules_from_checkpoint(
            path_to_act=path_to_act,
            path_to_rt1=path_to_rt1,
            path_to_rxn=path_to_rxn,
            path_to_rt2=path_to_rt2,
            featurize=featurize,
            rxn_template=rxn_template,
            out_dim=out_dim,
            nbits=nbits,
            ncpu=ncpu,
        )

        # load the query molecules (i.e. molecules to decode)
        path_to_data = f'{ref_dir}st_data.json.gz'
        sts = SyntheticTreeSet()
        sts.load(path_to_data)
        smis_query = [st.root.smiles for st in sts.sts]

        # start to decode the query molecules (no multiprocessing for the unit tests here)
        smis_decoded = []
        similarities = []
        trees        = []
        for smi in smis_query:
            emb = mol_fp(smi)
            smi, similarity, tree, action = synthetic_tree_decoder_multireactant(
                z_target=emb,
                building_blocks=building_blocks,
                bb_dict=bb_dict,
                reaction_templates=rxns,
                mol_embedder=mol_fp,
                action_net=act_net,
                reactant1_net=rt1_net,
                rxn_net=rxn_net,
                reactant2_net=rt2_net,
                bb_emb=bb_emb,
                rxn_template=rxn_template,
                n_bits=nbits,
                beam_width=1,
                max_step=15)

            smis_decoded.append(smi)
            similarities.append(similarity)
            trees.append(tree)

        # check the results and compare to the reference values
        recovery_rate      = np.sum(np.array(similarities) == 1.0) / len(similarities)
        average_similarity = np.mean(np.array(similarities))

        recovery_rate_ref      = 0.0
        average_similarity_ref = 0.395045045045045

        self.assertEqual(recovery_rate, recovery_rate_ref)
        self.assertEqual(average_similarity, average_similarity_ref)
