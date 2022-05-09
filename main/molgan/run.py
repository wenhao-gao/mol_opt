import os
import numpy as np 
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer

import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.gan import GraphGANOptimizer

from rdkit import Chem


class MolGAN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "molgan"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        

        data = SparseMolecularDataset()
        data.load(os.path.join(path_here, 'data/gdb9_9nodes.sparsedataset'))
        steps = (len(data) // config['batch_dim'])
        n_samples = config['n_samples']
        la = 1


        def reward(mols):
            smiles_list = [Chem.MolToSmiles(mol) if mol is not None else 'CCC' for mol in mols] 
            scores = self.oracle(smiles_list)  
            scores = np.array(scores).reshape(-1,1) 
            return scores 

        def train_fetch_dict(i, steps, epoch, epochs, min_epochs, model, optimizer):
            a = [optimizer.train_step_G] if i % config['n_critic'] == 0 else [optimizer.train_step_D]
            b = [optimizer.train_step_V] if i % config['n_critic'] == 0 and la < 1 else []
            return a + b


        def train_feed_dict(i, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
            mols, _, _, a, x, _, _, _, _ = data.next_train_batch(batch_dim)
            embeddings = model.sample_z(batch_dim)

            if la < 1:

                if i % config['n_critic'] == 0:
                    rewardR = reward(mols)

                    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                                       feed_dict={model.training: False, model.embeddings: embeddings})
                    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
                    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

                    rewardF = reward(mols)

                    feed_dict = {model.edges_labels: a,
                                 model.nodes_labels: x,
                                 model.embeddings: embeddings,
                                 model.rewardR: rewardR,
                                 model.rewardF: rewardF,
                                 model.training: True,
                                 model.dropout_rate: config['dropout'],
                                 optimizer.la: la if epoch > 0 else 1.0}

                else:
                    feed_dict = {model.edges_labels: a,
                                 model.nodes_labels: x,
                                 model.embeddings: embeddings,
                                 model.training: True,
                                 model.dropout_rate: config['dropout'],
                                 optimizer.la: la if epoch > 0 else 1.0}
            else:
                feed_dict = {model.edges_labels: a,
                             model.nodes_labels: x,
                             model.embeddings: embeddings,
                             model.training: True,
                             model.dropout_rate: config['dropout'],
                             optimizer.la: 1.0}

            return feed_dict


        def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
            return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
                    'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
                    'la': optimizer.la}


        def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
            mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
            embeddings = model.sample_z(a.shape[0])

            rewardR = reward(mols)

            n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                               feed_dict={model.training: False, model.embeddings: embeddings})
            n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
            mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

            rewardF = reward(mols)

            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.rewardR: rewardR,
                         model.rewardF: rewardF,
                         model.training: False}
            return feed_dict


        def test_fetch_dict(model, optimizer):
            return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
                    'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
                    'la': optimizer.la}


        def test_feed_dict(model, optimizer, batch_dim):
            mols, _, _, a, x, _, _, _, _ = data.next_test_batch()
            embeddings = model.sample_z(a.shape[0])

            rewardR = reward(mols)

            n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                               feed_dict={model.training: False, model.embeddings: embeddings})
            n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
            mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

            rewardF = reward(mols)

            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.rewardR: rewardR,
                         model.rewardF: rewardF,
                         model.training: False}
            return feed_dict


        def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
            mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
            m0, m1 = all_scores(mols, data, norm=True)
            m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
            m0.update(m1)
            return m0


        def _test_update(model, optimizer, batch_dim, test_batch):
            mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
            m0, m1 = all_scores(mols, data, norm=True)
            m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
            m0.update(m1)
            return m0

        model = GraphGANModel(data.vertexes, data.bond_num_types, data.atom_num_types, config['z_dim'],
                              decoder_units=(128, 256, 512), discriminator_units=((128, 64), 128, (128, 64)),
                              decoder=decoder_adj, discriminator=encoder_rgcn,
                              soft_gumbel_softmax=False, hard_gumbel_softmax=False, batch_discriminator=False)
        optimizer = GraphGANOptimizer(model, learning_rate=1e-3, feature_matching=False)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        trainer = Trainer(model, optimizer, session)
        print('Parameters: {}'.format(np.sum([np.prod(e.shape) for e in session.run(tf.trainable_variables())])))

        trainer.train(batch_dim=config['batch_dim'],
                      epochs=config['epochs'],
                      steps=steps,
                      train_fetch_dict=train_fetch_dict,
                      train_feed_dict=train_feed_dict,
                      eval_fetch_dict=eval_fetch_dict,
                      eval_feed_dict=eval_feed_dict,
                      test_fetch_dict=test_fetch_dict,
                      test_feed_dict=test_feed_dict,
                      save_every=config['save_every'],
                      directory='save', # here users need to first create and then specify a folder where to save the model
                      _eval_update=_eval_update,
                      _test_update=_test_update)
