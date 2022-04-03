"""
Unit tests for model training.
"""
import unittest
import shutil
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from syn_net.models.mlp import MLP, load_array
from scipy import sparse


class TestTraining(unittest.TestCase):
    """
    Tests for model training: (1) action network, (2) reactant 1 network, (3)
    reaction network, (4) reactant 2 network.
    """
    def test_action_network(self):
        """
        Tests the Action Network.
        """
        embedding         = 'fp'
        radius            = 2
        nbits             = 4096
        batch_size        = 10
        epochs            = 2
        ncpu              = 2
        validation_option = 'accuracy'
        ref_dir           = f'./data/ref/'

        X = sparse.load_npz(ref_dir + 'X_act_train.npz')
        y = sparse.load_npz(ref_dir + 'y_act_train.npz')
        X = torch.Tensor(X.A)
        y = torch.LongTensor(y.A.reshape(-1, ))
        train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

        # use the train data for validation too (just for the unit tests)
        valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

        pl.seed_everything(0)
        mlp = MLP(input_dim=int(3 * nbits),
                  output_dim=4,
                  hidden_dim=100,
                  num_layers=3,
                  dropout=0.5,
                  num_dropout_layers=1,
                  task='classification',
                  loss='cross_entropy',
                  valid_loss=validation_option,
                  optimizer='adam',
                  learning_rate=1e-4,
                  val_freq=10,
                  ncpu=ncpu)

        tb_logger = pl_loggers.TensorBoardLogger(f'act_{embedding}_{radius}_{nbits}_logs/')
        trainer   = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=20, logger=tb_logger)
        trainer.fit(mlp, train_data_iter, valid_data_iter)

        train_loss     = float(trainer.callback_metrics["train_loss"])
        train_loss_ref = 1.4203987121582031

        shutil.rmtree(f'act_{embedding}_{radius}_{nbits}_logs/')
        self.assertEqual(train_loss, train_loss_ref)

    def test_reactant1_network(self):
        """
        Tests the Reactant 1 Network.
        """
        embedding         = 'fp'
        radius            = 2
        nbits             = 4096
        out_dim           = 300
        batch_size        = 10
        epochs            = 2
        ncpu              = 2
        validation_option = 'nn_accuracy_gin'
        ref_dir           = f'./data/ref/'

        X = sparse.load_npz(ref_dir + 'X_rt1_train.npz')
        y = sparse.load_npz(ref_dir + 'y_rt1_train.npz')
        X = torch.Tensor(X.A)
        y = torch.Tensor(y.A)
        train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

        # use the train data for validation too (just for the unit tests)
        valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

        pl.seed_everything(0)
        mlp = MLP(input_dim=int(3 * nbits),
                  output_dim=out_dim,
                  hidden_dim=100,
                  num_layers=3,
                  dropout=0.5,
                  num_dropout_layers=1,
                  task='regression',
                  loss='mse',
                  valid_loss=validation_option,
                  optimizer='adam',
                  learning_rate=1e-4,
                  val_freq=10,
                  ncpu=ncpu)

        tb_logger = pl_loggers.TensorBoardLogger(f'rt1_{embedding}_{radius}_{nbits}_logs/')
        trainer   = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=20, logger=tb_logger)
        trainer.fit(mlp, train_data_iter, valid_data_iter)

        train_loss     = float(trainer.callback_metrics["train_loss"])
        train_loss_ref = 0.3557135760784149

        shutil.rmtree(f'rt1_{embedding}_{radius}_{nbits}_logs/')
        self.assertEqual(train_loss, train_loss_ref)

    def test_reaction_network(self):
        """
        Tests the Reaction Network.
        """
        embedding         = 'fp'
        radius            = 2
        nbits             = 4096
        batch_size        = 10
        epochs            = 2
        ncpu              = 2
        n_templates       = 3  # num templates in 'data/rxn_set_hb_test.txt'
        validation_option = 'accuracy'
        ref_dir           = f'./data/ref/'

        X = sparse.load_npz(ref_dir + 'X_rxn_train.npz')
        y = sparse.load_npz(ref_dir + 'y_rxn_train.npz')
        X = torch.Tensor(X.A)
        y = torch.LongTensor(y.A.reshape(-1, ))
        train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

        # use the train data for validation too (just for the unit tests)
        valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

        pl.seed_everything(0)
        mlp = MLP(input_dim=int(4 * nbits),
                  output_dim=n_templates,
                  hidden_dim=100,
                  num_layers=5,
                  dropout=0.5,
                  num_dropout_layers=1,
                  task='classification',
                  loss='cross_entropy',
                  valid_loss=validation_option,
                  optimizer='adam',
                  learning_rate=1e-4,
                  val_freq=10,
                  ncpu=ncpu)

        tb_logger = pl_loggers.TensorBoardLogger(f'rxn_{embedding}_{radius}_{nbits}_logs/')
        trainer   = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=20, logger=tb_logger)
        trainer.fit(mlp, train_data_iter, valid_data_iter)

        train_loss     = float(trainer.callback_metrics["train_loss"])
        train_loss_ref = 1.121474027633667

        shutil.rmtree(f'rxn_{embedding}_{radius}_{nbits}_logs/')
        self.assertEqual(train_loss, train_loss_ref)

    def test_reactant2_network(self):
        """
        Tests the Reactant 2 Network.
        """
        embedding         = 'fp'
        radius            = 2
        nbits             = 4096
        out_dim           = 300
        batch_size        = 10
        epochs            = 2
        ncpu              = 2
        n_templates       = 3  # num templates in 'data/rxn_set_hb_test.txt'
        validation_option = 'nn_accuracy_gin'
        ref_dir           = f'./data/ref/'

        X = sparse.load_npz(ref_dir + 'X_rt2_train.npz')
        y = sparse.load_npz(ref_dir + 'y_rt2_train.npz')
        X = torch.Tensor(X.A)
        y = torch.Tensor(y.A)
        train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

        print("X shape", X.shape)
        print("y shape", y.shape)
        # use the train data for validation too (just for the unit tests)
        valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

        pl.seed_everything(0)
        mlp = MLP(input_dim=int(4 * nbits + n_templates),
                  output_dim=out_dim,
                  hidden_dim=100,
                  num_layers=3,
                  dropout=0.5,
                  num_dropout_layers=1,
                  task='regression',
                  loss='mse',
                  valid_loss=validation_option,
                  optimizer='adam',
                  learning_rate=1e-4,
                  val_freq=10,
                  ncpu=ncpu)

        tb_logger = pl_loggers.TensorBoardLogger(f'rt2_{embedding}_{radius}_{nbits}_logs/')
        trainer   = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=20, logger=tb_logger)
        trainer.fit(mlp, train_data_iter, valid_data_iter)

        train_loss     = float(trainer.callback_metrics["train_loss"])
        train_loss_ref = 0.4124651551246643

        shutil.rmtree(f'rt2_{embedding}_{radius}_{nbits}_logs/')
        self.assertEqual(train_loss, train_loss_ref)
