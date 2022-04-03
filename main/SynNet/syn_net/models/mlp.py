"""
Multi-layer perceptron (MLP) class.
"""
import time
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from sklearn.neighbors import BallTree
import numpy as np


class MLP(pl.LightningModule):

    def __init__(self, input_dim=3072,
                 output_dim=4,
                 hidden_dim=1000,
                 num_layers=5,
                 dropout=0.5,
                 num_dropout_layers=1,
                 task='classification',
                 loss='cross_entropy',
                 valid_loss='accuracy',
                 optimizer='adam',
                 learning_rate=1e-4,
                 val_freq=10,
                 ncpu=16):
        super().__init__()

        self.loss = loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.ncpu = ncpu
        self.val_freq = val_freq

        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.BatchNorm1d(hidden_dim))
        modules.append(nn.ReLU())

        for i in range(num_layers-2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.BatchNorm1d(hidden_dim))
            modules.append(nn.ReLU())
            if i > num_layers - 3 - num_dropout_layers:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_dim, output_dim))
        if task == 'classification':
            modules.append(nn.Softmax())

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x)
        if self.loss == 'cross_entropy':
            loss = F.cross_entropy(y_hat, y)
        elif self.loss == 'mse':
            loss = F.mse_loss(y_hat, y)
        elif self.loss == 'l1':
            loss = F.l1_loss(y_hat, y)
        elif self.loss == 'huber':
            loss = F.huber_loss(y_hat, y)
        else:
            raise ValueError('Not specified loss function')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.current_epoch % self.val_freq == 0:
            out_feat = self.valid_loss[12:]
            if out_feat == 'gin':
                bb_emb_gin = np.load('/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_gin.npy')
                kdtree = BallTree(bb_emb_gin, metric='euclidean')
            elif out_feat == 'fp_4096':
                bb_emb_fp_4096 = np.load('/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_4096.npy')
                kdtree = BallTree(bb_emb_fp_4096, metric='euclidean')
            elif out_feat == 'fp_256':
                bb_emb_fp_256 = np.load('/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_256.npy')
                kdtree = BallTree(bb_emb_fp_256, metric=cosine_distance)
            elif out_feat == 'rdkit2d':
                bb_emb_rdkit2d = np.load('/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_rdkit2d.npy')
                kdtree = BallTree(bb_emb_rdkit2d, metric='euclidean')
            x, y = batch
            y_hat = self.layers(x)
            if self.valid_loss == 'cross_entropy':
                loss = F.cross_entropy(y_hat, y)
            elif self.valid_loss == 'accuracy':
                y_hat = torch.argmax(y_hat, axis=1)
                loss = 1 - (sum(y_hat == y) / len(y))
            elif self.valid_loss[:11] == 'nn_accuracy':
                y = nn_search_list(y.detach().cpu().numpy(), out_feat=out_feat, kdtree=kdtree)
                y_hat = nn_search_list(y_hat.detach().cpu().numpy(), out_feat=out_feat, kdtree=kdtree)
                loss = 1 - (sum(y_hat == y) / len(y))
                # import ipdb; ipdb.set_trace(context=11)
            elif self.valid_loss == 'mse':
                loss = F.mse_loss(y_hat, y)
            elif self.valid_loss == 'l1':
                loss = F.l1_loss(y_hat, y)
            elif self.valid_loss == 'huber':
                loss = F.huber_loss(y_hat, y)
            else:
                raise ValueError('Not specified validation loss function')
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            pass

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

def load_array(data_arrays, batch_size, is_train=True, ncpu=-1):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, num_workers=ncpu)

def cosine_distance(v1, v2, eps=1e-15):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2) + eps)

def nn_search(_e, _tree, _k=1):
    dist, ind = _tree.query(_e, k=_k)
    return ind[0][0]

def nn_search_list(y, out_feat, kdtree):
    if out_feat == 'gin':
        return np.array([nn_search(emb.reshape(1, -1), _tree=kdtree) for emb in y])
    elif out_feat == 'fp_4096':
        return np.array([nn_search(emb.reshape(1, -1), _tree=kdtree) for emb in y])
    elif out_feat == 'fp_256':
        return np.array([nn_search(emb.reshape(1, -1), _tree=kdtree) for emb in y])
    elif out_feat == 'rdkit2d':
        return np.array([nn_search(emb.reshape(1, -1), _tree=kdtree) for emb in y])
    else:
        raise ValueError


if __name__ == '__main__':

    states_list = []
    steps_list = []
    for i in range(1):
        states_list.append(np.load('/home/rociomer/data/synth_net/pis_fp/states_' + str(i) + '_valid.npz', allow_pickle=True))
        steps_list.append(np.load('/home/rociomer/data/synth_net/pis_fp/steps_' + str(i) + '_valid.npz', allow_pickle=True))

    states = np.concatenate(states_list, axis=0)
    steps = np.concatenate(steps_list, axis=0)

    X = states
    y = steps[:, 0]

    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(y)

    batch_size = 64
    train_data_iter = load_array((X_train, y_train), batch_size, is_train=True)

    pl.seed_everything(0)
    mlp = MLP()
    tb_logger = pl_loggers.TensorBoardLogger('temp_logs/')

    trainer = pl.Trainer(gpus=[0], max_epochs=30, progress_bar_refresh_rate=20, logger=tb_logger)
    t = time.time()
    trainer.fit(mlp, train_data_iter, train_data_iter)
    print(time.time() - t, 's')
