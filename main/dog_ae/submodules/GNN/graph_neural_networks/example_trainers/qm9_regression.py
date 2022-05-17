
import abc
import time
from os import path

import tqdm
import numpy as np
import torch
from torch.utils import data
from torch import nn
from torch import optim

from ..core import utils
from ..datasets import qm9
from ..datasets import loader


DATA_DETAILS = {
    'CHEM_ACC': 0.066513725,
    'train_data_path':  path.join(loader.get_qm9_data_path(), "molecules_train.json"),
    'valid_data_path':  path.join(loader.get_qm9_data_path(), "molecules_valid.json"),
}


class ExperimentParams:
    def __init__(self,
                 learning_rate=1e-4,
                 num_epochs=100,
                 batch_size_train=64,
                 batch_size_val=20,
                 T=5,
                 hidden_layer_size=100,
                 num_workers=0,
                 ):

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.T = T
        self.hidden_layer_size = hidden_layer_size
        self.num_workers = num_workers
        self.edge_names = [f"edge_{k}" for k in range(1, 5)]
        self.edge_names_as_ints = [k for k in range(1, 5)]


class ExperimentParts(metaclass=abc.ABCMeta):
    """
    This is what you need to implement for each particular variant.
    """
    def __init__(self, exp_params: ExperimentParams):
        self.exp_params = exp_params

    @abc.abstractmethod
    def create_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def create_transform(self):
        raise NotImplementedError

    @abc.abstractmethod
    def create_collate_function(self):
        raise NotImplementedError

    @abc.abstractmethod
    def data_split_and_cudify_func(self, data):
        """
        data from the dataloader
        :return: (tuple that will be unpacked into model), targets
        """
        raise NotImplementedError


def validate(validation_data_loader, model, exp_parts: ExperimentParts):
    model.eval()

    losses = 0.0
    num_done = 0
    criterion = nn.L1Loss(reduction='sum')
    with torch.no_grad():
        for i, (data) in enumerate(validation_data_loader):
            model_input, targets = exp_parts.data_split_and_cudify_func(data)

            outputs = model(*model_input)
            losses = losses + criterion(outputs, targets).item()
            num_done = num_done + outputs.shape[0]

        loss_no_adj = losses / (num_done)
        loss_adj = loss_no_adj / DATA_DETAILS['CHEM_ACC']

    print(f"Validation run completed. Avg loss is {loss_no_adj} (chemically accuracy normalized version is {loss_adj})")
    print(f"V__{loss_no_adj:.5f}__{loss_adj:.5f}")  # to make it easy to regexp this info too!
    return loss_no_adj, loss_adj


def train(train_dataloader, model, exp_parts: ExperimentParts, optimizer, criterion):
    model.train()

    loss_meter = utils.AverageMeter()
    time_meter = utils.AverageMeter()
    time_on_calc = utils.AverageMeter()

    pre_time = time.time()
    with tqdm.tqdm(train_dataloader, total=np.ceil(len(train_dataloader.dataset) / train_dataloader.batch_size)) as t:
        for i, (data) in enumerate(t):
            # Set up the data
            model_input, targets = exp_parts.data_split_and_cudify_func(data)
            pre_calc_time = time.time()
            prediction = model(*model_input)

            # Compute the loss
            loss = criterion(input=prediction, target=targets)

            # Update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the meters that record the various statistics.
            loss_meter.update(loss.item())
            time_meter.update(time.time() - pre_time)
            time_on_calc.update(time.time() - pre_calc_time)
            pre_time = time.time()

            # Update the stats in the progress bar
            t.set_postfix(avg_epoch_loss=f'{loss_meter.avg:.4E}',
                          total_time=f'{time_meter.avg:.3E}', calc_time=f'{time_on_calc.avg:.3E}')


def main_runner(exp_parts: ExperimentParts):
    torch.manual_seed(56145612)

    # === Deal with the data! ===
    trsfm = exp_parts.create_transform()

    qm9_train = qm9.Qm9Dataset(DATA_DETAILS['train_data_path'], trsfm)
    qm9_val = qm9.Qm9Dataset(DATA_DETAILS['valid_data_path'], trsfm)

    collate_fn = exp_parts.create_collate_function()
    qm9_dataloader_train = data.DataLoader(qm9_train, batch_size=exp_parts.exp_params.batch_size_train, shuffle=True,
                                           num_workers=exp_parts.exp_params.num_workers, collate_fn=collate_fn)
    qm9_dataloader_val = data.DataLoader(qm9_val, batch_size=exp_parts.exp_params.batch_size_val, shuffle=False,
                                           num_workers=exp_parts.exp_params.num_workers, collate_fn=collate_fn)

    # === Set up the model ===
    ggnn = exp_parts.create_model()
    if torch.cuda.is_available():
        ggnn = ggnn.cuda()

    # === Set up loss and the trainer ===
    criterion = nn.MSELoss()
    optimiser = optim.Adam(ggnn.parameters(), lr=exp_parts.exp_params.learning_rate)

    # === Train and evaluate! ===
    print("running initial validation run...")
    s_time = time.time()
    validate(qm9_dataloader_val, ggnn, exp_parts)
    print(f"Initial validation time is {time.time() - s_time}")

    best_val_loss = np.inf

    for epoch_num in range(exp_parts.exp_params.num_epochs):
        print(f"\n\n === Starting Epoch {epoch_num} ===")
        train(qm9_dataloader_train, ggnn, exp_parts, optimiser, criterion)
        val_loss, *_ = validate(qm9_dataloader_val, ggnn, exp_parts)

        # Save
        best_flag = val_loss < best_val_loss
        if best_flag:
            best_val_loss = val_loss
        torch.save({
                'num_epochs_completed': epoch_num + 1,
                'state_dict': ggnn.state_dict(),
                'optimizer': optimiser.state_dict(),
            }, f"chkpts/qm9_mu_{str(type(exp_parts).__name__)}_epochs_done_{epoch_num + 1}")
