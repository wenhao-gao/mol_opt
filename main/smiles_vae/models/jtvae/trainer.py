import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.optim.lr_scheduler as lr_scheduler

from tqdm.auto import tqdm
import numpy as np

from models.trainer import Trainer
from models.jtvae.mol_tree import MolTree
from utils.jtvae_data_utils import MolTreeFolder
from utils.vocab import Vocab


class JTVAETrainer(Trainer):
    def __init__(self, config):
        self.config = config

    def get_vocabulary(self, _):
        vocab = [x.strip("\r\n ") for x in open(self.config.vocab_load)]
        self.vocab = Vocab(vocab)
        return self.vocab

    def save_vocabulary(self, vocab):
        torch.save(vocab, self.config.vocab_save)
        return None

    def _train_epoch(self, model, epoch, total_step, tqdm_data, optimizer=None):

        if optimizer is None:
            model.eval()
        else:
            model.train()

        loss_list = []
        meters = np.zeros(4)
        for batch in tqdm_data:

            total_step += 1

            try:
                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc = model(batch, self.beta)
                loss.backward()
                loss_list.append(loss.item())
                nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_norm)
                optimizer.step()
            except Exception as e:
                print(e)
                continue

            meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

            postfix = [f'loss={loss:.3f}',
                       f'kl={meters[0]:.2f}',
                       f'Word={meters[1]:.2f}',
                       f'Topo={meters[2]:.2f}',
                       f'Word={meters[3]:.2f}']
            tqdm_data.set_postfix_str(' '.join(postfix))
            meters *= 0

        postfix = {
            'epoch': epoch,
            'loss': np.array(loss_list).mean(),
            'mode': 'Eval' if optimizer is None else 'Train'}

        return postfix, total_step

    def _train(self, model, train_loader, val_loader=None, logger=None):
        device = model.device

        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.config.anneal_rate)

        # param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
        # grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None])

        model.zero_grad()
        total_step = 0
        self.beta = self.config.beta
        for epoch in range(self.config.epoch):
            train_loader = MolTreeFolder(self.config.data_path, self.vocab, self.config.batch_size, num_workers=4)
            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))
            postfix, total_step = self._train_epoch(model, epoch, total_step, 
                                        tqdm_data, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, epoch, tqdm_data)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(),
                           self.config.model_save[:-3] +
                           '_{0:03d}.pt'.format(epoch))
                model = model.to(device)

            if epoch % self.config.anneal_iter == 0:
                scheduler.step()

            if epoch % self.config.kl_anneal_iter == 0 and epoch >= self.config.warmup:
                self.beta = min(self.config.max_beta, self.beta + self.config.step_beta)

    def fit(self, model, train_data, val_data=None):
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        if self.config.load_epoch > 0:
            model.load_state_dict(torch.load(self.config.save_dir + "/model.iter-" + str(self.config.load_epoch)))

        print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
        self._train(model, train_data, val_data)
        return model

    def get_optim_params(self, model):
        return (p for p in model.vae.parameters() if p.requires_grad)

    def load_train_data(self):
        return self.config.data_path

    def load_val_data(self):
        return None


