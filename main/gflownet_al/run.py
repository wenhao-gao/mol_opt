from __future__ import print_function

import torch
import gzip
import os
import pickle
import pdb
import threading
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from copy import deepcopy
from rdkit import Chem
from rdkit import DataStructs

from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
import model_atom, model_block, model_fingerprint
import os, sys 
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from main.gflownet_al.train_proxy import Dataset as _ProxyDataset
from main.gflownet_al.gflownet import Dataset as GenModelDataset
from main.optimizer import BaseOptimizer, Objdict


class ProxyDataset(_ProxyDataset):
    def add_samples(self, samples):
        for m in samples:
            self.train_mols.append(m)

tmp_dir = '/tmp/molexp/'
os.makedirs(tmp_dir, exist_ok=True)


class Proxy:
    def __init__(self, args, bpath, device):
        self.args = args
        # eargs = pickle.load(gzip.open(f'{args.proxy_path}/info.pkl.gz'))['args']
        # params = pickle.load(gzip.open(f'{args.proxy_path}/best_params.pkl.gz'))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, args.proxy_repr_type)
        self.mdp.floatX = torch.double
        self.proxy = make_model(args, self.mdp, is_proxy=True)
        # for a,b in zip(self.proxy.parameters(), params):
        #     a.data = torch.tensor(b, dtype=self.mdp.floatX)
        self.proxy.to(device)
        self.device = device

    def reset(self):
        for layer in self.proxy.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train(self, dataset):
        self.reset()
        stop_event = threading.Event()
        best_model = self.proxy
        best_test_loss = 1000
        opt = torch.optim.Adam(self.proxy.parameters(), self.args.proxy_learning_rate, betas=(self.args.proxy_opt_beta, 0.999),
                                weight_decay=self.args.proxy_weight_decay)
        debug_no_threads = False
        mbsize = self.args.mbsize

        if not debug_no_threads:
            sampler = dataset.start_samplers(8, mbsize)

        last_losses = []

        def stop_everything():
            stop_event.set()
            print('joining')
            dataset.stop_samplers_and_join()

        train_losses = []
        test_losses = []
        time_start = time.time()
        time_last_check = time.time()

        max_early_stop_tolerance = 5
        early_stop_tol = max_early_stop_tolerance

        for i in range(self.args.proxy_num_iterations+1):
            if not debug_no_threads:
                r = sampler()
                for thread in dataset.sampler_threads:
                    if thread.failed:
                        stop_event.set()
                        stop_everything()
                        pdb.post_mortem(thread.exception.__traceback__)
                s, r = r
            else:
                p, pb, a, r, s, d = dataset.sample2batch(dataset.sample(mbsize))

            # state outputs
            stem_out_s, mol_out_s = self.proxy(s, None, do_stems=False)
            loss = (mol_out_s[:, 0] - r).pow(2).mean()
            loss.backward()
            last_losses.append((loss.item(),))
            train_losses.append((loss.item(),))
            opt.step()
            opt.zero_grad()
            self.proxy.training_steps = i + 1

            if not i % 1000:
                last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
                print(i, last_losses)
                print('time:', time.time() - time_last_check)
                time_last_check = time.time()
                last_losses = []

                if i % 5000:
                    continue

                if 0:
                    continue

                t0 = time.time()
                total_test_loss = 0
                total_test_n = 0

                for s, r in dataset.itertest(max(mbsize, 128)):
                    with torch.no_grad():
                        stem_o, mol_o = self.proxy(s, None, do_stems=False)
                        loss = (mol_o[:, 0] - r).pow(2)
                        total_test_loss += loss.sum().item()
                        total_test_n += loss.shape[0]
                test_loss = total_test_loss / total_test_n
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_model = deepcopy(self.proxy)
                    best_model.to('cpu')
                    early_stop_tol = max_early_stop_tolerance
                else:
                    early_stop_tol -= 1
                    print('test loss:', test_loss)
                    print('test took:', time.time() - t0)
                    test_losses.append(test_loss)

        stop_everything()
        self.proxy = deepcopy(best_model)
        self.proxy.to(self.device)
        print('Done.')

    def __call__(self, m):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        return self.proxy(m, do_stems=False)[1].item()


def make_model(args, mdp, is_proxy=False):
    repr_type = args.proxy_repr_type if is_proxy else args.repr_type
    nemb = args.proxy_nemb if is_proxy else args.nemb
    num_conv_steps = args.proxy_num_conv_steps if is_proxy else args.num_conv_steps
    model_version = args.proxy_model_version if is_proxy else args.model_version
    if repr_type == 'block_graph':
        model = model_block.GraphAgent(nemb=nemb,
                                       nvec=0,
                                       out_per_stem=mdp.num_blocks,
                                       out_per_mol=1,
                                       num_conv_steps=num_conv_steps,
                                       mdp_cfg=mdp,
                                       version='v4')
    elif repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=1,
                                     num_conv_steps=num_conv_steps,
                                     version=model_version,
                                     dropout_rate=args.proxy_dropout)
    elif repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)
    return model


_stop = [None]
def train_generative_model(args, model, proxy, dataset, num_steps=None, do_save=True):
    debug_no_threads = False
    device = torch.device('cuda')

    if num_steps is None:
        num_steps = args.num_iterations + 1

    tau = args.bootstrap_tau
    if args.bootstrap_tau > 0:
        target_model = deepcopy(model)

    if do_save:
        exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
        os.makedirs(exp_dir, exist_ok=True)

    model = model.double()
    proxy.proxy = proxy.proxy.double()
    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)

    def save_stuff():
        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/params.pkl.gz', 'wb'))

        pickle.dump(dataset.sampled_mols,
                    gzip.open(f'{exp_dir}/sampled_mols.pkl.gz', 'wb'))

        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'test_infos': test_infos,
                     'time_start': time_start,
                     'time_now': time.time(),
                     'args': args,},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

        pickle.dump(train_infos,
                    gzip.open(f'{exp_dir}/train_info.pkl.gz', 'wb'))


    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay,
                           betas=(args.opt_beta, args.opt_beta2),
                           eps=args.opt_epsilon)

    tf = lambda x: torch.tensor(x, device=device).to(torch.float64 if args.floatX else torch.float32)

    mbsize = args.mbsize

    if not debug_no_threads:
        sampler = dataset.start_samplers(8, mbsize)

    last_losses = []

    def stop_everything():
        print('joining')
        dataset.stop_samplers_and_join()
    _stop[0] = stop_everything

    train_losses = []
    test_losses = []
    test_infos = []
    train_infos = []
    time_start = time.time()
    time_last_check = time.time()

    log_reg_c = args.log_reg_c
    clip_loss = tf([args.clip_loss])
    balanced_loss = args.balanced_loss
    do_nblocks_reg = False
    max_blocks = args.max_blocks
    leaf_coef = args.leaf_coef

    for i in range(num_steps):
        if not debug_no_threads:
            r = sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
                    return
            p, pb, a, r, s, d, mols = r
        else:
            p, pb, a, r, s, d, mols = dataset.sample2batch(dataset.sample(mbsize))
        # Since we sampled 'mbsize' trajectories, we're going to get
        # roughly mbsize * H (H is variable) transitions
        ntransitions = r.shape[0]
        # state outputs
        if tau > 0:
            with torch.no_grad():
                stem_out_s, mol_out_s = target_model(s, None)
        else:
            stem_out_s, mol_out_s = model(s, None)
        # parents of the state outputs
        stem_out_p, mol_out_p = model(p, None)
        # index parents by their corresponding actions
        qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[:, 0], a)
        # then sum the parents' contribution, this is the inflow
        exp_inflow = (torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
                      .index_add_(0, pb, torch.exp(qsa_p))) # pb is the parents' batch index
        inflow = torch.log(exp_inflow + log_reg_c)
        # sum the state's Q(s,a), this is the outflow
        exp_outflow = model.sum_output(s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        outflow_plus_r = torch.log(log_reg_c + r + exp_outflow * (1-d))
        if do_nblocks_reg:
            losses = _losses = ((inflow - outflow_plus_r) / (s.nblocks * max_blocks)).pow(2)
        else:
            losses = _losses = (inflow - outflow_plus_r).pow(2)
        if clip_loss > 0:
            ld = losses.detach()
            losses = losses / ld * torch.minimum(ld, clip_loss)

        term_loss = (losses * d).sum() / (d.sum() + 1e-20)
        flow_loss = (losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
        if balanced_loss:
            loss = term_loss * leaf_coef + flow_loss
        else:
            loss = losses.mean()
        opt.zero_grad()
        loss.backward(retain_graph=(not i % 50))

        _term_loss = (_losses * d).sum() / (d.sum() + 1e-20)
        _flow_loss = (_losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
        last_losses.append((loss.item(), term_loss.item(), flow_loss.item()))
        train_losses.append((loss.item(), _term_loss.item(), _flow_loss.item(),
                             term_loss.item(), flow_loss.item()))
        if not i % 50:
            train_infos.append((
                _term_loss.data.cpu().numpy(),
                _flow_loss.data.cpu().numpy(),
                exp_inflow.data.cpu().numpy(),
                exp_outflow.data.cpu().numpy(),
                r.data.cpu().numpy(),
                mols[1],
                [i.pow(2).sum().item() for i in model.parameters()],
                torch.autograd.grad(loss, qsa_p, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_s, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_p, retain_graph=True)[0].data.cpu().numpy(),
            ))
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(),
                                           args.clip_grad)
        opt.step()
        model.training_steps = i + 1
        if tau > 0:
            for _a,b in zip(model.parameters(), target_model.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)


        if not i % 100:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(i, last_losses)
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []

            if not i % 1000 and do_save:
                save_stuff()

    stop_everything()
    if do_save:
        save_stuff()
    return model, dataset, {'train_losses': train_losses,
                            'test_losses': test_losses,
                            'test_infos': test_infos,
                            'train_infos': train_infos}


def sample_and_update_dataset(args, model, proxy_dataset, generator_dataset, oracle):
    print("Sampling")
    mdp = generator_dataset.mdp
    nblocks = mdp.num_blocks
    sampled_mols = []
    rews = []
    smis = []
    while len(sampled_mols) < args.num_samples:
        mol = BlockMoleculeDataExtended()
        for i in range(args.max_blocks):
            s = mdp.mols2batch([mdp.mol2repr(mol)])
            stem_o, mol_o = model(s)
            logits = torch.cat([stem_o.flatten(), mol_o.flatten()])
            if i < args.min_blocks:
                logits[-1] = -20
            cat = Categorical(logits=logits)
            act = cat.sample().item()
            if act == logits.shape[0] - 1:
                break
            else:
                act = (act % nblocks, act // nblocks)
                mol = mdp.add_block_to(mol, block_idx=act[0], stem_idx=act[1])
            if not len(mol.stems):
                break
        if mol.mol is None:
            continue
        smis.append(mol.smiles)
        sampled_mols.append(mol)
    
    rews = oracle(sampled_mols)
    for i in range(len(sampled_mols)):
        sampled_mols[i].reward = rews[i]

    print("Computing distances")
    dists =[]
    for m1, m2 in zip(sampled_mols, sampled_mols[1:] + sampled_mols[:1]):
        dist = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(m1.mol), Chem.RDKFingerprint(m2.mol))
        dists.append(dist)
    print("Get batch rewards")
    rewards = []
    for m in sampled_mols:
        rewards.append(m.reward)
    print("Add to dataset")
    proxy_dataset.add_samples(sampled_mols)
    return proxy_dataset, rews, smis, {
        'dists': dists, 'rewards': rewards, 'reward_mean': np.mean(rewards), 'reward_max': np.max(rewards),
        'dists_mean': np.mean(dists), 'dists_sum': np.sum(dists)
    }


class GFlowNet_AL_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "gflownet_al"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        config = Objdict(config)
        bpath = os.path.join(path_here, 'data/blocks_PDB_105.json')
        device = torch.device('cuda')

        proxy_repr_type = config.proxy_repr_type
        repr_type = config.repr_type
        rews = []
        smis = []

        config.repr_type = proxy_repr_type
        config.replay_mode = "dataset"
        proxy_dataset = ProxyDataset(config, bpath, device, floatX=torch.float)

        # starting_examples = np.random.choice(self.all_smiles, config.num_init_examples)
        # starting_scores = self.oracle(starting_examples)

        # proxy_dataset
        dpath = os.path.join(path_here, 'data/docked_mols.h5')
        proxy_dataset.load_h5(dpath, config, num_examples=config.num_init_examples)


        rews.append(proxy_dataset.rews)
        smis.append([mol.smiles for mol in proxy_dataset.train_mols])
        rew_max = np.max(proxy_dataset.rews)
        print(np.max(proxy_dataset.rews))
        exp_dir = f'{config.save_path}/proxy_{config.array}_{config.run}/'
        os.makedirs(exp_dir, exist_ok=True)

        print(len(proxy_dataset.train_mols), 'train mols')
        print(len(proxy_dataset.test_mols), 'test mols')
        print(config)
        proxy = Proxy(config, bpath, device)
        mdp = proxy_dataset.mdp
        train_metrics = []
        metrics = []
        proxy.train(proxy_dataset)

        for i in range(config.num_outer_loop_iters):
            print(f"Starting step: {i}")
            # Initialize model and dataset for training generator
            config.sample_prob = 1
            config.repr_type = repr_type
            config.replay_mode = "online"
            gen_model_dataset = GenModelDataset(config, bpath, device)
            model = make_model(config, gen_model_dataset.mdp)

            if config.floatX == 'float64':
                model = model.double()
            model.to(device)
            # train model with with proxy
            print(f"Training model: {i}")
            model, gen_model_dataset, training_metrics = train_generative_model(config, model, proxy, gen_model_dataset, do_save=False)

            print(f"Sampling mols: {i}")
            # sample molecule batch for generator and update dataset with docking scores for sampled batch
            _proxy_dataset, r, s, batch_metrics = sample_and_update_dataset(config, model, proxy_dataset, gen_model_dataset, self.oracle)
            print(f"Batch Metrics: dists_mean: {batch_metrics['dists_mean']}, dists_sum: {batch_metrics['dists_sum']}, reward_mean: {batch_metrics['reward_mean']}, reward_max: {batch_metrics['reward_max']}")
            rews.append(r)
            smis.append(s)
            config.sample_prob = 0
            config.repr_type = proxy_repr_type
            config.replay_mode = "dataset"
            config.reward_exp = 1
            config.reward_norm = 1

            train_metrics.append(training_metrics)
            metrics.append(batch_metrics)

            proxy_dataset = ProxyDataset(config, bpath, device, floatX=torch.float)
            proxy_dataset.train_mols.extend(_proxy_dataset.train_mols)
            proxy_dataset.test_mols.extend(_proxy_dataset.test_mols)

            proxy = Proxy(config, bpath, device)
            mdp = proxy_dataset.mdp

            pickle.dump({'train_metrics': train_metrics,
                        'batch_metrics': metrics,
                        'rews': rews,
                        'smis': smis,
                        'rew_max': rew_max,
                        'args': config},
                        gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

            print(f"Updating proxy: {i}")
            # update proxy with new data
            proxy.train(proxy_dataset)
