import gzip
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tqdm import tqdm
import torch
from mol_mdp_ext import BlockMoleculeDataExtended
tmp_dir = "/tmp/molexp"
os.makedirs(tmp_dir, exist_ok=True)
from main.gflownet_al.gflownet import Dataset as _Dataset


class Dataset(_Dataset):

    def _get(self, i, dset):
        return [(dset[i], dset[i].reward)]

    def itertest(self, n):
        N = len(self.test_mols)
        for i in range(int(np.ceil(N/n))):
            samples = sum((self._get(j, self.test_mols) for j in range(i*n, min(N, (i+1)*n))), [])
            yield self.sample2batch(zip(*samples))

    def sample2batch(self, mb):
        s, r, *o = mb
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        r = torch.tensor(r, device=self._device).float()
        return (s, r, *o)

    def load_h5(self, path, args, test_ratio=0.1, num_examples=None):
        import json
        import pandas as pd
        columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
        store = pd.HDFStore(path, 'r')
        df = store.select('df')
        # Pandas has problem with calculating some stuff on float16
        df.dockscore = df.dockscore.astype("float64")
        for cl_mame in columns[2:]:
            df.loc[:, cl_mame] = df[cl_mame].apply(json.loads)

        test_idxs = self.test_split_rng.choice(len(df), int(test_ratio * len(df)), replace=False)

        split_bool = np.zeros(len(df), dtype=np.bool)
        split_bool[test_idxs] = True
        print("split test", sum(split_bool), len(split_bool), "num examples", num_examples)
        self.rews = []
        for i in tqdm(range(len(df)), disable=not args.progress):
            m = BlockMoleculeDataExtended()
            for c in range(1, len(columns)):
                setattr(m, columns[c], df.iloc[i, c - 1])
            m.blocks = [self.mdp.block_mols[i] for i in m.blockidxs]
            if len(m.blocks) > self.max_blocks:
                continue
            m.reward = self.r2r(dockscore=m.dockscore)
            m.numblocks = len(m.blocks)
            if split_bool[i]:
                self.test_mols.append(m)
            else:
                self.rews.append(m.reward)
                self.train_mols.append(m)
                self.train_mols_map[df.iloc[i].name] = m
            if len(self.train_mols) >= num_examples:
                break
        store.close()

    def load_pkl(self, path, args, test_ratio=0.05, num_examples=None):
        columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
        mols = pickle.load(gzip.open(path))
        if num_examples is None:
            num_examples = len(mols)
            idxs = range(len(mols))
        else:
            idxs = self.test_split_rng.choice(len(mols), int((1 - test_ratio) * num_examples), replace=False)
        test_idxs = self.test_split_rng.choice(len(mols), int(test_ratio * num_examples), replace=False)
        split_bool = np.zeros(len(mols), dtype=np.bool)
        split_bool[test_idxs] = True
        for i in tqdm(idxs, disable=not args.progress):
            m = BlockMoleculeDataExtended()
            for c in range(1, len(columns)):
                setattr(m, columns[c], mols[i][columns[c]])
            m.blocks = [self.mdp.block_mols[i] for i in m.blockidxs]
            if len(m.blocks) > self.max_blocks:
                continue
            m.reward = self.r2r(dockscore=m.dockscore)
            m.numblocks = len(m.blocks)
            if split_bool[i]:
                self.test_mols.append(m)
            else:
                self.train_mols.append(m)
                self.train_mols_map[m.smiles if len(m.blocks) else '[]'] = m

