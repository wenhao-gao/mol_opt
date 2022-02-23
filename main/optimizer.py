import json
import os
import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem
import tdc
from tdc.generation import MolGen
import wandb


class BaseOptimizer:

    def __init__(self, 
                 config=None,
                 args=None,
                 smi_file=None,
                 n_jobs=-1):
        self.n_jobs = n_jobs
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.mol_buffer = {}
        if smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:
            data = MolGen(name = 'ZINC')
            self.all_smiles = data.get_data()['smiles'].tolist()
            
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)
            
    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print('bad smiles')
        return new_mol_list
        
    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))
            
    def score_mol(self, oracle_func, mol_list):
        score_list = []
        for mol in mol_list:
            smi = Chem.MolToSmiles(mol)
            try:
                _ = self.mol_buffer[smi]
                score_list.append(_[0])
            except:
                score = oracle_func(smi)
                self.mol_buffer[smi] = (score, len(self.mol_buffer)+1)
                score_list.append(score)
                
        self.sort_buffer()
        return score_list
    
    def log_intermediate(self, mols, scores):
        smis = [Chem.MolToSmiles(m) for m in mols]
        temp_top10 = list(self.mol_buffer.items())[:10]
#             img = Chem.Draw.MolsToGridImage([Chem.MolFromSmiles(item[0]) for item in temp], 
#                           molsPerRow=5, subImgSize=(200,200), 
#                           legends=[f"f = {item[1][0]:.3f}, #oracle = {item[1][1]}" for item in temp])
#             img.save("temp.png")

        wandb.log({
            "avg_top1": np.max(scores), 
            "avg_top10": np.mean(sorted(scores, reverse=True)[:10]), 
            "avg_top100": np.mean(scores), 
            "avg_sa": np.mean(self.sa_scorer(smis)),
            "diversity_top100": self.diversity_evaluator(smis),
            "n_oracle": len(self.mol_buffer),
            # "best_mol": wandb.Image(Chem.Draw.MolsToGridImage([Chem.MolFromSmiles(item[0]) for item in temp_top10], 
            #           molsPerRow=5, subImgSize=(200,200), legends=[f"f = {item[1][0]:.3f}, #oracle = {item[1][1]}" for item in temp]))
        })
    
    def log_result(self):
        
        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        
        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]
        
        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))
        
        # Currently logging the top-100 moelcules, will update to PDD selection later
        data = [[i+1, results_all_level[-1][i][1][0], results_all_level[-1][i][1][1], results_all_level[-1][i][0]] for i in range(100)]
        columns = ["Rank", "Score", "#Oracle", "SMILES"]
        wandb.log({"Top 100 Molecules": wandb.Table(data=data, columns=columns)})
        
        # Log batch metrics at various oracle calls
        data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]
        wandb.log({"Batch metrics at various level": wandb.Table(data=data, columns=columns)})
        
    def save_result(self):
        pass
    
    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        return [np.mean(scores), 
                 np.mean(scores[:10]), 
                 np.max(scores), 
                 self.diversity_evaluator(smis), 
                 np.mean(self.sa_scorer(smis)), 
                 float(len(smis_pass) / 100), 
                 np.max([scores_dict[s] for s in smis_pass])]
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
    def hparam_tune(self, oracle, hparam_space, count, seed=0):
        wandb.init()
        sweep_id = wandb.sweep(hparam_space)
        np.random.seed(seed)
        
        def _func():
            with wandb.init() as run:
                self._optimize(oracle, wandb.config)
            self.mol_buffer = {}

        wandb.agent(sweep_id, function=_func, count=count)
        
    def optimize(self, oracle, config, seed=0):
        wandb.init(config=config)
        np.random.seed(seed)
        self._optimize(oracle, config)
        self.log_result()

