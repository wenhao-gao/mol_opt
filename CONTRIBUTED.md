# Contribution Guide

Thanks for your interest in contributing to our benchmark. 

## 1) Create a directory for new method XXX 

```bash
mkdir main/XXX
```

`main/XXX/run.py` is the main file. 

```bash
cd main/XXX/
python run.py 
python run.py --population_size 100 
```





## 2) BaseOptimizer (provided)

```python
class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args
        self.n_jobs = args.n_jobs
        self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.mol_buffer = {}
        if self.smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:
            data = MolGen(name = 'ZINC')
            self.all_smiles = data.get_data()['smiles'].tolist()
            
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

    def load_smiles_from_file(self, file_name):
        ...
            
    def score_mol(self, oracle_func, mol_list):
    	... 
```


## 3) New class inherited from BaseOptimizer 


```python

class XXX_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "graph_ga"

    def _optimize(self, oracle, config):
        ...

        population_size = config["population_size"]

        for it in range(iterations):

        	...

	        ### automatically save results 
    	    population_scores = self.score_mol(oracle, population_mol) 

        	### logging intermediate
        	self.log_intermediate(population_mol, population_scores)

```

## 4) main 

* hyperparameter
	* **running hyperparameter**: parser argument. 
	* **default model hyperparameter**: `hparam_default.yaml`
	* **tuning model hyperparameter**: `hparam_tune.yaml` 
* run optimizer: `simple`, `tune`, `production`.   


```python
def main():
    # 1. hyperparameter 
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = path_here
    elif not os.path.exist(args.output_dir):
        os.mkdir(args.output_dir)


    # 2. run optimizer 
    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(args.output_dir, args.config_default)))

        if args.task == "tune":
            try:
                config_tune = yaml.safe_load(open(args.config_tune))
            except:
                config_tune = yaml.safe_load(open(os.path.join(args.output_dir, args.config_tune)))

        oracle = Oracle(name = oracle_name)
        optimizer = GB_GA_Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main()
```

